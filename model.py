import os
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ------------------------------------------------------------
# Config

@dataclass
class GPTConfig:
    block_size: int  = 1024  # max context/sequence length
    vocab_size: int  = 50257 # number of tokens: 256 bytes tokens, 1 EoT token, and 50,000 BPE merges
    n_layer: int     = 12    # number of layers
    n_head: int      = 12    # number of attn heads 
    n_kv_head: int   = 12    # set to 1 for MQA; <n_head for GQA
    n_embd: int      = 768   # embedding dimension
    dropout: float   = 0.0   # percentage of neurons dropped out
    bias: bool       = False  

    # --- MoE ---
    use_moe: bool    = True
    n_experts: int   = 8      # number of expert MLPs
    top_k: int       = 2      # top-k routing

    # --- RMSNorm ---
    norm_eps: float = 1e-5
    use_rmsnorm: bool = True

# ------------------------------------------------------------
# Model

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k

        # router: d_model -> n_experts
        self.router = nn.Linear(config.n_embd, self.n_experts, bias=config.bias)

        # expert MLPs
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.n_experts)])

    def forward(self, x):
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        logits = self.router(x_flat)                # (N, E)
        probs  = torch.softmax(logits, dim=-1)      # (N, E)

        topw, topi = torch.topk(probs, self.top_k, dim=-1)  # (N, K)
        topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-9)

        top1 = topi[:, 0]  # (N,)
        load = torch.bincount(top1, minlength=self.n_experts).float() / top1.numel()  # (E,)
        importance = probs.mean(dim=0)  # (E,)

        # assess the overall importance and load of expert use
        # if importance not uniform, push it towards uniform
        # if load not uniform, push it towards uniform
        l_aux = self.n_experts * torch.sum(load * importance)

        # assess the overall size of the logits
        # penalize large vals
        z_loss = (logits.logsumexp(dim=-1) ** 2).mean()

        # tk_id | exp_id | weight | embd
        # tk_id[0] -> [exp_id[0], weight[0], embd[0]]
        # if tk_id[0] has more than one expert
        # tk_id[0] -> [exp_id[1], weight[1], embd[1]]
        token_ids  = torch.arange(N, device=x.device).repeat_interleave(self.top_k)  # (N*K,)
        expert_ids = topi.reshape(-1)                                                # (N*K,)
        weights    = topw.reshape(-1)                                                # (N*K,)
        x_rep      = x_flat.repeat_interleave(self.top_k, dim=0)                     # (N*K, C)

        # rearrange from tk_id order to exp_id order
        # tk_id[0] -> [exp_id[0], weight[0], embd[0]]
        # to exp_id[0] -> [tk_id[0], weight[0], embd[0]]
        order = torch.argsort(expert_ids)
        token_ids  = token_ids[order]
        expert_ids = expert_ids[order]
        weights    = weights[order]
        x_rep      = x_rep[order]


        # how many counts for each expert
        counts = torch.bincount(expert_ids, minlength=self.n_experts) # (E,) on GPU

        # only sync once to reduce CPU comms
        # resolves not using .item() in a loop to gather each experts tokens
        counts_list = counts.tolist()

        # split each into their respected experts
        x_chunks   = torch.split(x_rep,     counts_list, dim=0)
        w_chunks   = torch.split(weights,   counts_list, dim=0)
        tok_chunks = torch.split(token_ids, counts_list, dim=0)

        # accum each tk final output
        y_flat = torch.zeros_like(x_flat)  # (N, C)

        for e in range(self.n_experts):
            if counts_list[e] == 0:
                continue

            y_e = self.experts[e](x_chunks[e])               # (cnt, C)
            y_e = y_e * w_chunks[e].unsqueeze(-1)            # (cnt, 1)
            y_flat.index_add_(0, tok_chunks[e], y_e)
        
        y = y_flat.view(B, T, C)

        return y, l_aux, z_loss


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # check if the number of embeddings is a multiple of the number of heads
        # ensures that the embeddings can be properly split to each attn head
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        # initialize number of heads and embeddings
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # for GQA/MQA
        self.n_kv_head = config.n_kv_head
        self.head_dim  = config.n_embd // config.n_head

        # Q has n_heads, KV have n_kv_heads
        self.c_q   = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_kv  = nn.Linear(config.n_embd, 2 * self.n_kv_head * self.head_dim, bias=False)

        # project to 3 for query, key, values
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)

        # seems pointless but important to be used to mix info from concatenated heads
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # tag for scaling down the variance of the residual stream
        self.c_proj.SCALE_INIT = 1

        # added dropout for regularization
        self.resid_dropout = nn.Dropout(config.dropout)

        # -------- only needed if flash attention isn't used or is_causal=False
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, past_kv=None): # if not None, then we have T=1
        # get all dimensions
        B, T, C = x.size()

        # using GQA/MQA
        # get query, key, values
        q  = self.c_q(x)    # (B, T, n_head*hd)
        kv = self.c_kv(x)   # (B, T, 2*n_kv_head*hd)
        k, v = kv.split(self.n_kv_head * self.head_dim, dim=2)

        # w/o GQA/MQA
        # qkv = self.c_attn(x)
        # get three B, T, C, split along dim=2 (C)
        # q, k, v = qkv.split(self.n_embd, dim=2) # split into size of self.n_embd along the 2nd dim

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)     # (B, n_head, T, hd)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, hd)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, hd)

        # --- KV cache ---
        # append new k,v to cached past_k, past_v
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        kv = (k, v)

        # make KV heads match Q heads for attention
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)  # (B, n_head, T_total, hd)
            v = v.repeat_interleave(repeat, dim=1)  # (B, n_head, T_total, hd)

        
        # ------- flash attention (need CUDA)
        # if no CUDA, then regular PyTorch attention w/o efficient kernels
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.config.dropout)
        
        # concat all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 'contiguous' since transpose alone dn lay it into memory but view() needs it to be

        # mixes info from all head outputs and adds dropout
        y = self.resid_dropout(self.c_proj(y))
        return y, kv


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        # adjust for parameter count when using SwiGLU
        # GELU MLP 4x expansion has 8d**2 parameters
        # SwiGLU has 3dh parameters
        # therefore, 3dh = 8d**2 -> h = 8/3d
        hidden_dim = (8 * config.n_embd) // 3
        # systems trick for faster kernels
        # multiples of 64 used since smaller model
        hidden_dim = (hidden_dim + 63) // 64 * 64

        self.fc    = nn.Linear(config.n_embd, 2 * hidden_dim, bias=config.bias)

        # back from the 4x projection down to size of n_embd
        self.proj  = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)

        # tag for scaling down the variance of the residual stream
        self.proj.SCALE_INIT = 1

        # regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # (B, T, C) -> (B, T, 2H)
        x = self.fc(x)

        # went to 2 * hidden_dim to chunk
        x_gate, x_val = x.chunk(2, dim=-1)

        # SwiGLU = SiLU(gate) * value
        x = F.silu(x_gate) * x_val

        x = self.proj(x)
        x = self.dropout(x)
        return x

# GPT-2 transformer blocks
# different from the original transformer in GPT-2 because layernorm is added before attn as well
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MoE(config) if getattr(config, "use_moe", False) else MLP(config)

    
    def forward(self, x, past_kv=None):
        attn_out, kv = self.attn(self.ln_1(x), past_kv=past_kv)
        # residual stream + attn
        x = x + attn_out

        h = self.ln_2(x)
        
        # when there is no aux/z losses
        l_aux = x.new_zeros(())
        z_loss = x.new_zeros(())

        if isinstance(self.mlp, MoE):
            y, l_aux, z_loss = self.mlp(h)
        else:
            y = self.mlp(h)
        
        # residual stream + mlp
        x = x + y

        return x, kv, l_aux, z_loss
    
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ModuleDict = dict but with inherited nn.Module
        self.transformer = nn.ModuleDict(dict(

            # gives each token from the vocab an embedding of size n_embd
            wte  = nn.Embedding(config.vocab_size, config.n_embd), 
            # gives each position an embedding of size n_embd
            wpe  = nn.Embedding(config.block_size, config.n_embd),

            # ModuleList = list but with inherited nn.Module
            # creates n_layer amount of attn Blocks
            h    = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            # after the attn block mlp + residual stream, final layernorm
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        # projects the embeddings up to the vocab size for classification
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # GPT-2 does not use bias

        # tied weights:
        # to reduce weight number and improve consistency with the representation space
        self.transformer.wte.weight = self.lm_head.weight

        # initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # NOTE: Layernorm has proper default initialization w/ scale set to 1 and bias set to 0
        # so nothing extra is required here; we keep it defaulted
        # but for nn.Linear, linear weight initialization are set to a uniform distribution
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # so initializate weights according to GPT-2 -> normal dist w/ 0.02 std 

        # NOTE: 0.02 std is roughly around the Xavier initialization so no need for 1/sqrt(d_model)
        if isinstance(module, nn.Linear):
            std = 0.02 

            # need to control the activation growth of the residual stream
            # so we scale the weights down of the activations at the end of each block
            # hence, assign an attribute that acts as a tag for scaling
            # also, 2 * n_layer is done because each block adds 2 residual contributions (attn and mlp)
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            # set bias to zero instead of the pytorch defaulted uniform dist
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) 

        # set std to 0.02 instead of 1 
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, idx, targets=None, past_kv=None):

        # idx of shape (B, T) -> batch size by sequence length 
        B, T = idx.size()

        # --- KV cache
        if past_kv is None:
            # init the whole kv cache layers as None of size n_layer
            past_kv = [None] * self.config.n_layer
            past_len = 0
        else:
            # the cache length is stored in k
            # past_kv[0] is layer0, [0] is k, size(-2) is seq length
            past_len = past_kv[0][0].size(-2)


        # KV Cache: add past_len now 
        # ensure that the sequence length is not bigger than context 
        assert past_len + T <= self.config.block_size, f"Cannot forward sequence of length {past_len + T}"

        # KV cache: positions now have to be offset by past_len during decode
        # position of each token up to the sequence length, T
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)

        # input the positions to get their embeddings
        pos_emb = self.transformer.wpe(pos)

        # input the tokens to get their embeddings
        tok_emb = self.transformer.wte(idx)

        # combine the token embeddings and the positional embeddings
        x = tok_emb + pos_emb

        # for if MoE isn't used
        total_l_aux = x.new_zeros(())
        total_z_loss = x.new_zeros(())

        # KV cache: now require to pass each layer's respected past_kv[i]
        kv = []
        # pass through the 12 blocks, each w/ their layernorms, attn, and mlp
        for i,block in enumerate(self.transformer.h):
            x, pkv, l_aux, z_loss = block(x, past_kv=past_kv[i])
            kv.append(pkv)
            total_l_aux = total_l_aux + l_aux
            total_z_loss = total_z_loss + z_loss

        # pass through the layernorm after the attn mlp's
        x = self.transformer.ln_f(x)

        # get the logits via linear layer, i.e., from embedding size transformed to vocab size
        logits = self.lm_head(x) # (B, T, 50257)


        alpha = 0.01
        beta  = 1e-3

        loss = None
        # for training
        if targets is not None:
            # takes (B*T, V), the logits, and (B*T), the targets, as input to calculate the loss
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss + alpha * total_l_aux + beta * total_z_loss

        return logits, loss, kv
    
    # regularize weights
    # goal: pull down weights so the network finds a solution that doesn't involve large weights
    # improves generalization; prevents overfitting
    def configure_optimizers(self, weight_decay, learning_rate, device, master_process):

        # 'param_dict' returns a dict containing only those parameters that has a gradient
        # goal: want only the trainable parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}               # gets an iterator of (name, param) from the model
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # filters out 'frozen' params 

        # split the params into those that will be weight decayed and those that will not be
        # goal: want to weight decay matrices; don't want to weight decay biases or 1D tensors (doesn't make sense to decay single biases or the scale/shift in layernorm)
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]  # matrices containing params
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2] # 1D tensors: layernorm, biases

        # pass into AdamW to tune only the 'decay_params' with weight decay
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # sanity check
        # get the total number of params for each group
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        if master_process:        
            # print them out to verify correct behavior
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # fused is a parameter option for AdamW that 'fuses' the kernels together for all parameters 
        # without it, there are a lot of individual kernels (e.g., mul, add, decay); this just combines them all into one
        # goal: gets rid of a lot of overhead by calling one kernel on all parameter operations
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")

        # betas and eps are from GPT 3 paper
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer
    
    