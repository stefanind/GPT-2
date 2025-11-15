import os
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
    n_embd: int      = 768   # embedding dimension
    dropout: float   = 0.0   # percentage of neurons dropped out
    bias: bool       = True  # add bias or not

# ------------------------------------------------------------
# Model

# TODO: implement RoPE

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        # check if the number of embeddings is a multiple of the number of heads
        # ensures that the embeddings can be properly split to each attn head
        assert config.n_embd % config.n_head == 0

        # project to 3 for query, key, values
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # seems pointless but important to be used to mix info from concatenated heads
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # tag for scaling down the variance of the residual stream
        self.c_proj.SCALE_INIT = 1

        # added dropout for regularization
        self.resid_dropout = nn.Dropout(config.dropout)

        # initialize number of heads and embeddings
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # -------- only needed if flash attention isn't used or is_causal=False
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                     .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        # get all dimensions
        B, T, C = x.size()

        # get query, key, values
        qkv = self.c_attn(x)

        # get three B, T, C, split along dim=2 (C)
        q, k, v = qkv.split(self.n_embd, dim=2) # split into size of self.n_embd along the 2nd dim

        # make the number of heads into a batch dimension like B
        # want pytorch to treat them as batches
        # all turn into -> (batch size, number of heads, sequence length, embd size for each head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # ------- manual attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v      # B, T, nh, hs
        
        # ------- flash attention (need CUDA)
        # is_causal=True so the mask isn't needed
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # concat all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 'contiguous' since transpose alone dn lay it into memory but view() needs it to be

        # mixes info from all head outputs and adds dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

# GPT-2 MLP but with added dropout regularization
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 4x projection, exactly as the original transformer
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

        # uses approximate GELU; not needed anymore but used for proper replication
        # GELU over RELU to remove dead neurons
        self.gelu    = nn.GELU(approximate='tanh')

        # back from the 4x projection down to size of n_embd
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        # tag for scaling down the variance of the residual stream
        self.c_proj.SCALE_INIT = 1

        # regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # pass through the input x
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
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
        self.mlp  = MLP(config)
    
    def forward(self, x):
        # residual stream + attn
        x = x + self.attn(self.ln_1(x))
        # residual stream + mlp
        x = x + self.mlp(self.ln_2(x))
        return x
    
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
            # creates n_layer amount of attn Blocks to split embeddings
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

    
    def forward(self, idx, targets=None):

        # idx of shape (B, T) -> batch size by sequence length 
        B, T = idx.size()

        # ensure that the sequence length is not bigger than context 
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"

        # position of each token up to the sequence length, T
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # input the positions to get their embeddings
        pos_emb = self.transformer.wpe(pos)

        # input the tokens to get their embeddings
        tok_emb = self.transformer.wte(idx)

        # combine the token embeddings and the positional embeddings
        x = tok_emb + pos_emb

        # pass through the 12 blocks, each w/ their layernorms, attn, and mlp
        for block in self.transformer.h:
            x = block(x)

        # pass through the layernorm after the attn mlp's
        x = self.transformer.ln_f(x)

        # get the logits via linear layer, i.e., from embedding size transformed to vocab size
        logits = self.lm_head(x) # (B, T, 50257)

        loss = None
        # for training
        if targets is not None:
            # takes (B*T, V), the logits, and (B*T), the targets, as input to calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
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
    
    # use the pretrained GPT-2 for evaluation comparison
    # goal: want to compare my model against GPT-2
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    