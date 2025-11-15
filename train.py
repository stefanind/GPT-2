
import os
import time
import tiktoken

import torch
from torch.nn import functional as F

from hellaswag import iterate_examples, render_example
from utils import get_most_likely_row
from scheduler import cosine_warmup_lr
from dist import setup_ddp
from model import GPTConfig, GPT
from dataloader import DataLoaderLite

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import torch.profiler as profiler

# -------------------------------------------------------------------------------------------
# train.py
def main():

    # init DDP/device 
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, master_process = setup_ddp()

    # seeding
    torch.manual_seed(1337)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(1337)

    # init encoder
    enc = tiktoken.get_encoding("gpt2")

    # --- hyperparams ---
    dataset = 'tinyshakespeare' # or 'fineweb10B'
    if dataset == 'fineweb10B':
        total_batch_size = 524288       # ~0.5M batch size in the number of tokens; used a "nice number" (2**19)
        B, T             = 64, 1024     # batch size, sequence length
        max_steps        = 19073        # 10**9 / 2**19 = 19073 -> 10B tokens / 524288 batch size = 19073 steps
        warmup_steps     = 715          # GPT-3 paper says 375M tokens are for warmup; 375e6 / 2**19 = 715
        eval_every       = 500          # init evaluation every 500 steps of the main loop
        save_every       = 5000         # init model checkpoint every 5000 steps of the main loop

    # use tinyshakespeare for profiling the model
    elif dataset == 'tinyshakespeare':
        train_tokens     = 301966 * 10  # 10 epochs
        total_batch_size = 15360        
        B, T             = 12, 128      
        max_steps        = 196          # 3019660 // 15360
        warmup_steps     = 8            
        eval_every       = 500          # won't be used for tinyshakespeare; eval will only happen at last step
        save_every       = 5000         # won't be used for tinyshakespeare; save will only happen at last step

    max_lr           = 6e-4         # peak lr
    min_lr           = max_lr * 0.1 # final lr (10% of max)
    weight_decay     = 0.1          # weight decay the parameters for regularization
    use_compile      = False        # torch.compile set to False because of issues while using it in eval mode

    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure that the batch size is divisible by B * T * ddp_world_size"

    # --- initialize gradient accumulation ---
    # going to do many forward and backwards without updating
    # goal: want to simulate/handle the GPT-3 Small batch size but with less compute, i.e., cannot do B=488 to get 0.5M batch size

    # do forward and backward grad_accum_steps number of times
    # e.g., 2**19 // (16 * 1024) = 32
    # so do forward an backward 32 times before updating
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"the full batch size: {total_batch_size}")
        print(f"gradient accumulation steps: {grad_accum_steps}")

    # set the data/val loader based on the desired dataset
    train_loader = DataLoaderLite(B=B, 
                                  T=T, 
                                  process_rank=ddp_rank, 
                                  num_processes=ddp_world_size, 
                                  master_process=master_process, 
                                  split='train', 
                                  dataset=dataset)
    val_loader   = DataLoaderLite(B=B, 
                                  T=T, 
                                  process_rank=ddp_rank, 
                                  num_processes=ddp_world_size, 
                                  master_process=master_process, 
                                  split='val', 
                                  dataset=dataset)

    # using TF32 (10-bit mantissa instead of 23-bit) for speed, as long as I have some reasonable accuracy
    # useful for training/inference
    torch.set_float32_matmul_precision('high') 

    # --- initialize model w/ kernel-efficient numbers ---
    # override vocab size to be a "nice number"
    # "nice numbers" have more flops but are more efficient due to how GPU's are made
    # kernel's will generally operate on 64 x 64 blocks 
    # after computing all these, they will then perform on the odd numbers not within this multiple
    # these kernels that do this extra part are inefficient
    # so padding the input results in these optimized kernels working with all the numbers
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)

    if use_compile:
        model = torch.compile(model) 

    # init proper model if ddp is used
    if ddp:
        # DDP synchronizes and averages the gradients across all ranks
        model = DDP(model, device_ids=[ddp_local_rank])
    # since using DDP makes it a new object, need .module to access nn.Module
    raw_model = model.module if ddp else model

    # optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device=device, master_process=master_process)

    # --- logging ---
    # create the a directory that we will write checkpoints to and log to

    log_dir = f"log_{dataset}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{dataset}.txt")
    # open the file to clear it out before appending to it later
    # goal: start from scratch each time a training run is done
    with open(log_file, "w") as f:
        pass

    # --- profiler setup ---
    use_profiler = master_process  # only profile on rank 0
    if use_profiler:
        prof_schedule = profiler.schedule(
            wait=1,    # steps with no profiling (let things "warm up")
            warmup=1,  # steps that are recorded but marked as warmup
            active=3,  # steps actually saved to trace
            repeat=1
        )
        prof_dir = os.path.join(log_dir, "profiler")
        os.makedirs(prof_dir, exist_ok=True)

        prof = profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            schedule=prof_schedule,
            on_trace_ready=profiler.tensorboard_trace_handler(prof_dir),
            record_shapes=True,  # records input tensor shapes for each operation
            profile_memory=True, # track memory usage per operator
            with_stack=True,     # 
        )
        prof.__enter__()
    else:
        prof = None


    # --- main train/eval loop ---
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # --- validation ---
        if step % eval_every == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            # write the validation loss to the log file and set up a model checkpoint
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % save_every == 0 or last_step):
                    # model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt") # pad with zeros for consistent parsing
                    checkpoint = {
                        'model': raw_model.state_dict(),   # model weights
                        'config': raw_model.config,        # the config settings
                        'step': step,                      # the current step
                        'val_loss': val_loss_accum.item()  # the val loss at this step
                    }
                    torch.save(checkpoint, checkpoint_path)

        # NOTE: torch.compile doesn't work with HellaSwag
        # dataset has to be fineweb10B
        # --- HellaSwag eval (normalized) ---
        if (step % eval_every == 0 or last_step) and (not use_compile) and (dataset == 'fineweb10B'):
            num_correct_norm = 0
            num_total = 0
            model.eval()
            for i, ex in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:  # shard across ranks
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(ex)
                tokens, mask = tokens.to(device), mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # NOTE: torch.compile doesn't work with generating samples
        # --- short sampling ---
        if ((step > 0 and step % eval_every == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen) # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :] # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        # --- training ---
        # one step of the optimization
        model.train()

        # initialize/reset all gradients to zero
        optimizer.zero_grad()

        # accumulate loss for reporting
        loss_accum = 0.0

        # implements a way to do ~0.5M batch size in tokens to simulate what GPT-3 Small actually does
        # can't do it how they did it because of lack of GPUs
        for micro_step in range(grad_accum_steps):

            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # don't want to synchronize after every loss.backward() during each micro step
            # so toggle 'require_backward_grad_sync' = True only when it is the last step
            # goal: only sync gradients at the end of the accum cycle, resulting in simulating a normal batch forward/backward
            # also, this is required during a forward pass
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            # ------ CHECK GPU TYPES BEFORE RUNNING!!! ------
            with torch.autocast(dtype=torch.bfloat16, device_type=device_type): # only allowed with Ampere and newer GPUs
                logits, loss = model(x, y)

            # scale each loss by the total steps to match a normal training loop
            # if no scaling, then the gradients are SUMing instead of MEANing
            # i.e., cross entropy uses 'reduction=mean' not 'reduction=sum'
            # so we need to emulate this behavior by adding each normalized gradient
            loss = loss / grad_accum_steps 

            # detach to add the float to loss_accum 
            loss_accum += loss.detach()

            # compute gradients across nodes
            loss.backward()

        # Uses pytorch distributed to all-reduce (average) the loss values across processes,
        # so every rank ends up with the same global mean loss
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # ensures gradients squared and summed (global norm) do not go over 1.0
        # i.e., the magnitude of all the gradients do not go over 1.0
        # want gradients to not diverge too much if they become wild
        # sometimes a batch can end up with v high loss and this will end up w/ v high gradient
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # using the lr function, set the lr in each parameter group found in the optimizer
        lr = cosine_warmup_lr(step, min_lr, max_lr, warmup_steps, max_steps)
        # iterate over the groups and set the lr based on the conditions of the function
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update weights
        optimizer.step()

        # ensure CPU and GPU are in sync
        # i.e., finish all GPU work before moving on
        if device_type == "cuda":
            torch.cuda.synchronize()

        # tracking metrics for printing while training
        t1 = time.time()
        dt = t1 - t0 # time in seconds
        # calculates number of tokens processed globally in one optimizer step
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec   = tokens_processed / dt

        # only print out and log the master process metrics for tracking
        # goal: if ddp active, don't want multiple printouts for each process
        if master_process:
            print(f"step {step:4d}, | loss: {loss_accum.item():.6f}, | lr: {lr:.4e} | norm: {norm:.4f} | dt {dt:.2f}s | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

        # let profiler know one training iteration has finished
        if prof is not None:
            prof.step()

    # --- cleanly close the profiler ---
    if prof is not None:
        prof.__exit__(None, None, None)


if __name__ == "__main__":
    main()