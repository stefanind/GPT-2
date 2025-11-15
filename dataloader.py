import numpy as np
import torch
import os
import torch

def load_tokens(filename):
    ext = os.path.splitext(filename)[1]

    if ext == ".npy":
        # edu_fineweb10B shards
        npt = np.load(filename)
        npt = npt.astype(np.int32) 

    elif ext == ".bin":
        # tinyshakespeare raw uint16 token file
        n_bytes = os.path.getsize(filename)
        n_tokens = n_bytes // 2   # uint16 = 2 bytes

        # memory map for efficiency and large file support
        npt = np.memmap(filename, dtype=np.uint16, mode="r", shape=(n_tokens,))
        npt = npt.astype(np.int32)

    else:
        raise ValueError(f"Unknown token file format: {filename}")

    return torch.tensor(npt, dtype=torch.long)

# between shard shuffle
# goal: smooth out training loss curve 
# the curve isn't smooth because each shard likely reflects data from specific areas of the internet
# e.g., shard 1-5 might be Wikipedia, then 6-10 might be all of Arxiv, etc.
# so shuffling between shards makes training loss smoother rather than jagged
def between_shard_shuffle(shards, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(len(shards), generator=g).tolist()
    return [shards[i] for i in perm]

class DataLoaderLite:
    def __init__(self, 
                 B, 
                 T, 
                 process_rank, 
                 num_processes, 
                 split, 
                 master_process, 
                 dataset='fineweb10B',
                 shuffle_shards=True, 
                 seed=1337):
        
        self.B = B
        self.T = T

        # for when ddp is being used
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {'train', 'val'}

        self.dataset = dataset
        # build shards depending on dataset
        self.shards = self.build_shards(split, shuffle_shards, seed, master_process)

        if master_process:
            print(f"found {len(self.shards)} shards for split {split}")
        self.reset()


    def build_shards(self, split, shuffle_shards, seed, master_process):
        """Build self.shards depending on which dataset we're using."""
        shards = []

        if self.dataset == "fineweb10B":
            data_root = "edu_fineweb10B"
            shards = os.listdir(data_root)
            shards = [s for s in shards if split in s]
            shards = sorted(shards)
            shards = [os.path.join(data_root, s) for s in shards]

            assert len(shards) > 0, f"no shards found for split {split} in {data_root}"

            # between-shard shuffle only for train
            # NOTE: does not work with multiple epochs
            # but since we are only training on one epoch, it is okay for now
            if shuffle_shards and split == "train":
                shards = between_shard_shuffle(shards, seed)

        elif self.dataset == "tinyshakespeare":
            # single file per split: data/train.bin, data/val.bin
            data_root = os.path.join(os.path.dirname(__file__), "data")
            fname = "train.bin" if split == "train" else "val.bin"
            path = os.path.join(data_root, fname)
            assert os.path.exists(path), f"file not found: {path}"
            shards = [path]  # just one shard

            if master_process:
                print(f"using {path} for split {split}")

        else:
            raise ValueError(f"unknown dataset: {self.dataset}")

        return shards

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T

        # extract tokens based on the current position
        # ensures that each batch isn't overlapping
        buf = self.tokens[self.current_position : self.current_position+B*T+1] # +1 to get the target token

        # parse tokens into B, T
        x = (buf[:-1].view(B, T))
        y = (buf[1:].view(B, T))

        # increases position based on what data has already been used
        self.current_position += B * T * self.num_processes

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y