"""
This is meant for Windows installation of the data. Does not work with Linux.
"""


import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm
import signal                                        


local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] 

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint32) # widen temporarily 
    if (tokens_np >= 2**16).any():
        raise ValueError("token id >= 65536; uint16 overflow")
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# make pool interrupt-friendly 
def _init_worker():
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

def main():  # windows needs a main() when using multiprocessing
    # fewer HEAD requests + gentle retries
    FIRST_RUN = True  # set to False after the first successful download
    dl_cfg = DownloadConfig(
        max_retries=2,
    )

    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=remote_name,
        split="train",
        verification_mode="no_checks", # cuts metadata HEADs
        download_config=dl_cfg,
    )

    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs, initializer=_init_worker) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        # smaller chunksize is more responsive and there are fewer long hangs
        for tokens in pool.imap(tokenize, fw, chunksize=4):

            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar.close(); progress_bar = None
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

if __name__ == "__main__": 
    mp.freeze_support() 
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
