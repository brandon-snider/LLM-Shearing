"""
— Downloads and tokenizes a random sample of a HuggingFace dataset
— Saves data shards to [root]/data/[dataset_name]/[model_name]/mds/[train|eval]/[index.json,shard_000000.mds, ...]
    — e.g. from the root of the repo: data/dclm/qwen/mds/train/[index.json, shard_000000.mds, ...]
— Sample size and shard size are configurable
— Splits are "train" and "eval"
— Size of "eval" is currently determined by eval_size

Run as: python -m llmshearing.data.preprocess
"""

import os
import numpy as np
import multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from streaming import MDSWriter

print("Initializing tokenizer")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
tokenizer.pad_token = tokenizer.eos_token


def init_worker():
    global tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(sample):
    return tokenizer.encode(
        sample["text"], add_special_tokens=True, return_tensors="np"
    ).flatten()


def main():
    local_dir = "../../data/opencoder-annealing/pythia/synthetic_qa/mds"
    seq_length = 2048
    eval_size = int(2.8e5)  # 280K tokens for eval set
    total_size = int(5.6e7)  # 56M tokens
    tokens_collected = 0
    eval_tokens_collected = 0

    # Create the base data directory
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    eval_writer = MDSWriter(
        columns={"tokens": "bytes", "set": "str"},
        out=os.path.join(DATA_CACHE_DIR, "eval"),
        compression=None,
    )

    train_writer = MDSWriter(
        columns={"tokens": "bytes", "set": "str"},
        out=os.path.join(DATA_CACHE_DIR, "train"),
        compression=None,
    )

    print("Initializing dataset")
    ds = load_dataset(
        # "mlfoundations/dclm-baseline-1.0",
        "OpenCoder-LLM/opc-annealing-corpus",
        "synthetic_qa",  # "algorithmic_corpus" or "synthetic_code_snippet" or "synthetic_qa" 1B, 170M, 56M
        split="train",
        # num_proc=32,
        data_files="synthetic_qa/*.arrow",
        # streaming=True,
    )

    # ds = ds.shuffle(seed=42, buffer_size=10000)  # When streaming
    ds = ds.shuffle(seed=42)  # When not streaming

    print("Tokenizing documents")

    # Initialize the multiprocessing pool
    nprocs = max(1, os.cpu_count() - 2)
    with mp.Pool(nprocs, initializer=init_worker) as pool:
        current_tokens = []

        # Add overall progress bar
        total_sequences = total_size // seq_length
        overall_progress = tqdm(
            total=total_sequences, unit="sequences", desc="Total Progress", position=0
        )

        tokens_processed = 0

        for tokens in pool.imap(tokenize, ds, chunksize=32):
            if tokens_collected >= total_size:
                break

            tokens_processed += len(tokens)
            # Add new tokens to our buffer
            current_tokens.extend(tokens)

            # Create sequences of length seq_length while we have enough tokens
            while len(current_tokens) >= seq_length:
                sequence_tokens = current_tokens[:seq_length]
                current_tokens = current_tokens[seq_length:]
                tokens_collected += seq_length

                # Determine split based on eval_tokens_collected
                split = "eval" if eval_tokens_collected < eval_size else "train"

                # Update eval tokens count if necessary
                if split == "eval":
                    eval_tokens_collected += seq_length

                # Convert sequence to numpy array and save
                dtype = np.uint16
                # dtype = np.uint32 if max(sequence_tokens) >= 2**16 else np.uint16
                sequence_np = np.array(sequence_tokens, dtype=dtype)
                writer = eval_writer if split == "eval" else train_writer
                writer.write({"tokens": sequence_np.tobytes(), "set": split})

                overall_progress.update(1)

                if tokens_collected >= total_size:
                    break

        if tokens_collected < total_size:
            print(f"\nWarning: Dataset exhausted after {tokens_collected:,} tokens.")
            print(f"Requested {total_size:,} tokens but dataset was too small.")

        overall_progress.close()

    eval_writer.finish()
    train_writer.finish()


if __name__ == "__main__":
    main()
