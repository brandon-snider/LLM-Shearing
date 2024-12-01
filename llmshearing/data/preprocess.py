"""
— Downloads and tokenizes a random sample of a HuggingFace dataset
— Saves data shards to [root]/data/[dataset_name]/[model_name]/mds/[train|eval]/[index.json,shard_000000.mds, ...]
    — i.e. from the root of the repo: data/dclm/qwen/mds/train/[index.json, shard_000000.mds, ...]
— Sample size and shard size are configurable
— Splits are "train" and "eval"
— Size of "eval" is currently determined by eval_size

Run as: python -m llmshearing.data.preprocess
"""

import os
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from streaming import MDSWriter


print("Initializing tokenizer")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token


# Switch between these depending on whether we're using The Stack or some other dataset
def tokenize(sample):
    return tokenizer.encode(
        sample["text"], add_special_tokens=True, return_tensors="np"
    ).flatten()


def create_shard_progress(shard_index, sequences_per_shard):
    return tqdm(
        total=sequences_per_shard,
        unit="sequences",
        desc=f"Shard {shard_index}",
        position=1,
        leave=False,
    )


def main():
    # TODO: make this configurable
    local_dir = "../../data/dclm/qwen/mds"
    seq_length = 2048
    eval_size = int(1e5)  # 10M tokens for eval set
    shard_size = int(1e6)  # TODO: remove, since we're using MDSWriter
    total_size = int(1e7)  # 1B tokens
    sequences_per_shard = shard_size // seq_length
    total_sequences = total_size // seq_length
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
        "mlfoundations/dclm-baseline-1.0",
        split="train",
        streaming=True,
    )

    ds = ds.shuffle(seed=42, buffer_size=10000)

    print("Tokenizing documents")

    # Tokenize all documents
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        sequences = []
        current_tokens = []
        shard_progress = None

        # Add overall progress bar
        overall_progress = tqdm(
            total=total_sequences, unit="sequences", desc="Total Progress", position=0
        )

        for tokens in pool.imap(tokenize, ds, chunksize=16):
            if tokens_collected >= total_size:
                break

            if shard_progress is None:
                shard_progress = create_shard_progress(shard_index, sequences_per_shard)

            # Add new tokens to our buffer
            current_tokens.extend(tokens.tolist())

            # Create sequences of length 2048 while we have enough tokens
            while len(current_tokens) >= seq_length:
                sequences.append(current_tokens[:seq_length])
                current_tokens = current_tokens[seq_length:]
                tokens_collected += seq_length

                # Determine split based on eval_tokens_collected
                split = "eval" if eval_tokens_collected < eval_size else "train"

                # Update eval tokens count if necessary
                if split == "eval":
                    eval_tokens_collected += seq_length

                if shard_progress is None:
                    shard_progress = create_shard_progress(
                        shard_index, sequences_per_shard
                    )

                shard_progress.update(1)
                overall_progress.update(1)

                # If we've collected enough sequences for a shard, write it
                if len(sequences) >= sequences_per_shard or (
                    split == "eval" and eval_tokens_collected >= eval_size
                ):
                    # Convert sequences to numpy array and save
                    # Check if any value exceeds uint16 limit
                    if max(max(seq) for seq in sequences) >= 2**16:
                        sequences_np = np.array(sequences, dtype=np.uint32)
                    else:
                        sequences_np = np.array(sequences, dtype=np.uint16)

                    writer = eval_writer if split == "eval" else train_writer

                    for sequence_np in sequences_np:
                        writer.write({"tokens": sequence_np.tobytes(), "set": split})

                    # Reset for next shard
                    sequences = []
                    shard_index += 1
                    shard_progress.close()
                    shard_progress = None

        # Write any remaining complete sequences as the last shard
        if sequences:
            split = "eval" if eval_tokens_collected < eval_size else "train"

            # Check if any value exceeds uint16 limit
            if max(max(seq) for seq in sequences) >= 2**16:
                sequences_np = np.array(sequences, dtype=np.uint32)
            else:
                sequences_np = np.array(sequences, dtype=np.uint16)

            writer = eval_writer if split == "eval" else train_writer

            for sequence_np in sequences_np:
                writer.write({"tokens": sequence_np.tobytes(), "set": split})

        overall_progress.close()

    eval_writer.finish()
    train_writer.finish()


if __name__ == "__main__":
    main()
