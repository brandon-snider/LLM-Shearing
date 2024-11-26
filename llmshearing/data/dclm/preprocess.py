"""
— Downloads and tokenizes a random sample of the DCLM dataset
— Saves data shards to ../.data/dclm
    — i.e. from the root of the repo: .data/dclm/dclm_train_000000.npy, etc.
— Sample size and shard size are configurable
— Splits are "train" and "val"
— Size of "val" is currently determined by shared_size

Run as: python -m data.preprocess
"""

import os
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from streaming import MDSWriter

print("Initializing tokenizer")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
tokenizer.pad_token = tokenizer.eos_token


def tokenize(doc):
    # Tokenizes a document and returns a numpy array of uint16 tokens
    tokens_np = tokenizer.encode(
        doc["text"], add_special_tokens=True, return_tensors="np"
    )
    tokens_np = tokens_np.flatten()
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "Token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


# TODO: remove if we're sticking with MDSWriter
# def write_datafile(filename, tokens_np):
#     # Ensure directory exists
#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#     np.save(filename, tokens_np)


def create_shard_progress(shard_index, sequences_per_shard):
    return tqdm(
        total=sequences_per_shard,
        unit="sequences",
        desc=f"Shard {shard_index}",
        position=1,
        leave=False,
    )


def main():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN environment variable is not set")

    # TODO: make this configurable
    local_dir = "../../../data/dclm/pythia/mds"
    seq_length = 2048
    eval_size = int(1e7)  # 10M tokens for eval set
    shard_size = int(1e8)  # TODO: remove, since we're using MDSWriter
    total_size = int(1e9)  # 1B tokens
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
        token=hf_token,
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
                    sequences_np = np.array(sequences, dtype=np.uint16)

                    # TODO: remove if we're sticking with MDSWriter
                    # filename = os.path.join(
                    #     DATA_CACHE_DIR, split, f"dclm_{split}_{shard_index:06d}"
                    # )
                    # write_datafile(filename, sequences_np)

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

            sequences_np = np.array(sequences, dtype=np.uint16)

            # TODO: remove if we're sticking with MDSWriter
            # filename = os.path.join(
            #     DATA_CACHE_DIR, split, f"dclm_{split}_{shard_index:06d}"
            # )
            # write_datafile(filename, sequences_np)

            writer = eval_writer if split == "eval" else train_writer

            for sequence_np in sequences_np:
                writer.write({"tokens": sequence_np.tobytes(), "set": split})

        overall_progress.close()

    eval_writer.finish()
    train_writer.finish()


if __name__ == "__main__":
    main()
