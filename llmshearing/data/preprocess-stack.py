# Run from the project root as: python -m llmshearing.data.preprocess-stack

from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from streaming import MDSWriter
import boto3
from smart_open import open

print("Initializing tokenizer")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token


def init_worker():
    global s3
    global tokenizer
    # Initialize S3 client in each worker process
    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3 = session.client("s3")
    # Initialize tokenizer in each worker process
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizer.pad_token = tokenizer.eos_token


def download_file(file):
    s3_url = f"s3://softwareheritage/content/{file['blob_id']}"
    try:
        with open(
            s3_url, "rb", compression=".gz", transport_params={"client": s3}
        ) as fin:
            file["content"] = fin.read().decode(file["src_encoding"])
    except Exception as e:
        print(f"Error downloading {file['blob_id']}: {e}")
        file["content"] = ""
    return file


def tokenize(sample):
    files = sample["files"]
    with ThreadPoolExecutor(max_workers=16) as executor:
        files = list(executor.map(download_file, files))
    sample_content = " ".join(file["content"] for file in files if "content" in file)
    tokens = tokenizer.encode(sample_content, add_special_tokens=True)
    return tokens


def main():
    local_dir = "../../data/stack-smol/qwen/mds"
    seq_length = 32768
    eval_size = int(1e7)  # 10M tokens for eval set
    total_size = int(1e9)  # 1B tokens
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
        "bigcode/the-stack-v2-train-smol-ids",
        split="train",
        # streaming=True,
    )

    # ds = ds.shuffle(seed=42, buffer_size=10000)
    ds = ds.shuffle(seed=42)

    print("Tokenizing documents")

    # Initialize the multiprocessing pool
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs, initializer=init_worker) as pool:
        current_tokens = []

        # Add overall progress bar
        total_sequences = total_size // seq_length
        overall_progress = tqdm(
            total=total_sequences, unit="sequences", desc="Total Progress", position=0
        )

        for tokens in pool.imap(tokenize, ds, chunksize=16):
            if tokens_collected >= total_size:
                break

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
                dtype = np.uint32
                # dtype = np.uint32 if max(sequence_tokens) >= 2**16 else np.uint16
                sequence_np = np.array(sequence_tokens, dtype=dtype)
                writer = eval_writer if split == "eval" else train_writer
                writer.write({"tokens": sequence_np.tobytes(), "set": split})

                overall_progress.update(1)

                if tokens_collected >= total_size:
                    break

        overall_progress.close()

    eval_writer.finish()
    train_writer.finish()


if __name__ == "__main__":
    main()
