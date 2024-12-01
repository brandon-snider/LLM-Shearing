import os
import asyncio
import aiohttp
import numpy as np
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer
from streaming import MDSWriter
import multiprocessing as mp

print("Initializing tokenizer")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token


# Asynchronous tokenization function
async def tokenize(sample):
    return tokenizer.encode(
        sample["text"], add_special_tokens=True, return_tensors="np"
    ).flatten()


async def download_and_tokenize(sample, session):
    # Simulate async download (replace with actual download code)
    async with session.get(sample["url"]) as response:
        text = await response.text()
        sample["text"] = text
    tokens = await tokenize(sample)
    return tokens


async def process_dataset(
    ds,
    total_sequences,
    seq_length,
    eval_size,
    sequences_per_shard,
    eval_writer,
    train_writer,
):
    tokens_collected = 0
    eval_tokens_collected = 0
    sequences = []
    current_tokens = []
    shard_index = 0

    async with aiohttp.ClientSession() as session:
        tasks = []
        for sample in ds:
            tasks.append(download_and_tokenize(sample, session))
            if len(tasks) >= 100:  # Adjust concurrency level as needed
                tokens_list = await asyncio.gather(*tasks)
                tasks = []
                for tokens in tokens_list:
                    # Rest of your processing logic
                    if tokens_collected >= total_size:
                        break

                    current_tokens.extend(tokens.tolist())

                    while len(current_tokens) >= seq_length:
                        sequences.append(current_tokens[:seq_length])
                        current_tokens = current_tokens[seq_length:]
                        tokens_collected += seq_length

                        split = "eval" if eval_tokens_collected < eval_size else "train"
                        if split == "eval":
                            eval_tokens_collected += seq_length

                        if len(sequences) >= sequences_per_shard or (
                            split == "eval" and eval_tokens_collected >= eval_size
                        ):
                            sequences_np = np.array(sequences, dtype=np.uint16)
                            writer = eval_writer if split == "eval" else train_writer

                            for sequence_np in sequences_np:
                                writer.write(
                                    {"tokens": sequence_np.tobytes(), "set": split}
                                )

                            sequences = []
                            shard_index += 1
        # Process any remaining tasks
        if tasks:
            tokens_list = await asyncio.gather(*tasks)
            for tokens in tokens_list:
                # Similar processing as above
                pass


def main():
    local_dir = "../../data/dclm/qwen/mds"
    seq_length = 2048
    eval_size = int(1e5)  # 10M tokens for eval set
    total_size = int(1e7)  # 1B tokens
    sequences_per_shard = total_size // seq_length

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
        streaming=True,  # Enable streaming mode
    )

    asyncio.run(
        process_dataset(
            ds,
            total_sequences,
            seq_length,
            eval_size,
            sequences_per_shard,
            eval_writer,
            train_writer,
        )
    )

    eval_writer.finish()
    train_writer.finish()


if __name__ == "__main__":
    main()
