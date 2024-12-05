# Run from the project root as: python -m llmshearing.data.verify
# Loads data into a StreamingDataset and prints out the first num_samples_to_check samples
# Used to verify that the data preprocessing is working correctly

from streaming import StreamingDataset
from transformers import AutoTokenizer
import numpy as np
import os
from llmshearing.tokenizers.tokenization_qwen2_fast import Qwen2TokenizerFast


def verify_data(split, data_dir, num_samples_to_check=5):
    # Initialize the tokenizer
    # tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2.5-1.5B")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tokenizer.pad_token = tokenizer.eos_token

    # Set the path to your data split
    split_dir = os.path.join(data_dir, split)

    # Create a StreamingDataset instance
    dataset = StreamingDataset(
        local=split_dir,
        shuffle=False,  # Set to True if you want to shuffle the data
        batch_size=1,  # We'll process samples one at a time
        # prefetch=0,  # Adjust prefetch settings as needed
    )

    count = 0

    for sample in dataset:
        # Retrieve the tokens
        tokens_bytes = sample["tokens"]
        tokens = np.frombuffer(tokens_bytes, dtype=np.uint16)  # Adjust dtype if needed

        # Decode the tokens back to text
        decoded_text = tokenizer.decode(tokens, skip_special_tokens=True)

        # Print or inspect the decoded text
        print(f"Sample {count + 1}:")
        print(decoded_text)
        print("-" * 50)

        count += 1
        if count >= num_samples_to_check:
            break

    print(f"Verified {count} samples from the '{split}' split.")


if __name__ == "__main__":
    data_dir = "data/opencoder-annealing/pythia/for_prune_merged"
    verify_data("train", data_dir)
    verify_data("eval", data_dir)
