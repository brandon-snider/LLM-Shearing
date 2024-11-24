# Split tokenized files into many smaller files
# Each smaller file will contain `seq_per_file` sequences
# This is to prepare for splitting into prune/ft/eval sets
# Usage: python -m llmshearing.data.pythia.split_tokenized_files <input_base_dir> <output_base_dir> <rows_per_file>
# Example: python -m llmshearing.data.pythia.split_tokenized_files data/tokenized data/split 1000

import numpy as np
from pathlib import Path
import sys


def split_npy_file(input_path, output_dir, rows_per_file=100):
    # Load the original array
    data = np.load(input_path)

    # Calculate number of files needed
    total_rows = data.shape[0]
    num_files = (total_rows + rows_per_file - 1) // rows_per_file

    # Get the base name without extension
    base_name = input_path.stem.replace("_sample", "")

    # Split and save the array into multiple files
    for i in range(num_files):
        start_idx = i * rows_per_file
        end_idx = min((i + 1) * rows_per_file, total_rows)

        # Extract the chunk
        chunk = data[start_idx:end_idx]

        # Create filename with 4-digit numbering
        output_filename = f"{base_name}_part_{i+1:04d}.npy"
        output_path = output_dir / output_filename

        # Save the chunk
        np.save(output_path, chunk)

    return num_files


def process_directory(input_base_dir, output_base_dir, rows_per_file):
    input_base = Path(input_base_dir)
    output_base = Path(output_base_dir)

    # Create the base output directory if it doesn't exist
    output_base.mkdir(exist_ok=True)

    # Process each subdirectory
    for subdir in input_base.iterdir():
        if subdir.is_dir():
            # Create corresponding output subdirectory
            output_subdir = output_base / subdir.name
            output_subdir.mkdir(exist_ok=True)

            # Process each .npy file in the subdirectory
            for npy_file in subdir.glob("*.npy"):
                num_files = split_npy_file(npy_file, output_subdir, rows_per_file)
                print(f"Split {npy_file} into {num_files} files in {output_subdir}")


if __name__ == "__main__":
    input_base_dir, output_base_dir, rows_per_file = sys.argv[1:]
    process_directory(input_base_dir, output_base_dir, int(rows_per_file))
