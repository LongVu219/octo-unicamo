"""
Convert BridgeV2 3K subset (parquet + mp4) to RLDS/TFDS format for Octo training.

Usage:
    conda activate octo
    cd /prj/corp/airesearch/lasvegas/vol5-scratch/users/phongnh/Long/octo
    python convert_to_rlds.py

Output:
    data/bridge3k_dataset/1.0.0/  (TFRecord shards + metadata)
"""

import os
import sys

# Add the octo root so the builder module can be found
OCTO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OCTO_ROOT)

import tensorflow_datasets as tfds

DATA_SOURCE = "/prj/corp/airesearch/lasvegas/vol5-scratch/users/phongnh/Long/data/bridgeV2/real_3k_1"
TFDS_DATA_DIR = os.path.join(OCTO_ROOT, "data")


def main():
    print(f"Source data: {DATA_SOURCE}")
    print(f"TFDS output: {TFDS_DATA_DIR}/bridge3k_dataset/")
    print()

    # Import the builder
    from bridge3k_dataset.bridge3k_dataset_dataset_builder import Builder

    builder = Builder(
        data_dir=TFDS_DATA_DIR,
        data_source_dir=DATA_SOURCE,
    )

    print(f"Dataset info:")
    print(f"  Name: {builder.name}")
    print(f"  Version: {builder.version}")
    print()

    # Build the dataset (reads mp4 + parquet, writes TFRecords)
    print("Building RLDS dataset (this may take a while)...")
    builder.download_and_prepare()

    # Verify
    ds = builder.as_dataset(split="train")
    count = 0
    for episode in ds.take(3):
        steps = list(episode["steps"])
        count += 1
        ep_idx = episode["episode_metadata"]["episode_index"].numpy()
        n_steps = len(steps)
        lang = steps[0]["language_instruction"].numpy().decode()
        print(f"  Episode {ep_idx}: {n_steps} steps, task='{lang[:60]}'")

    total = sum(1 for _ in ds)
    print(f"\nTotal episodes in dataset: {total}")
    print(f"\nRLDS dataset ready at: {TFDS_DATA_DIR}/bridge3k_dataset/")


if __name__ == "__main__":
    main()
