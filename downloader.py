import json
import os
import sys

from dotenv import load_dotenv

from config.datasets import DATASETS
from config.logging import _LOGGER
from datasets import load_dataset


def download_dataset(dataset_id: str, output_dir: str = "datasets/raw"):
    """Download a single dataset from Hugging Face."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        _LOGGER.info(f"Downloading {dataset_id}...")

        # Parse dataset_id to get repo and split
        if ":" in dataset_id:
            repo, split = dataset_id.split(":", 1)
            dataset = load_dataset(repo, split)
        else:
            dataset = load_dataset(dataset_id)

        # Save each split
        for split_name, split_data in dataset.items():
            output_file = os.path.join(
                output_dir,
                f"{dataset_id.replace('/', '_').replace(':', '_')}_{split_name}.json",
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(list(split_data), f, ensure_ascii=False, indent=2)
            _LOGGER.info(f"Saved {split_name} to {output_file}")

    except Exception as e:
        _LOGGER.error(f"Error downloading {dataset_id}: {e}")


def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python downloader.py <dataset_name>")
        print("Available datasets:")
        for name, url in DATASETS.items():
            print(f"  {name}: {url}")
        return

    dataset_name = sys.argv[1]

    if dataset_name not in DATASETS:
        _LOGGER.error(f"Unknown dataset: {dataset_name}")
        print("Available datasets:")
        for name, url in DATASETS.items():
            print(f"  {name}: {url}")
        return

    dataset_url = DATASETS[dataset_name]
    download_dataset(dataset_url)


if __name__ == "__main__":
    main()
