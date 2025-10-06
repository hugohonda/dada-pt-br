#!/usr/bin/env python3
"""
Dataset downloader for Hugging Face datasets.
"""

import os
import sys

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from config.datasets import DATASETS, MODELS
from config.logging import setup_logger
from datasets import load_dataset
from utils import ensure_directory_exists, save_json_file

_LOGGER = setup_logger("downloader", log_to_file=True, log_prefix="download")


def download_dataset(dataset_id: str, output_dir: str = "datasets/raw"):
    """Download a dataset from Hugging Face."""
    ensure_directory_exists(output_dir)

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
            save_json_file(list(split_data), output_file)
            _LOGGER.info(f"Saved {split_name} to {output_file}")

    except Exception as e:
        _LOGGER.error(f"Error downloading {dataset_id}: {e}")


def download_model(model_name: str):
    """Download a model from Hugging Face."""
    if model_name not in MODELS:
        _LOGGER.error(f"Unknown model: {model_name}")
        return

    repo_id = MODELS[model_name]

    try:
        _LOGGER.info(f"Downloading {model_name} ({repo_id})...")

        # Use snapshot_download for all models
        local_dir = snapshot_download(repo_id=repo_id)
        _LOGGER.info(f"Model downloaded to {local_dir}")

        _LOGGER.info(f"Successfully downloaded {model_name}")

    except Exception as e:
        _LOGGER.error(f"Error downloading {model_name}: {e}")


def list_models():
    """List available models."""
    print("Available models:")
    print("-" * 20)
    for name, repo in MODELS.items():
        print(f"{name}: {repo}")


def main():
    """Main CLI entry point."""
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python downloader.py <dataset_name|model_name>")
        print("\nAvailable datasets:")
        for name, url in DATASETS.items():
            print(f"  {name}: {url}")
        print("\nAvailable models:")
        for name, config in MODELS.items():
            print(f"  {name}: {config['description']}")
        return

    target_name = sys.argv[1]

    # Check if it's a dataset
    if target_name in DATASETS:
        dataset_url = DATASETS[target_name]
        download_dataset(dataset_url)
    # Check if it's a model
    elif target_name in MODELS:
        download_model(target_name)
    else:
        _LOGGER.error(f"Unknown dataset or model: {target_name}")
        print("Available datasets:")
        for name, url in DATASETS.items():
            print(f"  {name}: {url}")
        print("\nAvailable models:")
        for name, config in MODELS.items():
            print(f"  {name}: {config['description']}")
        return


if __name__ == "__main__":
    main()
