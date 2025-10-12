#!/usr/bin/env python3
"""
Dataset downloader for Hugging Face datasets.
"""

import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from datasets import load_dataset

from .config.datasets import DATASETS, MODELS
from .config.logging import setup_logger
from .utils import ensure_directory_exists, save_json_file

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
        _LOGGER.error("Make sure the dataset ID exists on Hugging Face Hub")


def download_model(model_name: str):
    """Download a model from Hugging Face."""
    # Check if it's a predefined model first
    if model_name in MODELS:
        repo_id = MODELS[model_name]
    else:
        # Allow direct Hugging Face model IDs
        repo_id = model_name
        _LOGGER.info(f"Downloading custom model: {repo_id}")

    try:
        _LOGGER.info(f"Downloading {model_name} ({repo_id})...")
        models_dir = "models"
        ensure_directory_exists(models_dir)

        # Use repo_id for directory name to avoid conflicts
        safe_name = repo_id.replace("/", "_").replace(":", "_")
        local_model_dir = os.path.join(models_dir, safe_name)
        local_dir = snapshot_download(repo_id=repo_id, local_dir=local_model_dir)
        _LOGGER.info(f"Model downloaded to {local_dir}")

        _LOGGER.info(f"Successfully downloaded {model_name}")

    except Exception as e:
        _LOGGER.error(f"Error downloading {model_name}: {e}")
        _LOGGER.error("Make sure the model ID exists on Hugging Face Hub")


def list_models():
    """List available models."""
    print("Available models:")
    print("-" * 20)
    for name, repo in MODELS.items():
        print(f"{name}: {repo}")


def main():
    """Main CLI entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Download a dataset or model",
        epilog="Examples:\n  python downloader.py agent_harm_chat\n  python downloader.py xcomet-xl",
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Dataset or model name (see available lists)",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print("Available datasets:")
        for name, url in DATASETS.items():
            print(f"  {name}: {url}")
        return

    if args.list_models:
        print("Available models:")
        for name, repo in MODELS.items():
            print(f"  {name}: {repo}")
        return

    if not args.target:
        parser.print_help()
        return

    target_name = args.target

    if target_name in DATASETS:
        dataset_url = DATASETS[target_name]
        download_dataset(dataset_url)
    elif target_name in MODELS:
        download_model(target_name)
    else:
        _LOGGER.error(f"Unknown dataset or model: {target_name}")
        print("Available datasets:")
        for name, url in DATASETS.items():
            print(f"  {name}: {url}")
        print("\nAvailable models:")
        for name, repo in MODELS.items():
            print(f"  {name}: {repo}")
        return


if __name__ == "__main__":
    main()
