import urllib.request
from pathlib import Path

from .config.datasets import DATASETS


def download_dataset(dataset_id: str, output_dir: str = "datasets/raw") -> str:
    """Download dataset from Hugging Face"""
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_id}. Available: {list(DATASETS.keys())}")

    hf_name = DATASETS[dataset_id]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{hf_name.replace('/', '_')}_train.json"
    filepath = output_path / filename

    # Check if already exists
    if filepath.exists():
        print(f"Dataset already exists: {filepath}")
        return str(filepath)

    # Download from Hugging Face
    url = f"https://huggingface.co/datasets/{hf_name}/resolve/main/train.json"

    try:
        print(f"Downloading from: {url}")
        with urllib.request.urlopen(url) as response:
            data = response.read()
            if len(data) > 100:  # Valid data
                with open(filepath, 'wb') as f:
                    f.write(data)
                print(f"Downloaded {len(data)} bytes to {filepath}")
                return str(filepath)
            else:
                raise RuntimeError(f"Invalid data: {len(data)} bytes")
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}") from e


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
        download_dataset(dataset_id)
    else:
        print("Usage: python -m dadaptbr.downloader <dataset_id>")
        print(f"Available datasets: {list(DATASETS.keys())}")
