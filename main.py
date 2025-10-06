#!/usr/bin/env python3
"""
Main CLI interface for the dadaptbr project.
"""

import os
import sys

from dotenv import load_dotenv

from config.datasets import DATASETS, MODELS


def show_help():
    """Show help information."""
    print("Dataset Manager")
    print("=" * 20)
    print("Commands:")
    print("  download <dataset|model> - Download dataset or model")
    print("  translate <file>         - Translate dataset file")
    print("  evaluate <file>          - Evaluate translation quality")
    print("  list                     - List available datasets")
    print("  models                   - List available models")
    print("  files                    - List downloaded files")
    print("  help                     - Show this help")
    print()
    print("Examples:")
    print("  python main.py download agent_harm_chat")
    print("  python main.py download xcomet-xl")
    print("  python main.py translate datasets/raw/agent_harm_chat_test_public.json")
    print("  python main.py evaluate dataset.json --limit=100")


def list_datasets():
    """List available datasets."""
    print("Available datasets:")
    print("-" * 20)
    for name, url in DATASETS.items():
        print(f"{name}: {url}")


def list_models():
    """List available models."""
    print("Available models:")
    print("-" * 20)
    for name, repo in MODELS.items():
        print(f"{name}: {repo}")


def list_files():
    """List downloaded files."""
    dirs = ["datasets/raw", "datasets/processed", "datasets/evaluation"]

    for dir_name in dirs:
        if os.path.exists(dir_name):
            files = [f for f in os.listdir(dir_name) if f.endswith(".json")]
            if files:
                print(f"\n{dir_name.split('/')[-1].title()} files:")
                print("-" * 20)
                for file in sorted(files):
                    print(f"{dir_name.split('/')[-1]}/{file}")


def main():
    """Main CLI entry point."""
    load_dotenv()

    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1]

    if command == "help":
        show_help()
    elif command == "list":
        list_datasets()
    elif command == "models":
        list_models()
    elif command == "files":
        list_files()
    elif command == "download":
        if len(sys.argv) < 3:
            print("Usage: python main.py download <dataset|model>")
            return
        os.system(f"python downloader.py {' '.join(sys.argv[2:])}")
    elif command == "translate":
        if len(sys.argv) < 3:
            print(
                "Usage: python main.py translate <file> [--output=file] [--workers=N]"
            )
            return
        os.system(f"python translator.py {' '.join(sys.argv[2:])}")
    elif command == "evaluate":
        if len(sys.argv) < 3:
            print("Usage: python main.py evaluate <file> [--output=file] [--limit=N]")
            return
        os.system(f"python evaluator.py {' '.join(sys.argv[2:])}")
    else:
        print(f"Unknown command: {command}")
        show_help()


if __name__ == "__main__":
    main()
