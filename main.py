import os
import sys

from dotenv import load_dotenv

from config.datasets import DATASETS


def show_help():
    """Show help information."""
    print("Dataset Manager")
    print("=" * 20)
    print("Commands:")
    print("  download <dataset>     - Download dataset")
    print("  translate <file>       - Translate dataset file")
    print("  list                   - List available datasets")
    print("  files                  - List downloaded files")
    print("  help                   - Show this help")
    print()
    print("Examples:")
    print("  python main.py download agent_harm_chat")
    print("  python main.py download alert")
    print("  python main.py translate datasets/raw/agent_harm_chat_test_public.json")
    print("  python main.py translate dataset.json --workers=8")
    print("  python main.py translate dataset.json --output=output.json --workers=8")


def list_datasets():
    """List available datasets."""
    print("Available datasets:")
    print("-" * 20)
    for name, url in DATASETS.items():
        print(f"{name}: {url}")


def list_files():
    """List downloaded files."""
    raw_dir = "datasets/raw"
    processed_dir = "datasets/processed"

    print("Downloaded files:")
    print("-" * 20)
    if os.path.exists(raw_dir):
        for file in sorted(os.listdir(raw_dir)):
            if file.endswith(".json"):
                print(f"raw/{file}")

    print("\nProcessed files:")
    print("-" * 20)
    if os.path.exists(processed_dir):
        for file in sorted(os.listdir(processed_dir)):
            if file.endswith(".json"):
                print(f"processed/{file}")


def main():
    load_dotenv()

    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1]

    if command == "help":
        show_help()
    elif command == "list":
        list_datasets()
    elif command == "files":
        list_files()
    elif command == "download":
        if len(sys.argv) < 3:
            print("Usage: python main.py download <dataset>")
            return
        os.system(f"python downloader.py {' '.join(sys.argv[2:])}")
    elif command == "translate":
        if len(sys.argv) < 3:
            print("Usage: python main.py translate <file> [--output=file] [--workers=N]")
            print("Examples:")
            print("  python main.py translate dataset.json")
            print("  python main.py translate dataset.json --workers=8")
            print("  python main.py translate dataset.json --output=output.json --workers=8")
            return
        os.system(f"python translator.py {' '.join(sys.argv[2:])}")
    else:
        print(f"Unknown command: {command}")
        show_help()


if __name__ == "__main__":
    main()
