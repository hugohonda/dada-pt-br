#!/usr/bin/env python3
"""
CLI interface DADA-PT-BR
"""

import argparse
import os

from dotenv import load_dotenv

from config.datasets import DATASETS, MODELS
from llm_client import get_ollama_name


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
    import glob

    print("Available files:")
    print("  Data files:")
    for file in sorted(glob.glob("data/**/*.json", recursive=True)):
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # MB
            print(f"    • {file} ({size:.1f} MB)")

    print("  Analysis files:")
    for file in sorted(glob.glob("outputs/visualizations/**/*.png", recursive=True)):
        if os.path.exists(file):
            print(f"    • {file}")

    print("  Reports:")
    for file in sorted(glob.glob("outputs/reports/**/*.json", recursive=True)):
        if os.path.exists(file):
            print(f"    • {file}")
    for file in sorted(glob.glob("outputs/reports/**/*.txt", recursive=True)):
        if os.path.exists(file):
            print(f"    • {file}")

    print("  Logs:")
    for file in sorted(glob.glob("outputs/logs/**/*.log", recursive=True)):
        if os.path.exists(file):
            print(f"    • {file}")


def handle_download(args):
    # Lazy import to avoid heavy deps when not needed
    from downloader import download_dataset, download_model

    target = args.target
    if target in DATASETS:
        download_dataset(DATASETS[target])
    elif target in MODELS:
        download_model(target)
    else:
        print(f"Unknown dataset or model: {target}")
        list_datasets()
        print()
        list_models()


def handle_translate(args):
    # Lazy import to avoid heavy deps when not needed
    from translator import process_dataset as translate_process

    model_name = get_ollama_name("towerinstruct" if args.tower else "gemma3")
    translate_process(
        args.input_file,
        args.output,
        model_name,
        args.workers,
        args.limit,
    )


def handle_evaluate(args):
    # Lazy import to avoid heavy deps when not needed
    from evaluator import process_dataset as evaluate_process
    from utils import generate_evaluation_filename

    output = args.output or generate_evaluation_filename(args.input_file)
    evaluate_process(args.input_file, output, args.limit)


def main():
    """Main CLI entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="dadaptbr CLI",
        epilog=(
            "Examples:\n"
            "  python main.py download agent_harm_chat\n"
            "  python main.py translate datasets/raw/file.json --tower --limit=100\n"
            "  python main.py evaluate datasets/processed/file.json --limit=100\n"
            "  python main.py list\n  python main.py models\n  python main.py files"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # download
    p_download = subparsers.add_parser("download", help="Download dataset or model")
    p_download.add_argument("target", help="Dataset or model name")
    p_download.set_defaults(func=handle_download)

    # translate
    p_translate = subparsers.add_parser("translate", help="Translate a dataset file")
    p_translate.add_argument("input_file", help="Input JSON file to translate")
    p_translate.add_argument("--output", "-o", help="Output JSON file")
    p_translate.add_argument(
        "--workers", "-w", type=int, default=4, help="Number of parallel workers"
    )
    p_translate.add_argument("--limit", "-l", type=int, help="Limit examples")
    p_translate.add_argument(
        "--tower",
        action="store_true",
        help="Use TowerInstruct-Mistral-7B-v0.2 instead of Gemma3",
    )
    p_translate.set_defaults(func=handle_translate)

    # evaluate
    p_evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate translation quality with XCOMET-XL"
    )
    p_evaluate.add_argument("input_file", help="Input JSON file to evaluate")
    p_evaluate.add_argument("--output", "-o", help="Output JSON file")
    p_evaluate.add_argument("--limit", "-l", type=int, help="Limit examples")
    p_evaluate.set_defaults(func=handle_evaluate)

    # list datasets
    p_list = subparsers.add_parser("list", help="List available datasets")
    p_list.set_defaults(func=lambda _args: list_datasets())

    # models
    p_models = subparsers.add_parser("models", help="List available models")
    p_models.set_defaults(func=lambda _args: list_models())

    # files
    p_files = subparsers.add_parser("files", help="List downloaded files")
    p_files.set_defaults(func=lambda _args: list_files())

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
