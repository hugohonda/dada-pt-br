#!/usr/bin/env python3
"""
CLI interface DADA-PT-BR
"""

import argparse
import os

from dotenv import load_dotenv

from .config.datasets import DATASETS, MODELS
from .llm_client import get_ollama_name


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
    from .downloader import download_dataset, download_model

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
    from .translator import process_dataset as translate_process
    from .utils import generate_output_filename

    model_name = get_ollama_name(args.model)
    output_file = args.output or generate_output_filename(
        args.input_file, "translated", model_name
    )

    translate_process(
        args.input_file,
        output_file,
        model_name,
        args.workers,
        args.limit,
    )


def handle_evaluate(args):
    # Lazy import to avoid heavy deps when not needed
    # Extract model key from input filename for proper folder structure
    import os

    from .evaluator import process_dataset as evaluate_process
    from .utils import generate_evaluation_filename

    input_filename = os.path.basename(args.input_file)
    model_key = None
    if "gemma" in input_filename.lower():
        model_key = "gemma3"
    elif "tower" in input_filename.lower():
        model_key = "towerinstruct"

    output = args.output or generate_evaluation_filename(args.input_file, model_key)
    evaluate_process(args.input_file, output, args.limit)


def handle_analyze(args):
    """Handle analyze command."""
    # Lazy import to avoid heavy deps when not needed
    from .analyser import main as analyze_main
    from .utils import get_dataset_id

    dataset_id = args.dataset or get_dataset_id(args.input_file)
    analyze_main(dataset_id)


def handle_single_analyze(args):
    """Handle single-analyze command."""
    from .single_analyzer import main as single_analyze_main

    single_analyze_main(args.evaluation_file)


def handle_compare(args):
    """Handle compare command."""
    from .comparison_analyzer import main as compare_main

    compare_main(args.evaluation_files)


def main():
    """Main CLI entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="dadaptbr CLI",
        epilog=(
            "Examples:\n"
            "  dada download m_alert\n"
            "  dada download xcomet-xl\n"
            "  dada translate datasets/raw/file.json --model towerinstruct --limit=100\n"
            "  dada evaluate datasets/processed/file.json --limit=100\n"
            "  dada list\n  dada models\n  dada files"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # download
    p_download = subparsers.add_parser("download", help="Download dataset or model")
    p_download.add_argument("target", help="Dataset or model name from config")
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
        "--model",
        "-m",
        choices=["gemma3", "towerinstruct"],
        default="gemma3",
        help="Translation model to use (default: gemma3)",
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

    # analyze
    p_analyze = subparsers.add_parser(
        "analyze", help="Analyze translation and evaluation results"
    )
    p_analyze.add_argument("input_file", help="Input JSON file to analyze")
    p_analyze.add_argument(
        "--dataset", "-d", help="Dataset ID (auto-detected if not provided)"
    )
    p_analyze.set_defaults(func=handle_analyze)

    # single-analyze
    p_single_analyze = subparsers.add_parser(
        "single-analyze", help="Analyze a single evaluation file"
    )
    p_single_analyze.add_argument(
        "evaluation_file", help="Evaluation JSON file to analyze"
    )
    p_single_analyze.set_defaults(func=handle_single_analyze)

    # compare
    p_compare = subparsers.add_parser(
        "compare", help="Compare multiple evaluation files"
    )
    p_compare.add_argument(
        "evaluation_files", nargs="+", help="Evaluation JSON files to compare"
    )
    p_compare.set_defaults(func=handle_compare)

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
