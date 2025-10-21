import argparse
import os

from dotenv import load_dotenv

from .config.datasets import DATASETS, DEFAULT_MODELS, TRANSLATION_MODELS, PHASE_WORKERS


def get_default_model(operation: str) -> str:
    """Get default model for a specific operation."""
    return DEFAULT_MODELS.get(operation, "tower")


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
    for name, config in TRANSLATION_MODELS.items():
        print(f"{name}: {config['display_name']} ({config['ollama_name']})")


def list_files():
    """List downloaded files."""
    import glob

    print("Available files:")
    print("  Data files:")
    for file in sorted(glob.glob("output/**/*.json", recursive=True)):
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # MB
            print(f"    • {file} ({size:.1f} MB)")

    print("  Reports:")
    for file in sorted(glob.glob("output/*_report.json", recursive=True)):
        if os.path.exists(file):
            print(f"    • {file}")
    for file in sorted(glob.glob("output/*_summary.txt", recursive=True)):
        if os.path.exists(file):
            print(f"    • {file}")

    print("  Logs:")
    for file in sorted(glob.glob("logs/**/*.log", recursive=True)):
        if os.path.exists(file):
            print(f"    • {file}")


def handle_download(args):
    """Download functionality removed - datasets should be placed in datasets/raw/"""
    print("Download functionality removed.")
    print("Please place your datasets in the datasets/raw/ directory.")
    print("Available datasets:")
    list_datasets()


def handle_translate(args):
    # Lazy import to avoid heavy deps when not needed
    from .translator import process_dataset as translate_process
    from .utils import generate_output_filename

    # Use model name directly - simplified approach
    model_name = args.model
    output_file = args.output or generate_output_filename(
        args.input_file, "translated", model_name
    )

    translate_process(
        args.input_file,
        output_file,
        model_name,
        args.workers,
        args.limit,
        args.device,
    )


def handle_evaluate(args):
    # Lazy import to avoid heavy deps when not needed
    from .evaluator import process_dataset as evaluate_process
    from .utils import generate_evaluation_filename

    output = args.output or generate_evaluation_filename(args.input_file)
    evaluate_process(args.input_file, output, args.limit, args.device)


def handle_review(args):
    # Lazy import to avoid heavy deps when not needed
    from .reviewer import process_dataset as review_process
    from .utils import generate_review_filename

    output = args.output or generate_review_filename(args.input_file)
    review_process(args.input_file, output, args.model, args.limit, args.workers)


def handle_merge(args):
    # Lazy import to avoid heavy deps when not needed
    from .merger import merge_evaluations
    from .utils import generate_merge_filename

    output = args.output or generate_merge_filename(args.file1, args.file2)
    merge_evaluations(args.file1, args.file2, output, args.limit)


def download_models_cmd():
    """Download all models automatically - simplified."""
    print("Models are downloaded automatically when first used.")
    print("Available models:")
    list_models()


def check_models():
    """Check which models are available - simplified."""
    print("Available models:")
    list_models()
    print("\nModels are downloaded automatically when first used.")


def main():
    """Main CLI entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="dadaptbr CLI",
        epilog=(
            "Examples:\n"
            "  dada download m_alert\n"
            "  dada translate datasets/raw/file.json --model tower --limit=100\n"
            "  dada evaluate output/translated/file.json --limit=100\n"
            "  dada merge file1.json file2.json --limit=100\n"
            "  dada review output/evaluated/merged_file.json --limit=100\n"
            "  dada list\n  dada models\n  dada files"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # download
    p_download = subparsers.add_parser(
        "download", help="Download dataset or model (simplified)"
    )
    p_download.add_argument("target", help="Dataset or model name from config")
    p_download.set_defaults(func=handle_download)

    # translate
    p_translate = subparsers.add_parser("translate", help="Translate a dataset file")
    p_translate.add_argument("input_file", help="Input JSON file to translate")
    p_translate.add_argument("--output", "-o", help="Output JSON file")
    p_translate.add_argument(
        "--workers", "-w", type=int, default=PHASE_WORKERS["translation"]["default"], help="Number of parallel workers"
    )
    p_translate.add_argument("--limit", "-l", type=int, help="Limit examples")
    p_translate.add_argument(
        "--model",
        "-m",
        default=get_default_model("translation"),
        help="Translation model to use (default: tower).",
    )
    p_translate.add_argument(
        "--device",
        "-d",
        default="cpu",
        choices=["cuda", "cpu", "auto"],
        help="Device to use (default: cpu).",
    )
    p_translate.set_defaults(func=handle_translate)

    # evaluate
    p_evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate translation quality with XCOMET-XL"
    )
    p_evaluate.add_argument("input_file", help="Input JSON file to evaluate")
    p_evaluate.add_argument("--output", "-o", help="Output JSON file")
    p_evaluate.add_argument("--limit", "-l", type=int, help="Limit examples")
    p_evaluate.add_argument(
        "--device",
        "-d",
        default="cpu",
        choices=["cuda", "cpu", "auto"],
        help="Device to use (default: cpu).",
    )
    p_evaluate.set_defaults(func=handle_evaluate)

    # review
    p_review = subparsers.add_parser(
        "review", help="Review and improve translations based on evaluation results"
    )
    p_review.add_argument("input_file", help="Input evaluation JSON file to review")
    p_review.add_argument("--output", "-o", help="Output JSON file")
    p_review.add_argument("--limit", "-l", type=int, help="Limit examples")
    p_review.add_argument(
        "--workers", "-w", type=int, default=PHASE_WORKERS["review"]["default"], help="Number of parallel workers"
    )
    p_review.add_argument(
        "--model",
        "-m",
        default=get_default_model("review"),
        help="Model to use for review (default: gemma3).",
    )
    p_review.set_defaults(func=handle_review)

    # merge
    p_merge = subparsers.add_parser(
        "merge", help="Merge evaluation results, keeping the best translations"
    )
    p_merge.add_argument("file1", help="First evaluation JSON file")
    p_merge.add_argument("file2", help="Second evaluation JSON file")
    p_merge.add_argument("--output", "-o", help="Output JSON file")
    p_merge.add_argument("--limit", "-l", type=int, help="Limit examples")
    p_merge.set_defaults(func=handle_merge)

    # list datasets
    p_list = subparsers.add_parser("list", help="List available datasets")
    p_list.set_defaults(func=lambda _args: list_datasets())

    # models
    p_models = subparsers.add_parser("models", help="List available models")
    p_models.set_defaults(func=lambda _args: list_models())

    # files
    p_files = subparsers.add_parser("files", help="List downloaded files")
    p_files.set_defaults(func=lambda _args: list_files())

    # download models (simplified)
    p_download_models = subparsers.add_parser(
        "download-models", help="Download all models automatically (simplified)"
    )
    p_download_models.set_defaults(func=lambda _args: download_models_cmd())

    # check models (simplified)
    p_check = subparsers.add_parser(
        "check-models", help="Check which models are available (simplified)"
    )
    p_check.set_defaults(func=lambda _args: check_models())

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
