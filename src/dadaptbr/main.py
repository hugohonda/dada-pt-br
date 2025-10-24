import argparse
import os

from dotenv import load_dotenv

from .config.datasets import DATASETS, DEFAULT_MODELS, TRANSLATION_MODELS


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
    """Download dataset from Hugging Face"""
    from .downloader import download_dataset

    try:
        filepath = download_dataset(args.target)
        print(f"Dataset downloaded successfully: {filepath}")
    except Exception as e:
        print(f"Download failed: {e}")
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
    evaluate_process(
        args.input_file, output, args.limit, args.device, batch_size=args.batch_size
    )


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


def handle_run(args):
    """Run full pipeline: translate → evaluate → merge → review."""
    from .evaluator import process_dataset as evaluate_process
    from .merger import merge_evaluations
    from .reviewer import process_dataset as review_process
    from .translator import process_dataset as translate_process
    from .utils import (
        generate_evaluation_filename,
        generate_merge_filename,
        generate_output_filename,
        generate_review_filename,
        get_dataset_id,
        get_timestamp,
        write_run_manifest,
    )

    pipeline_id = args.pipeline_id or get_timestamp()
    dataset_id = get_dataset_id(args.input_file)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    # Resolve dataset file if user passed a key or a non-existing path
    from .utils import resolve_dataset_file

    input_file = resolve_dataset_file(args.input_file)
    artifacts = []

    # 1) Translate per model
    translated_paths = []
    for model_name in models:
        out_translate = generate_output_filename(
            input_file, "translated", model_name, dataset_id, pipeline_id
        )
        translate_process(
            input_file,
            out_translate,
            model_name,
            args.workers,
            args.limit,
            args.device,
            pipeline_id,
        )
        translated_paths.append((model_name, out_translate))
        artifacts.append(
            {"phase": "translated", "model": model_name, "path": out_translate}
        )

    # 2) Evaluate each translation
    evaluated_paths = []
    for model_name, tpath in translated_paths:
        out_eval = generate_evaluation_filename(tpath)
        evaluate_process(
            tpath, out_eval, args.limit, args.device, pipeline_id, args.batch_size
        )
        evaluated_paths.append(out_eval)
        artifacts.append({"phase": "evaluated", "model": model_name, "path": out_eval})

    # 3) Merge best
    if len(evaluated_paths) >= 2:
        out_merge = generate_merge_filename(evaluated_paths[0], evaluated_paths[1])
        merge_evaluations(evaluated_paths[0], evaluated_paths[1], out_merge, args.limit)
        artifacts.append({"phase": "merged", "path": out_merge})
        review_input = out_merge
    else:
        # If only one, review that one
        review_input = evaluated_paths[0]

    # 4) Review
    out_review = generate_review_filename(review_input)
    # Choose first model for review LLM choice default
    review_model = models[0] if models else get_default_model("review")
    review_process(review_input, out_review, review_model, args.limit, args.workers)
    artifacts.append({"phase": "reviewed", "path": out_review})

    # Write manifest
    cfg = {
        "models": models,
        "workers": args.workers,
        "device": args.device,
        "batch_size": args.batch_size,
        "limit": args.limit,
    }
    manifest_path = write_run_manifest(pipeline_id, dataset_id, cfg, artifacts)

    print("Pipeline completed.")
    print(f"Manifest: {manifest_path}")


def add_common_args(
    parser, include_workers=False, include_device=False, include_model=False
):
    """Add common arguments to a parser."""
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of examples")
    if include_workers:
        parser.add_argument(
            "--workers", "-w", type=str, default="auto", help="Workers (int or 'auto')"
        )
    if include_device:
        parser.add_argument(
            "--device",
            "-d",
            default="auto",
            choices=["cuda", "cpu", "auto"],
            help="Device",
        )
    if include_model:
        parser.add_argument("--model", "-m", help="Model name")


def main():
    """Main CLI entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="dadaptbr CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download
    p_download = subparsers.add_parser(
        "download", help="Download dataset or model (simplified)"
    )
    p_download.add_argument("target", help="Dataset or model name from config")
    p_download.set_defaults(func=handle_download)

    # translate
    p_translate = subparsers.add_parser("translate", help="Translate dataset")
    p_translate.add_argument("input_file", help="Input JSON file")
    p_translate.add_argument("--model", "-m", default=get_default_model("translation"))
    add_common_args(p_translate, include_workers=True, include_device=True)
    p_translate.set_defaults(func=handle_translate)

    # evaluate
    p_evaluate = subparsers.add_parser("evaluate", help="Evaluate translation quality")
    p_evaluate.add_argument("input_file", help="Input JSON file")
    p_evaluate.add_argument("--batch-size", "-b", type=int, default=8)
    add_common_args(p_evaluate, include_device=True)
    p_evaluate.set_defaults(func=handle_evaluate)

    # review
    p_review = subparsers.add_parser("review", help="Review and improve translations")
    p_review.add_argument("input_file", help="Input JSON file")
    p_review.add_argument("--model", "-m", default=get_default_model("review"))
    add_common_args(p_review, include_workers=True)
    p_review.set_defaults(func=handle_review)

    # merge
    p_merge = subparsers.add_parser("merge", help="Merge evaluation results")
    p_merge.add_argument("file1", help="First evaluation file")
    p_merge.add_argument("file2", help="Second evaluation file")
    add_common_args(p_merge)
    p_merge.set_defaults(func=handle_merge)

    # Simple info commands
    subparsers.add_parser("list", help="List datasets").set_defaults(
        func=lambda _: list_datasets()
    )
    subparsers.add_parser("models", help="List models").set_defaults(
        func=lambda _: list_models()
    )
    subparsers.add_parser("files", help="List files").set_defaults(
        func=lambda _: list_files()
    )

    # run (full pipeline)
    p_run = subparsers.add_parser("run", help="Run full pipeline")
    p_run.add_argument("input_file", help="Input JSON file")
    p_run.add_argument(
        "--models",
        default=get_default_model("translation"),
        help="Comma-separated models",
    )
    p_run.add_argument("--batch-size", "-b", type=int, default=8)
    p_run.add_argument("--pipeline-id", help="Pipeline ID")
    add_common_args(p_run, include_workers=True, include_device=True)
    p_run.set_defaults(func=handle_run)

    args = parser.parse_args()
    # Normalize workers and device centrally
    from .utils import resolve_device, resolve_workers

    if hasattr(args, "workers"):
        phase = (
            "translation"
            if args.command == "translate"
            else ("review" if args.command == "review" else args.command)
        )
        try:
            args.workers = resolve_workers(args.workers, phase)
        except Exception:
            # fallback to existing value if parsing fails
            pass

    if hasattr(args, "device"):
        args.device = resolve_device(args.device)

    args.func(args)


if __name__ == "__main__":
    main()
