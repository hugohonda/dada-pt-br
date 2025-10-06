import argparse
import json
import logging
import os
import warnings
from datetime import datetime

from comet import download_model, load_from_checkpoint

from config.logging import setup_logger

warnings.filterwarnings("ignore", category=UserWarning)

_LOGGER = setup_logger("evaluator", log_to_file=True, log_prefix="evaluation")

# Silence verbose logs
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("torchmetrics").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


def get_timestamp():
    """Get timestamp for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_output_filename(input_file: str) -> str:
    """Generate output filename with timestamp pattern for evaluation."""
    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = get_timestamp()

    # Create evaluation directory path
    evaluation_dir = "datasets/evaluation"
    os.makedirs(evaluation_dir, exist_ok=True)

    return os.path.join(evaluation_dir, f"{base_name}_evaluated_{timestamp}.json")


def detect_dataset_type(example: dict) -> str:
    """Detect dataset type."""
    keys = set(example.keys())

    if (
        "fr" in keys
        and "de" in keys
        and "es" in keys
        and "it" in keys
        and "en" in keys
        and "pt" in keys
    ):
        return "multilingual"
    elif "category" in keys and "prompt" in keys:
        return "safety"
    else:
        return "single"


def extract_texts(example: dict, dataset_type: str) -> tuple:
    """Extract source and translation texts."""
    if dataset_type == "multilingual":
        return example.get("en", ""), example.get("pt", "")
    elif dataset_type == "safety":
        return example.get("prompt", ""), example.get("prompt", "")
    else:
        return example.get("text", example.get("content", "")), example.get(
            "translation", ""
        )


def evaluate_batch_with_xcomet(model, data_batch, batch_size=8) -> list:
    """Evaluate a batch of translations using XCOMET-XL (CPU only)."""
    try:
        output = model.predict(data_batch, batch_size=batch_size, gpus=0)

        results = []
        for i, score in enumerate(output.scores):
            result = {
                "score": float(score),
                "system_score": float(output.system_score),
                "error_spans": output.metadata.error_spans[i]
                if hasattr(output.metadata, "error_spans")
                and i < len(output.metadata.error_spans)
                else None,
            }
            results.append(result)

        return results
    except Exception as e:
        print(f"Error in batch evaluation: {e}")
        # Return zero scores for all items in batch
        return [
            {"score": 0.0, "system_score": 0.0, "error_spans": None} for _ in data_batch
        ]


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Translation Quality Evaluator")
    parser.add_argument("input_file", help="Input JSON file")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--limit", "-l", type=int, help="Limit examples (default: all)")

    args = parser.parse_args()

    # Generate single timestamp for both log and report files
    timestamp = get_timestamp()

    if not os.path.exists(args.input_file):
        _LOGGER.error(f"File not found: {args.input_file}")
        return

    # Load data
    with open(args.input_file, encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        _LOGGER.error("Empty dataset")
        return

    # Limit examples if specified
    if args.limit:
        data = data[: args.limit]
    dataset_type = detect_dataset_type(data[0])

    _LOGGER.info(f"Evaluating {len(data)} examples ({dataset_type})")

    # Load model
    _LOGGER.info("Loading XCOMET-XL...")
    try:
        model_path = download_model("Unbabel/XCOMET-XL")
        model = load_from_checkpoint(model_path)
        _LOGGER.info("Model loaded")
    except Exception as e:
        _LOGGER.error(f"✗ Error loading model: {e}")
        return

    _LOGGER.info("Using CPU evaluation")

    # Prepare all data for XCOMET batch processing
    _LOGGER.info(f"Preparing {len(data)} examples for batch evaluation...")

    xcomet_data = []
    valid_indices = []
    results = []

    for i, example in enumerate(data):
        source, translation = extract_texts(example, dataset_type)

        if source and translation:
            xcomet_data.append({"src": source, "mt": translation})
            valid_indices.append(i)

        results.append(
            {
                "index": i,
                "id": example.get(
                    "id", i
                ),  # Preserve original ID or use index as fallback
                "source": source,
                "translation": translation,
                "score": 0.0,  # Will be filled by batch processing
                "system_score": 0.0,
                "error_spans": None,
            }
        )

        if not source or not translation:
            results[i]["error"] = "Missing source or translation"

    # Setup output files
    output_file = args.output or generate_output_filename(args.input_file)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # JSONL file for line-by-line saving (use same timestamp as output)
    jsonl_file = output_file.replace(".json", ".jsonl")

    # Evaluate data in batches (XCOMET will show its own progress bar)
    if xcomet_data:
        _LOGGER.info(f"Evaluating {len(xcomet_data)} examples in batches...")

        batch_size = 8

        with open(jsonl_file, "w", encoding="utf-8") as f:
            for i in range(0, len(xcomet_data), batch_size):
                batch = xcomet_data[i : i + batch_size]
                batch_results = evaluate_batch_with_xcomet(model, batch, batch_size)

                # Update results with actual scores
                for j, batch_result in enumerate(batch_results):
                    # Map batch index back to original result index
                    original_index = valid_indices[i + j]
                    results[original_index].update(batch_result)

                # Save only the newly processed results to JSONL
                for j in range(len(batch_results)):
                    original_index = valid_indices[i + j]
                    f.write(
                        json.dumps(results[original_index], ensure_ascii=False) + "\n"
                    )
                f.flush()  # Ensure data is written to disk

    # Calculate stats
    scores = [r["score"] for r in results if r["score"] > 0]

    summary = {
        "dataset_info": {
            "input_file": args.input_file,
            "dataset_type": dataset_type,
            "total_examples": len(data),
            "evaluated": len(results),
            "limit": args.limit,
        },
        "statistics": {
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "total_evaluated": len(scores),
        },
        "results": results,
    }

    # Save final JSON to evaluation folder
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save report to reports folder
    report_file = f"reports/{os.path.splitext(os.path.basename(args.input_file))[0]}_evaluation_report_{timestamp}.json"
    os.makedirs("reports", exist_ok=True)

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _LOGGER.info(f"✓ Evaluated data saved to {output_file}")
    _LOGGER.info(f"✓ Line-by-line backup saved to {jsonl_file}")
    _LOGGER.info(f"✓ Evaluation report saved to {report_file}")
    _LOGGER.info(f"Average score: {summary['statistics']['mean_score']:.4f}")


if __name__ == "__main__":
    main()
