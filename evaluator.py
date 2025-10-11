#!/usr/bin/env python3
"""
Translation quality evaluator / scorer.
"""

import argparse
import json
import logging
import os
import warnings

from comet import load_from_checkpoint

from config.logging import setup_logger
from utils import (
    detect_dataset_type,
    ensure_directory_exists,
    extract_texts,
    generate_evaluation_filename,
    generate_report_filename,
    get_dataset_id,
    get_timestamp,
    load_json_file,
    save_json_file,
    validate_file_exists,
)

warnings.filterwarnings("ignore", category=UserWarning)

_LOGGER = setup_logger("evaluator", log_to_file=True, log_prefix="evaluation")

# Silence verbose logs
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("torchmetrics").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


def load_xcomet_model():
    """Load XCOMET-XL model from local models folder."""
    try:
        _LOGGER.info("Loading XCOMET-XL model...")
        model_path = os.path.join("models", "xcomet-xl", "checkpoints", "model.ckpt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = load_from_checkpoint(model_path)
        _LOGGER.info("Model loaded successfully")
        return model

    except Exception as e:
        _LOGGER.error(f"Error loading XCOMET-XL model: {e}")
        _LOGGER.error("Please run: uv run main.py download xcomet-xl")
        raise


def generate_evaluation_summary(report, summary_file):
    """Generate a human-readable evaluation summary report."""
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("TRANSLATION EVALUATION REPORT SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # Evaluation Summary
        summary = report["evaluation_summary"]
        f.write("EVALUATION SUMMARY:\n")
        f.write(f"  Timestamp: {summary['timestamp']}\n")
        f.write(f"  Input File: {summary['input_file']}\n")
        f.write(f"  Output File: {summary['output_file']}\n")
        f.write(f"  Dataset Type: {summary['dataset_type']}\n")
        f.write(f"  Total Examples: {summary['total_examples']}\n")
        f.write(f"  Evaluated Examples: {summary['evaluated_examples']}\n")
        f.write(f"  Limit: {summary['limit'] or 'All'}\n")
        f.write(f"  Mean Score: {summary['mean_score']}\n")
        f.write(f"  Min Score: {summary['min_score']}\n")
        f.write(f"  Max Score: {summary['max_score']}\n\n")

        # Model Information
        f.write("MODEL INFORMATION:\n")
        model = report["model_information"]
        for key, value in model.items():
            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("End of Report\n")
        f.write("=" * 60 + "\n")


def evaluate_batch(model, data_batch, batch_size=8):
    """Evaluate a batch of translations using XCOMET-XL."""
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
        _LOGGER.error(f"Error in batch evaluation: {e}")
        return [
            {"score": 0.0, "system_score": 0.0, "error_spans": None} for _ in data_batch
        ]


def process_dataset(input_file, output_file, limit=None):
    """Process dataset for evaluation."""
    data = load_json_file(input_file)
    if not data:
        _LOGGER.error("Empty dataset")
        return

    if limit:
        data = data[:limit]

    dataset_type = detect_dataset_type(data[0])
    _LOGGER.info(f"Evaluating {len(data)} examples ({dataset_type})")

    model = load_xcomet_model()

    # Prepare data for batch processing
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
                "id": example.get("id", i),
                "source": source,
                "translation": translation,
                "score": 0.0,
                "system_score": 0.0,
                "error_spans": None,
            }
        )

        if not source or not translation:
            results[i]["error"] = "Missing source or translation"

    # Setup output files
    ensure_directory_exists(os.path.dirname(output_file))
    jsonl_file = output_file.replace(".json", ".jsonl")

    # Evaluate data in batches
    if xcomet_data:
        _LOGGER.info(f"Evaluating {len(xcomet_data)} examples in batches...")
        batch_size = 8

        with open(jsonl_file, "w", encoding="utf-8") as f:
            for i in range(0, len(xcomet_data), batch_size):
                batch = xcomet_data[i : i + batch_size]
                batch_results = evaluate_batch(model, batch, batch_size)

                # Update results with actual scores
                for j, batch_result in enumerate(batch_results):
                    original_index = valid_indices[i + j]
                    results[original_index].update(batch_result)

                # Save to JSONL
                for j in range(len(batch_results)):
                    original_index = valid_indices[i + j]
                    f.write(
                        json.dumps(results[original_index], ensure_ascii=False) + "\n"
                    )
                f.flush()

    # Calculate statistics
    scores = [r["score"] for r in results if r["score"] > 0]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    # Save files
    save_json_file(results, output_file)

    # Generate concise evaluation report
    dataset_id = get_dataset_id(input_file)
    report_file = generate_report_filename(dataset_id, "evaluation", extension="json")
    summary_file = generate_report_filename(dataset_id, "evaluation", extension="txt")

    # Create concise report
    timestamp = get_timestamp()
    report = {
        "evaluation_summary": {
            "timestamp": timestamp,
            "input_file": input_file,
            "output_file": output_file,
            "dataset_type": dataset_type,
            "total_examples": len(data),
            "evaluated_examples": len(scores),
            "limit": limit,
            "mean_score": round(mean_score, 4),
            "min_score": round(min_score, 4),
            "max_score": round(max_score, 4),
        },
        "model_information": {
            "name": "XCOMET-XL",
            "repo_id": "Unbabel/XCOMET-XL",
            "local_path": "models/xcomet-xl",
        },
    }

    save_json_file(report, report_file)

    # Generate human-readable summary
    generate_evaluation_summary(report, summary_file)

    _LOGGER.info(f"Evaluated data saved to {output_file}")
    _LOGGER.info(f"Line-by-line backup saved to {jsonl_file}")
    _LOGGER.info(f"Evaluation report saved to {report_file}")
    _LOGGER.info(f"Evaluation summary saved to {summary_file}")
    _LOGGER.info(f"Average score: {mean_score:.4f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Translation Quality Evaluator")
    parser.add_argument("input_file", help="Input JSON file")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--limit", "-l", type=int, help="Limit examples (default: all)")

    args = parser.parse_args()

    if not validate_file_exists(args.input_file):
        _LOGGER.error(f"File not found: {args.input_file}")
        return

    output_file = args.output or generate_evaluation_filename(args.input_file)
    process_dataset(args.input_file, output_file, args.limit)


if __name__ == "__main__":
    main()
