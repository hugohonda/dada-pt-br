#!/usr/bin/env python3
"""
Translation quality evaluator / scorer.
"""

import argparse
import logging
import os
import warnings

from comet import load_from_checkpoint

from .config.logging import setup_logger
from .utils import (
    detect_dataset_type,
    ensure_directory_exists,
    extract_texts,
    generate_evaluation_filename,
    generate_report_filename,
    get_dataset_id,
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

        # Find the actual model directory using the same logic as downloader
        from .config.datasets import MODELS

        model_name = "xcomet-xl"
        if model_name in MODELS:
            repo_id = MODELS[model_name]
            safe_name = repo_id.replace("/", "_").replace(":", "_")
            model_dir = os.path.join("models", safe_name)
        else:
            # Fallback to old path structure
            model_dir = os.path.join("models", "xcomet-xl")

        model_path = os.path.join(model_dir, "checkpoints", "model.ckpt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        model = load_from_checkpoint(model_path)
        _LOGGER.info("Model loaded successfully")
        return model

    except Exception as e:
        _LOGGER.error(f"Error loading XCOMET-XL model: {e}")
        _LOGGER.error("Please run: uv run dada download xcomet-xl")
        raise


def evaluate_batch(model, data_batch, batch_size=8):
    """Evaluate a batch of translations using XCOMET-XL."""
    import time

    start_time = time.time()

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

        processing_time = time.time() - start_time
        return results, processing_time
    except Exception as e:
        _LOGGER.error(f"Error in batch evaluation: {e}")
        processing_time = time.time() - start_time
        return [
            {"score": 0.0, "system_score": 0.0, "error_spans": None} for _ in data_batch
        ], processing_time


def process_dataset(input_file, output_file, limit=None):
    """Process dataset for evaluation."""
    import time

    start_time = time.time()

    data = load_json_file(input_file)
    if not data:
        _LOGGER.error("Empty dataset")
        return

    if limit:
        data = data[:limit]

    dataset_id = get_dataset_id(input_file)
    dataset_type = detect_dataset_type(data[0], dataset_id)

    # Extract model key from input file path for proper folder structure
    from .config.datasets import TRANSLATION_MODELS

    # Extract model key from directory path (e.g., data/translated/gemma3/...)
    model_key = None
    path_parts = input_file.split(os.sep)
    for part in path_parts:
        if part in TRANSLATION_MODELS.keys():
            model_key = part
            break

    _LOGGER.info(
        f"Evaluating {len(data)} examples ({dataset_type}) with model: {model_key or 'unknown'}"
    )

    model = load_xcomet_model()

    # Prepare data for batch processing
    xcomet_data = []
    valid_indices = []
    results = []

    for i, example in enumerate(data):
        source, translation = extract_texts(example, dataset_type, dataset_id)

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

    # Generate proper output file with model key for correct folder structure
    from .utils import extract_pipeline_id, generate_output_filename

    # Extract pipeline_id from input file to maintain consistency
    pipeline_id = extract_pipeline_id(input_file)
    proper_output_file = generate_output_filename(
        input_file, "evaluated", model_key, dataset_id, pipeline_id
    )

    # Setup output files
    ensure_directory_exists(os.path.dirname(proper_output_file))

    # Evaluate data in batches with optimized processing
    batch_processing_times = []
    if xcomet_data:
        _LOGGER.info(f"Evaluating {len(xcomet_data)} examples in batches...")
        batch_size = 16  # Increased batch size for better GPU utilization

        # Process batches and collect results in order
        for i in range(0, len(xcomet_data), batch_size):
            batch = xcomet_data[i : i + batch_size]
            batch_results, batch_time = evaluate_batch(model, batch, batch_size)
            batch_processing_times.append(batch_time)

            # Update results with actual scores in order
            for j, batch_result in enumerate(batch_results):
                original_index = valid_indices[i + j]
                results[original_index].update(batch_result)

    # Calculate statistics
    scores = [r["score"] for r in results if r["score"] > 0]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    # Save results directly to JSON
    _LOGGER.info("Saving evaluation results...")
    save_json_file(results, proper_output_file)

    # Generate standardized evaluation report
    from .report_generator import generate_standard_reports

    model_name = model_key or "unknown"
    report_file = generate_report_filename(
        dataset_id, "evaluation", model_name, extension="json", pipeline_id=pipeline_id
    )
    summary_file = generate_report_filename(
        dataset_id, "evaluation", model_name, extension="txt", pipeline_id=pipeline_id
    )

    # Calculate standard deviation
    std_score = 0.0
    if len(scores) > 1:
        import numpy as np

        std_score = round(np.std(scores), 4)

    # Use batch processing times for more detailed metrics
    processing_times = (
        batch_processing_times if batch_processing_times else [time.time() - start_time]
    )

    # Generate standardized reports
    generate_standard_reports(
        operation="evaluation",
        input_file=input_file,
        output_file=output_file,
        dataset_type=dataset_type,
        model_name="XCOMET-XL",
        pipeline_id=pipeline_id,
        report_file=report_file,
        summary_file=summary_file,
        total_examples=len(data),
        evaluated_examples=len(scores),
        limit=limit,
        mean_score=round(mean_score, 4),
        min_score=round(min_score, 4),
        max_score=round(max_score, 4),
        std_score=std_score,
        processing_times=processing_times,
        batch_size=batch_size,
        total_batches=len(batch_processing_times),
    )

    _LOGGER.info(f"Evaluated data saved to {proper_output_file}")
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

    # Extract model key from input filename for proper folder structure
    input_filename = os.path.basename(args.input_file)
    model_key = None
    # Simple extraction: look for common model patterns in filename
    if "gemma" in input_filename.lower():
        model_key = "gemma3"
    elif "tower" in input_filename.lower():
        model_key = "towerinstruct"

    output_file = args.output or generate_evaluation_filename(
        args.input_file, model_key
    )
    process_dataset(args.input_file, output_file, args.limit)


if __name__ == "__main__":
    main()
