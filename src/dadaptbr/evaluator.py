import argparse
import logging
import os
import warnings

import torch
from comet import download_model, load_from_checkpoint
from tqdm import tqdm

from .config.datasets import FILE_PROCESSING
from .config.logging import setup_logger
from .utils import (
    ensure_directory_exists,
    generate_evaluation_filename,
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

        # Find the actual model directory using the same logic as downloader
        from .config.datasets import EVALUATION_MODELS

        model_name = "xcomet-xl"
        if model_name in EVALUATION_MODELS:
            repo_id = EVALUATION_MODELS[model_name]["hf_model_id"]
            safe_name = repo_id.replace("/", "_").replace(":", "_")
            model_dir = os.path.join("models", safe_name)
        else:
            # Fallback to old path structure
            model_dir = os.path.join("models", "xcomet-xl")

        # First try to find the model in our local models directory
        local_model_path = os.path.join(model_dir, "checkpoints", "model.ckpt")
        if os.path.exists(local_model_path):
            model_path = local_model_path
            _LOGGER.info(f"Using local model: {model_path}")
        else:
            # Check if model exists in Hugging Face cache
            cache_path = os.path.expanduser(
                "~/.cache/huggingface/hub/models--Unbabel--XCOMET-XL"
            )
            if os.path.exists(cache_path):
                # Find the latest snapshot
                snapshots_dir = os.path.join(cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = [
                        d
                        for d in os.listdir(snapshots_dir)
                        if os.path.isdir(os.path.join(snapshots_dir, d))
                    ]
                    if snapshots:
                        latest_snapshot = snapshots[
                            0
                        ]  # Usually the first one is the latest
                        cached_model_path = os.path.join(
                            snapshots_dir, latest_snapshot, "checkpoints", "model.ckpt"
                        )
                        if os.path.exists(cached_model_path):
                            model_path = cached_model_path
                            _LOGGER.info(f"Using cached model: {model_path}")
                        else:
                            _LOGGER.info("Model not found locally, downloading...")
                            model_path = download_model("Unbabel/XCOMET-XL")
                            _LOGGER.info(f"Model downloaded to: {model_path}")
                    else:
                        _LOGGER.info("Model not found locally, downloading...")
                        model_path = download_model("Unbabel/XCOMET-XL")
                        _LOGGER.info(f"Model downloaded to: {model_path}")
                else:
                    _LOGGER.info("Model not found locally, downloading...")
                    model_path = download_model("Unbabel/XCOMET-XL")
                    _LOGGER.info(f"Model downloaded to: {model_path}")
            else:
                _LOGGER.info("Model not found locally, downloading...")
                model_path = download_model("Unbabel/XCOMET-XL")
                _LOGGER.info(f"Model downloaded to: {model_path}")

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
        # Use CPU only for stability
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

        # Clear GPU memory after batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results, processing_time
    except Exception as e:
        _LOGGER.error(f"Error in batch evaluation: {e}")
        processing_time = time.time() - start_time
        return [
            {"score": 0.0, "system_score": 0.0, "error_spans": None} for _ in data_batch
        ], processing_time


def process_dataset(
    input_file, output_file, limit=None, device="cpu", pipeline_id=None
):
    """Process dataset for evaluation."""
    import time

    start_time = time.time()

    raw_data = load_json_file(input_file)
    if not raw_data:
        _LOGGER.error("Empty dataset")
        return

    # Handle new data structure with metadata
    if isinstance(raw_data, dict) and "data" in raw_data:
        data = raw_data["data"]
        metadata = raw_data.get("metadata", {})
        _LOGGER.info(f"Loaded data with metadata: {metadata}")
    else:
        # Handle old data structure (list of examples)
        data = raw_data
        metadata = {}

    if limit:
        data = data[:limit]

    # Use dataset_id from metadata if available, otherwise extract from filename
    if metadata and "dataset_id" in metadata:
        dataset_id = metadata["dataset_id"]
    else:
        dataset_id = get_dataset_id(input_file)

    # Generate pipeline ID if not provided
    if pipeline_id is None:
        pipeline_id = get_timestamp()

    _LOGGER.info(
        f"Evaluating {len(data)} examples from dataset: {dataset_id} - Pipeline ID: {pipeline_id}"
    )

    try:
        model = load_xcomet_model()
    except Exception as e:
        _LOGGER.error(f"Failed to load XCOMET model: {e}")
        _LOGGER.info("Evaluation cannot proceed without the model")
        return

    # Prepare data for batch processing
    xcomet_data = []
    valid_indices = []
    results = []

    for i, example in enumerate(data):
        # Extract source and translation using simplified logic
        source = example.get("en", "")
        translation = example.get("pt-br", "")

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

    # Generate output file
    from .utils import extract_pipeline_id, generate_output_filename

    pipeline_id = extract_pipeline_id(input_file)
    # Use clean dataset_id from metadata, not from filename
    clean_dataset_id = (
        metadata.get("dataset_id", dataset_id) if metadata else dataset_id
    )
    proper_output_file = generate_output_filename(
        input_file, "evaluated", None, clean_dataset_id, pipeline_id
    )

    ensure_directory_exists(os.path.dirname(proper_output_file))

    # Evaluate data in batches
    batch_processing_times = []
    batch_size = FILE_PROCESSING["default_batch_size"]  # Default batch size
    if xcomet_data:
        _LOGGER.info(f"Evaluating {len(xcomet_data)} examples in batches...")

        for i in tqdm(
            range(0, len(xcomet_data), batch_size), desc="Evaluating batches"
        ):
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

    # Calculate standard deviation
    std_score = 0.0
    if len(scores) > 1:
        import numpy as np

        std_score = round(np.std(scores), 4)

    # Calculate total processing time
    total_processing_time = time.time() - start_time

    # Save results with metadata
    _LOGGER.info("Saving evaluation results...")

    # Calculate evaluation statistics
    scores = [item.get("score", 0) for item in results if item.get("score") is not None]
    avg_score = sum(scores) / len(scores) if scores else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0

    # Create evaluation metadata with analysis using consolidated approach
    from .metadata_utils import create_evaluation_metadata

    eval_metadata = create_evaluation_metadata(
        pipeline_id=pipeline_id,
        dataset_id=dataset_id,
        total_examples=len(results),
        processing_time=total_processing_time,
        results=results,
        batch_size=batch_size,
        total_batches=len(batch_processing_times),
        model_name="xcomet-xl",
    )

    # Create final data structure
    final_data = {"metadata": eval_metadata, "data": results}

    save_json_file(final_data, proper_output_file)

    _LOGGER.info(f"Evaluated data saved to {proper_output_file}")
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

    output_file = args.output or generate_evaluation_filename(
        args.input_file,
        None,  # Don't include model name in evaluation filename
    )
    process_dataset(args.input_file, output_file, args.limit)


if __name__ == "__main__":
    main()
