import argparse
import logging
import os
import time
import warnings

import torch
from comet import load_from_checkpoint
from huggingface_hub import snapshot_download
from tqdm import tqdm

from .config.datasets import EVALUATION_MODELS, FILE_PROCESSING
from .config.logging import log_model_info, setup_logger
from .utils import (
    ensure_directory_exists,
    extract_pipeline_id,
    generate_evaluation_filename,
    generate_output_filename,
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


def load_xcomet_model(model_name: str = None):
    """Load evaluation model using HuggingFace cache.

    Args:
        model_name: Name of the model from EVALUATION_MODELS config.
                   If None, uses the default model.
    """
    try:
        # Get default model if not specified
        if model_name is None:
            model_name = next(
                (
                    name
                    for name, config in EVALUATION_MODELS.items()
                    if config.get("default", False)
                ),
                next(iter(EVALUATION_MODELS.keys()), None),
            )
            if model_name is None:
                raise ValueError("No evaluation models configured")

        if model_name not in EVALUATION_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list[str](EVALUATION_MODELS.keys())}"
            )

        model_config = EVALUATION_MODELS[model_name]
        repo_id = model_config["hf_model_id"]
        display_name = model_config.get("display_name", model_name)

        _LOGGER.info(f"Loading {display_name} model...")

        model_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=None,  # Use default HF cache
            local_files_only=False,  # Allow download if not cached
        )

        checkpoint_path = os.path.join(model_dir, "checkpoints", "model.ckpt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        model = load_from_checkpoint(checkpoint_path)
        _LOGGER.info(f"{display_name} model loaded successfully")
        return model

    except Exception as e:
        _LOGGER.error(f"Error loading evaluation model: {e}")
        _LOGGER.error(f"Please run: uv run dada download {model_name}")
        raise


def evaluate_batch(model, data_batch, batch_size=8):
    """Evaluate a batch of translations using XCOMET-XL."""
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
    input_file,
    output_file,
    limit=None,
    device="cpu",
    pipeline_id=None,
    batch_size=None,
):
    """Process dataset for evaluation."""
    start_time = time.time()

    data, metadata = load_json_file(input_file)
    if not data:
        _LOGGER.error("Empty dataset")
        return

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

    # Determine which model to use (get default from config)
    model_name = next(
        (
            name
            for name, config in EVALUATION_MODELS.items()
            if config.get("default", False)
        ),
        next(iter(EVALUATION_MODELS.keys()), None),
    )
    if model_name is None:
        _LOGGER.error("No evaluation models configured")
        return

    model_config = EVALUATION_MODELS.get(model_name, {})
    batch_size = batch_size or FILE_PROCESSING["default_batch_size"]
    log_model_info(
        _LOGGER,
        "evaluation",
        model_name,
        model_config,
        batch_size=batch_size,
        device=device,
    )

    try:
        model = load_xcomet_model(model_name)
    except Exception as e:
        _LOGGER.error(f"Failed to load evaluation model: {e}")
        _LOGGER.info("Evaluation cannot proceed without the model")
        return

    # Prepare data for batch processing
    xcomet_data = []
    valid_indices = []
    results = []

    for i, example in enumerate(data):
        source = example.get("en", "")
        translation = example.get("pt-br", "")

        # Create result entry
        result = {
            "index": i,
            "id": example.get("id", i),
            "source": source,
            "translation": translation,
            "score": 0.0,
            "system_score": 0.0,
            "error_spans": None,
        }

        if not source or not translation:
            result["error"] = "Missing source or translation"
        else:
            xcomet_data.append({"src": source, "mt": translation})
            valid_indices.append(i)

        results.append(result)

    # Determine output file path: honor provided output_file if given
    pipeline_id = extract_pipeline_id(input_file)
    clean_dataset_id = (
        metadata.get("dataset_id", dataset_id) if metadata else dataset_id
    )
    proper_output_file = (
        output_file
        if output_file
        else generate_output_filename(
            input_file, "evaluated", None, clean_dataset_id, pipeline_id
        )
    )

    ensure_directory_exists(os.path.dirname(proper_output_file))

    # Evaluate data in batches
    if batch_size is None:
        batch_size = FILE_PROCESSING["default_batch_size"]
    if xcomet_data:
        _LOGGER.info(f"Evaluating {len(xcomet_data)} examples in batches...")

        for i in tqdm(
            range(0, len(xcomet_data), batch_size), desc="Evaluating batches"
        ):
            batch = xcomet_data[i : i + batch_size]
            batch_results, _ = evaluate_batch(model, batch, batch_size)

            # Update results with actual scores in order
            for j, batch_result in enumerate(batch_results):
                original_index = valid_indices[i + j]
                results[original_index].update(batch_result)

    # Calculate total processing time
    total_processing_time = time.time() - start_time

    # Save results with metadata
    _LOGGER.info("Saving evaluation results...")

    # Calculate evaluation statistics
    scores = [r["score"] for r in results if r["score"] > 0]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    # Create evaluation metadata with analysis using consolidated approach
    from .metadata_utils import create_evaluation_metadata

    eval_metadata = create_evaluation_metadata(
        pipeline_id=pipeline_id,
        dataset_id=dataset_id,
        total_examples=len(results),
        processing_time=total_processing_time,
        results=results,
        batch_size=batch_size,
        model_name=model_name,
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
