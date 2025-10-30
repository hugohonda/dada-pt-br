import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .config.datasets import DEFAULT_MODELS, PHASE_WORKERS, TRANSLATION_MODELS
from .config.logging import log_model_info, setup_logger
from .llm_client import init_ollama
from .metadata_utils import create_review_metadata
from .utils import (
    get_dataset_id,
    get_timestamp,
    load_json_file,
    save_json_file,
    validate_file_exists,
)

_LOGGER = setup_logger("reviewer", log_to_file=True, log_prefix="review")


def load_review_prompt() -> str:
    """Load review prompt."""
    prompt_path = Path(__file__).parent / "prompts" / "review.md"
    return prompt_path.read_text(encoding="utf-8").strip()


def review_translation(client, model_name: str, prompt: str) -> str:
    """Review translation using specified model with custom prompt."""
    try:
        # Check if this model should disable thinking
        from .config.datasets import TRANSLATION_MODELS

        think_param = None
        for _, config in TRANSLATION_MODELS.items():
            if config["ollama_name"] == model_name:
                if "think" in config:
                    think_param = config["think"]
                break

        # Build chat call parameters
        chat_kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.1, "top_p": 0.9, "max_tokens": 2048},
        }

        # Add think parameter if specified (for thinking models like qwen3)
        if think_param is not None:
            chat_kwargs["think"] = think_param

        response = client.chat(**chat_kwargs)
        return response["message"]["content"].strip()
    except Exception as e:
        _LOGGER.error(f"Error reviewing translation: {e}")
        raise


def format_review_prompt(
    source: str,
    translation: str,
    score: float,
    alternative_translation: str,
    alternative_score: float,
    error_spans: list[dict],
) -> str:
    """Format the review prompt with merged data including both translations."""
    # Load the review prompt template
    prompt_template = load_review_prompt()

    # Format error spans as JSON
    # error_spans_json = json.dumps(error_spans, indent=2, ensure_ascii=False)

    # Format the prompt template with the merged data
    formatted_prompt = prompt_template.format(
        source=source,
        translation=translation,
        score=score,
        alternative_translation=alternative_translation,
        alternative_score=alternative_score,
    )

    return formatted_prompt


def create_review_entry(example: dict, review_result: dict, index: int) -> dict:
    """Create a clean review entry from example and review result."""
    reviewed_translation = review_result.get("improved_translation", "")

    return {
        "index": example.get("index", index),
        "id": example.get("id", index),
        "source": example.get("source", ""),
        "translation": example.get("translation", ""),
        "score": example.get("score", 0.0),
        "alternative_translation": example.get("second_translation", ""),
        "alternative_score": example.get("second_score", 0.0),
        "merged_from": example.get("merged_from", "unknown"),
        "reviewed_translation": reviewed_translation,
    }


def review_single_example(
    example: dict, client, model_name: str, prompt_template: str
) -> dict:
    """Review a single example from merged data and propose a better translation."""
    try:
        source = example.get("source", "")
        selected_translation = example.get("translation", "")
        selected_score = example.get("score", 0.0)
        alternative_translation = example.get("second_translation", "")
        alternative_score = example.get("second_score", 0.0)
        error_spans = example.get("error_spans", [])
        merged_from = example.get("merged_from", "unknown")

        # Filter out low confidence error spans (confidence < 0.5)
        filtered_error_spans = []
        for span in error_spans:
            confidence = span.get("confidence", 0.0)
            if confidence >= 0.5:
                filtered_error_spans.append(span)

        error_spans = filtered_error_spans

        # Smart review decision: always review if there's an alternative translation
        # or if there are significant errors, even with high scores
        should_skip = (
            selected_score > 0.99
            and not error_spans
            and not alternative_translation.strip()
        )

        if should_skip:
            return {
                "improved_translation": "",  # Empty means not reviewed
            }

        # Format the review prompt with both translations
        review_prompt = format_review_prompt(
            source=source,
            translation=selected_translation,
            score=selected_score,
            alternative_translation=alternative_translation,
            alternative_score=alternative_score,
            error_spans=error_spans,
        )

        # Get the improved translation using a custom review function
        improved_translation = review_translation(
            client=client, model_name=model_name, prompt=review_prompt
        )

        # Extract the improved translation from the response
        # The model should return just the improved translation
        improved_translation = improved_translation.strip()

        # Remove quotes if present
        if improved_translation.startswith('"') and improved_translation.endswith('"'):
            improved_translation = improved_translation[1:-1]

        # If the model returned the English text instead of Portuguese, skip this example
        if improved_translation == source:
            return {
                "reviewed": False,
                "reason": "Model returned English text instead of Portuguese translation",
                "original_score": selected_score,
            }

        return {
            "reviewed": True,
            "original_translation": selected_translation,
            "improved_translation": improved_translation,
            "original_score": selected_score,
            "alternative_translation": alternative_translation,
            "alternative_score": alternative_score,
            "merged_from": merged_from,
            "error_spans": error_spans,
        }

    except Exception as e:
        _LOGGER.error(f"Error reviewing example {example.get('id', 'unknown')}: {e}")
        return {
            "reviewed": False,
            "error": str(e),
            "original_score": example.get("score", 0.0),
        }


def review_single_parallel(args):
    """Review a single example - used for parallel processing."""
    example, client, model_name, prompt_template, index = args
    start_time = time.time()

    try:
        review_result = review_single_example(
            example, client, model_name, prompt_template
        )
        processing_time = time.time() - start_time

        return {
            "success": True,
            "result": review_result,
            "processing_time": processing_time,
            "index": index,
            "error": None,
        }
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "success": False,
            "result": {"improved_translation": "", "error": str(e)},
            "processing_time": processing_time,
            "index": index,
            "error": str(e),
        }


def process_dataset(
    input_file: str,
    output_file: str,
    model_name: str = None,
    limit: int | None = None,
    max_workers: int = None,
):
    """Process merged evaluation results and generate improved translations."""
    start_time = time.time()
    _LOGGER.info(f"Reviewing merged translations from: {input_file}")

    # Load merged evaluation results
    data, metadata = load_json_file(input_file)
    if not data:
        _LOGGER.error("No data found in input file")
        return

    # Apply limit if specified
    if limit:
        data = data[:limit]
        _LOGGER.info(f"Limited to {limit} examples")

    # Get dataset ID
    dataset_id = get_dataset_id(input_file)

    _LOGGER.info(f"Reviewing {len(data)} examples from dataset: {dataset_id}")

    # Get default model if not specified
    if model_name is None:
        model_name = DEFAULT_MODELS["review"]

    # Set default workers if not provided
    if max_workers is None:
        max_workers = PHASE_WORKERS["review"]["default"]
    max_workers = min(max_workers, PHASE_WORKERS["review"]["max"])

    model_config = TRANSLATION_MODELS.get(model_name, {})
    ollama_model_name = model_config.get("ollama_name", model_name)
    log_model_info(
        _LOGGER,
        "review",
        model_name,
        model_config,
        max_workers=max_workers,
    )

    client = init_ollama(ollama_model_name, max_workers=max_workers)

    # Load review prompt
    prompt_template = load_review_prompt()

    _LOGGER.info(f"Processing {len(data)} examples with {max_workers} workers...")

    # Prepare task arguments for parallel processing
    task_args = [
        (example, client, ollama_model_name, prompt_template, i)
        for i, example in enumerate(data)
    ]

    # Process examples in parallel
    reviewed_data = [None] * len(data)
    review_stats = {"total": len(data), "reviewed": 0, "skipped": 0, "errors": 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(review_single_parallel, args): args[4] for args in task_args
        }

        for future in tqdm(
            as_completed(future_to_index),
            desc=f"Reviewing with {model_name.split('/')[-1]}",
            total=len(task_args),
        ):
            result = future.result()
            index = result["index"]

            review_entry = create_review_entry(data[index], result["result"], index)

            if result["result"].get("error"):
                review_entry["error"] = result["result"].get("error")
                review_stats["errors"] += 1
            elif bool(review_entry["reviewed_translation"].strip()):
                review_stats["reviewed"] += 1
            else:
                review_stats["skipped"] += 1

            reviewed_data[index] = review_entry

    # Save reviewed data with metadata
    _LOGGER.info("Saving review results...")

    # Calculate total processing time
    total_processing_time = time.time() - start_time

    # Create review metadata
    review_metadata = create_review_metadata(
        pipeline_id=get_timestamp(),
        dataset_id=get_dataset_id(input_file),
        total_examples=len(reviewed_data),
        processing_time=total_processing_time,
        model_name=model_name,
        reviewed_data=reviewed_data,
        review_stats=review_stats,
    )

    # Create final data structure
    final_data = {"metadata": review_metadata, "data": reviewed_data}

    save_json_file(final_data, output_file)

    # Generate summary
    _LOGGER.info("Review completed:")
    _LOGGER.info(f"  Total examples: {review_stats['total']}")
    _LOGGER.info(f"  Reviewed: {review_stats['reviewed']}")
    _LOGGER.info(f"  Skipped: {review_stats['skipped']}")
    _LOGGER.info(f"  Errors: {review_stats['errors']}")
    _LOGGER.info(f"  Review results saved to: {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Translation Reviewer")
    parser.add_argument("input_file", help="Input evaluation JSON file")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODELS["review"],
        help="Model to use for review",
    )
    parser.add_argument("--limit", "-l", type=int, help="Limit examples (default: all)")

    args = parser.parse_args()

    if not validate_file_exists(args.input_file):
        _LOGGER.error(f"File not found: {args.input_file}")
        return

    # Generate output filename if not provided
    if not args.output:
        input_path = Path(args.input_file)
        output_file = (
            input_path.parent / f"{input_path.stem}_reviewed{input_path.suffix}"
        )
    else:
        output_file = args.output

    process_dataset(args.input_file, str(output_file), args.model, args.limit)


if __name__ == "__main__":
    main()
