import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .config.datasets import PHASE_WORKERS
from .config.logging import setup_logger
from .llm_client import get_ollama_name, init_ollama
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
    """Load cultural review prompt."""
    prompt_path = Path(__file__).parent / "prompts" / "cultural_review.md"
    return prompt_path.read_text(encoding="utf-8").strip()


def review_translation(client, model_name: str, prompt: str) -> str:
    """Review translation using specified model with custom prompt."""
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "top_p": 0.9, "max_tokens": 2048},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        _LOGGER.error(f"Error reviewing translation: {e}")
        raise


def format_review_prompt(
    source: str, translation: str, needs_review: list[dict]
) -> str:
    """Format cultural review prompt."""
    entities_list = (
        "\n".join(
            [
                f"- {ent['text']} ({ent['type']}): {ent['reason']}"
                for ent in needs_review
            ]
        )
        if needs_review
        else "(None)"
    )

    return load_review_prompt().format(
        source=source,
        translation=translation,
        entities_list=entities_list,
    )


def create_review_entry(example: dict, review_result: dict, index: int) -> dict:
    """Create review entry from example and result."""
    return {
        "index": example.get("index", index),
        "id": example.get("id", index),
        "source": example.get("source", ""),
        "translation": example.get("translation", ""),
        "reviewed_translation": review_result.get("improved_translation", ""),
        "entities_adapted": review_result.get("entities_adapted", []),
    }


def review_single_example(
    example: dict, client, model_name: str, prompt_template: str
) -> dict:
    """Review example for cultural adaptation."""
    try:
        analysis = example.get("analysis", {})
        needs_review = analysis.get("ner_accuracy", {}).get("needs_review", [])

        if not needs_review:
            return {"improved_translation": ""}

        prompt = format_review_prompt(
            example.get("source", ""), example.get("translation", ""), needs_review
        )

        result = review_translation(client, model_name, prompt).strip().strip('"')

        # Remove any explanations/markdown that might have been added
        if "\n" in result:
            result = result.split("\n")[0].strip()
        if "**" in result or "*" in result:
            # Take first line before any markdown
            result = result.split("**")[0].split("*")[0].strip()

        if result == example.get("source", ""):
            return {"reviewed": False, "reason": "Returned English"}

        return {
            "reviewed": True,
            "original_translation": example.get("translation", ""),
            "improved_translation": result,
            "entities_adapted": [e["text"] for e in needs_review],
        }

    except Exception as e:
        _LOGGER.error(f"Error reviewing {example.get('id', 'unknown')}: {e}")
        return {"reviewed": False, "error": str(e)}


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
    model_name: str = "tower",
    limit: int | None = None,
    max_workers: int = None,
):
    """Process analyzer results and generate culturally adapted translations."""
    start_time = time.time()
    _LOGGER.info(f"Reviewing analyzed translations from: {input_file}")

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

    # Initialize Ollama client
    ollama_model_name = get_ollama_name(model_name)
    client = init_ollama(ollama_model_name)

    # Load review prompt
    prompt_template = load_review_prompt()

    # Set default workers if not provided
    if max_workers is None:
        max_workers = PHASE_WORKERS["review"]["default"]
    max_workers = min(max_workers, PHASE_WORKERS["review"]["max"])

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
        "--model", "-m", default="tower", help="Model to use for review"
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
