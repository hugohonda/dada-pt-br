import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from .config.datasets import FILE_PROCESSING, PHASE_WORKERS, TRANSLATION_MODELS
from .config.logging import setup_logger
from .llm_client import get_ollama_name, init_ollama, translate_text
from .metadata_utils import create_translation_metadata
from .utils import (
    ensure_directory_exists,
    generate_output_filename,
    get_dataset_id,
    get_timestamp,
    load_json_file,
    save_json_file,
    validate_file_exists,
)

_LOGGER = setup_logger("translator", log_to_file=True, log_prefix="translation")


def load_prompt() -> str:
    """Load translation prompt."""
    from pathlib import Path

    prompt_file = Path(__file__).parent / "prompts" / "translation.md"
    return prompt_file.read_text(encoding="utf-8").strip()


def standardize_category_name(category: str) -> str:
    """Standardize category names to lowercase with underscores."""
    if not category:
        return category

    # Convert to lowercase and replace spaces/special chars with underscores
    return category.lower().replace(" ", "_").replace("-", "_").replace("+", "_plus")


def preprocess_text(text: str) -> str:
    """Standardized text preprocessing for all datasets."""
    if not text:
        return text

    # Remove instruction markers (common across datasets)
    text = re.sub(
        r"### (Instruction|Instrução|Response|Resposta|Sugestão):\s*\n?", "", text
    )
    text = re.sub(r"###\s*", "", text)

    # Clean up whitespace
    return text.strip()


def ensure_standardized_output(translated: dict, original: dict) -> dict:
    """Ensure the output follows the standardized format with only essential fields."""
    # Find English text from original
    en_text = ""
    for field in ["prompt", "text", "content", "en"]:
        if field in original and original[field]:
            en_text = preprocess_text(original[field])
            break

    # Find Portuguese translation
    pt_text = ""
    for field in ["pt-br", "pt", "translation"]:
        if field in translated and translated[field]:
            pt_text = preprocess_text(translated[field])
            break

    # Create clean output with only essential fields
    clean_output = {"en": en_text, "pt-br": pt_text}

    # Add category and id from original
    if "category" in original:
        clean_output["category"] = standardize_category_name(original["category"])
    if "id" in original:
        clean_output["id"] = original["id"]

    return clean_output


def translate_single(args):
    """Translate a single example - used for parallel processing."""
    example, client, model_name, prompt_template, index = args
    start_time = time.time()

    try:
        # Find English text to translate
        en_text = ""
        for field in ["en", "prompt", "text", "content"]:
            if field in example and example[field] and isinstance(example[field], str):
                en_text = preprocess_text(example[field])
                break

        if not en_text:
            raise ValueError("No English text found to translate")

        # Translate to Portuguese
        pt_text = translate_text(en_text, client, model_name, prompt_template)

        # Create standardized output
        translated = {"en": en_text, "pt-br": pt_text}

        # Add category and id from original
        if "category" in example:
            translated["category"] = standardize_category_name(example["category"])
        if "id" in example:
            translated["id"] = example["id"]

        processing_time = time.time() - start_time
        return {
            "success": True,
            "data": translated,
            "processing_time": processing_time,
            "index": index,
            "error": None,
        }
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "success": False,
            "data": example,
            "processing_time": processing_time,
            "index": index,
            "error": str(e),
        }


def process_dataset(
    input_file: str,
    output_file: str,
    model_name: str,
    max_workers: int = PHASE_WORKERS["translation"]["default"],
    limit: int = None,
    device: str = FILE_PROCESSING["default_device"],
    pipeline_id: str = None,
):
    """Process and translate a dataset file using specified model."""
    # Generate pipeline ID if not provided
    if pipeline_id is None:
        pipeline_id = get_timestamp()

    # Track total processing time
    total_start_time = time.time()
    _LOGGER.info(f"Translation process started - Pipeline ID: {pipeline_id}")

    try:
        # Get Ollama model name from config
        ollama_model_name = get_ollama_name(model_name)
        client = init_ollama(ollama_model_name)

        data, _ = load_json_file(input_file)
        if not data:
            _LOGGER.error("Empty dataset")
            return

        if limit:
            data = data[:limit]
            _LOGGER.info(f"Limited to {limit} examples")

        dataset_id = get_dataset_id(input_file)
        _LOGGER.info(f"Dataset ID: {dataset_id}")

        # Use single standardized prompt template
        prompt_template = load_prompt()
        ensure_directory_exists(os.path.dirname(output_file))

        # Check for existing progress
        processed_count = 0
        if os.path.exists(output_file):
            existing_data, _ = load_json_file(output_file)
            processed_count = len(existing_data)
            _LOGGER.info(f"Resuming from {processed_count} examples")

        remaining_data = data[processed_count:]
        _LOGGER.info(
            f"Processing {len(remaining_data)} examples with {max_workers} workers..."
        )

        task_args = [
            (example, client, ollama_model_name, prompt_template, i + processed_count)
            for i, example in enumerate(remaining_data)
        ]

        # Collect results in order
        results = [None] * len(task_args)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(translate_single, args): args[4] for args in task_args
            }

            for future in tqdm(
                as_completed(future_to_index),
                desc=f"Translating with {model_name.split('/')[-1]}",
                total=len(task_args),
                initial=processed_count,
            ):
                result = future.result()
                index = result["index"] - processed_count  # Adjust for resume
                results[index] = result

                # Log translation result
                if result["success"]:
                    _LOGGER.debug(
                        f"Translation completed in {result['processing_time']:.2f}s"
                    )
                else:
                    _LOGGER.error(
                        f"Translation failed for example {result['index']}: {result['error']}"
                    )

        # Write results directly to JSON in order
        _LOGGER.info("Saving results in order...")
        translated_data = []

        # Load existing data if resuming
        if processed_count > 0 and os.path.exists(output_file):
            translated_data, _ = load_json_file(output_file)

        # Add new results in order
        for result in results:
            if result and result["success"]:
                translated_data.append(result["data"])

        # Calculate total processing time
        total_processing_time = time.time() - total_start_time

        # Add comprehensive metadata to the data
        metadata = create_translation_metadata(
            pipeline_id=pipeline_id,
            dataset_id=dataset_id,
            total_examples=len(translated_data),
            processing_time=total_processing_time,
            model_name=model_name,
            max_workers=max_workers,
            translated_data=translated_data,
        )

        # Create final data structure with metadata
        final_data = {"metadata": metadata, "data": translated_data}

        save_json_file(final_data, output_file)

        _LOGGER.info(f"Saved: {output_file}")

    finally:
        _LOGGER.info("Translation process completed")


def main():
    """Main CLI entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Dataset Translator - Ollama Gemma3 & TowerInstruct-Mistral-7B-v0.2",
        epilog="Examples:\n  python translator.py dataset.json\n  python translator.py dataset.json --model=tower --limit=100\n  python translator.py dataset.json --model=gemma3",
    )

    parser.add_argument("input_file", help="Input JSON file to translate")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=PHASE_WORKERS["translation"]["default"],
        help="Number of parallel workers",
    )
    parser.add_argument("--limit", "-l", type=int, help="Limit examples")
    parser.add_argument(
        "--model",
        "-m",
        default="tower",
        help="Model to use for translation (default: tower). Options: tower, gemma3, or custom model name",
    )

    args = parser.parse_args()

    if not validate_file_exists(args.input_file):
        _LOGGER.error(f"File not found: {args.input_file}")
        return

    output_file = args.output or generate_output_filename(args.input_file)

    if args.model in TRANSLATION_MODELS:
        model_name = TRANSLATION_MODELS[args.model]["ollama_name"]
    else:
        model_name = args.model

    _LOGGER.info(f"Processing: {args.input_file} -> {output_file}")
    _LOGGER.info(
        f"Workers: {args.workers}, Limit: {args.limit or 'all'}, Model: {model_name}"
    )

    process_dataset(args.input_file, output_file, model_name, args.workers, args.limit)


if __name__ == "__main__":
    main()
