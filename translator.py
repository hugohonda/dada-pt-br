#!/usr/bin/env python3
"""
Dataset translator.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from config.logging import setup_logger
from llm_client import get_ollama_name, init_ollama, translate_text
from report_generator import (
    add_translation_result,
    end_translation,
    generate_translation_report,
    start_translation,
)
from utils import (
    detect_dataset_type,
    ensure_directory_exists,
    generate_output_filename,
    get_dataset_id,
    load_json_file,
    load_jsonl_file,
    save_json_file,
    save_jsonl_line,
    validate_file_exists,
)

_LOGGER = setup_logger("translator", log_to_file=True, log_prefix="translation")


def load_prompt(prompt_type: str) -> str:
    """Load translation prompt based on type."""
    prompt_file = f"prompts/translation_{prompt_type}.md"
    with open(prompt_file, encoding="utf-8") as f:
        return f.read().strip()


def translate_single(args):
    """Translate a single example - used for parallel processing."""
    example, client, model_name, prompt_template, dataset_type, index = args
    start_time = time.time()

    try:
        translated = {}
        for field_name, field_value in example.items():
            if not field_value or not isinstance(field_value, str):
                translated[field_name] = field_value
                continue

            # Skip translation for certain fields
            if field_name in [
                "id",
                "id_original",
                "detailed_prompt",
                "hint_included",
                "category",
            ]:
                translated[field_name] = field_value
                continue

            # Handle different dataset types
            if dataset_type == "multilingual" and field_name == "en":
                context = {
                    "fr": example.get("fr", ""),
                    "de": example.get("de", ""),
                    "es": example.get("es", ""),
                    "it": example.get("it", ""),
                    "en": field_value,
                }
                prompt = prompt_template.format(text=field_value, **context)
                translated["pt"] = translate_text(
                    field_value, client, model_name, prompt
                )
                translated[field_name] = field_value  # Keep original English
            elif field_name in [
                "name",
                "prompt",
                "detailed_prompt",
                "text",
                "content",
                "en",
            ]:
                prompt = prompt_template.format(text=field_value)
                translated[field_name] = translate_text(
                    field_value, client, model_name, prompt
                )
            else:
                translated[field_name] = field_value

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
    max_workers: int = 4,
    limit: int = None,
):
    """Process and translate a dataset file using specified model."""
    start_translation()

    try:
        client = init_ollama(model_name)
        data = load_json_file(input_file)
        if not data:
            _LOGGER.error("Empty dataset")
            return

        if limit:
            data = data[:limit]
            _LOGGER.info(f"Limited to {limit} examples")

        dataset_type = detect_dataset_type(data[0])
        dataset_id = get_dataset_id(input_file)
        _LOGGER.info(f"Dataset type: {dataset_type}, Dataset ID: {dataset_id}")

        prompt_template = load_prompt(dataset_type)
        ensure_directory_exists(os.path.dirname(output_file))
        jsonl_file = output_file.replace(".json", ".jsonl")

        # Check for existing progress
        processed_count = 0
        if os.path.exists(jsonl_file):
            with open(jsonl_file, encoding="utf-8") as f:
                processed_count = sum(1 for line in f if line.strip())
            _LOGGER.info(f"Resuming from {processed_count} examples")

        remaining_data = data[processed_count:]
        _LOGGER.info(
            f"Processing {len(remaining_data)} examples with {max_workers} workers..."
        )

        task_args = [
            (
                example,
                client,
                model_name,
                prompt_template,
                dataset_type,
                i + processed_count,
            )
            for i, example in enumerate(remaining_data)
        ]

        with open(jsonl_file, "a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(translate_single, args): args[5]
                    for args in task_args
                }

                for future in tqdm(
                    as_completed(future_to_index),
                    desc=f"Translating with {model_name.split('/')[-1]}",
                    total=len(task_args),
                    initial=processed_count,
                ):
                    result = future.result()
                    save_jsonl_line(result["data"], jsonl_file)
                    add_translation_result(
                        result["success"], result["processing_time"], result["error"]
                    )

                    if not result["success"]:
                        _LOGGER.error(
                            f"Translation failed for example {result['index']}: {result['error']}"
                        )

        # Convert to final JSON
        _LOGGER.info("Converting to final JSON format...")
        translated_data = load_jsonl_file(jsonl_file)
        save_json_file(translated_data, output_file)

        _LOGGER.info(f"Saved: {output_file}")

    finally:
        end_translation()
        report_file = generate_translation_report(
            input_file, output_file, dataset_type, model_name
        )
        if report_file:
            _LOGGER.info(f"Report: {report_file}")


def main():
    """Main CLI entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Dataset Translator - Ollama Gemma3 & TowerInstruct-Mistral-7B-v0.2",
        epilog="Examples:\n  python translator.py dataset.json\n  python translator.py dataset.json --tower --limit=100",
    )

    parser.add_argument("input_file", help="Input JSON file to translate")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument(
        "--workers", "-w", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument("--limit", "-l", type=int, help="Limit examples")
    parser.add_argument(
        "--tower",
        action="store_true",
        help="Use TowerInstruct-Mistral-7B-v0.2 instead of Gemma3",
    )

    args = parser.parse_args()

    if not validate_file_exists(args.input_file):
        _LOGGER.error(f"File not found: {args.input_file}")
        return

    output_file = args.output or generate_output_filename(args.input_file)
    model_name = get_ollama_name("towerinstruct" if args.tower else "gemma3")

    _LOGGER.info(f"Processing: {args.input_file} -> {output_file}")
    _LOGGER.info(
        f"Workers: {args.workers}, Limit: {args.limit or 'all'}, Model: {model_name}"
    )

    process_dataset(args.input_file, output_file, model_name, args.workers, args.limit)


if __name__ == "__main__":
    main()
