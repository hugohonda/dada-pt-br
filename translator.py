#!/usr/bin/env python3
"""
Dataset translator using Ollama with Gemma3.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import ollama
from dotenv import load_dotenv
from tqdm import tqdm

from config.logging import setup_logger
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
    load_json_file,
    load_jsonl_file,
    save_json_file,
    save_jsonl_line,
    validate_file_exists,
)

_LOGGER = setup_logger("translator", log_to_file=True, log_prefix="translation")


def load_prompt(prompt_type: str) -> str:
    """Load translation prompt based on type."""
    prompt_files = {
        "single": "prompts/translation_single.md",
        "multilingual": "prompts/translation_multilingual.md",
        "safety": "prompts/translation_safety.md",
    }

    prompt_file = prompt_files.get(prompt_type, "prompts/translation_single.md")

    try:
        with open(prompt_file, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        _LOGGER.error(f"Translation prompt file not found: {prompt_file}")
        # Fallback prompt
        return """[INST]Translate the following text from English to Brazilian Portuguese. Don't translate slang literally and use the best colloquial adaptation. Complete only with the translation, don't create notes, justification, etc.[/INST]
[Source: English]
\"\"\"
{text}
\"\"\"
[Translation: Portuguese from Brazil]
\"\"\"
"""


def init_ollama():
    """Initialize Ollama client."""
    try:
        client = ollama.Client()
        _LOGGER.info("Connected to Ollama")

        # Check if gemma3:latest is available
        models = client.list()
        if hasattr(models, "models") and models.models:
            model_names = [model.model for model in models.models]
            if "gemma3:latest" not in model_names:
                raise Exception("gemma3:latest model not found")
            _LOGGER.info("gemma3:latest model found")
        else:
            raise Exception("No models found")

        return client
    except Exception as e:
        _LOGGER.error(f"Error initializing Ollama: {e}")
        raise


def translate_text(text: str, client, prompt_template: str, **kwargs) -> str:
    """Translate text using Gemma3."""
    prompt = prompt_template.format(text=text, **kwargs)

    try:
        response = client.chat(
            model="gemma3:latest",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "top_p": 0.9, "max_tokens": 2048},
        )

        translation = response["message"]["content"].strip()

        # Clean up the response
        if '"""' in translation:
            parts = translation.split('"""')
            if len(parts) >= 2:
                translation = parts[-2].strip()

        return translation
    except Exception as e:
        _LOGGER.error(f"Error translating text: {e}")
        return text


def translate_example(
    example: dict[str, Any], client, prompt_template: str, dataset_type: str
) -> dict[str, Any]:
    """Translate a single example based on dataset type."""
    translated = {}

    for field_name, field_value in example.items():
        if not field_value or not isinstance(field_value, str):
            translated[field_name] = field_value
            continue

        # Skip translation for certain fields
        if field_name in ["id", "id_original", "detailed_prompt", "hint_included"]:
            translated[field_name] = field_value
            continue

        # Keep original categories - no translation needed
        if field_name == "category":
            translated[field_name] = field_value
            continue

        # Handle different dataset types
        if dataset_type == "multilingual" and field_name == "en":
            # For multilingual datasets, translate the English version using context
            context = {
                "fr": example.get("fr", ""),
                "de": example.get("de", ""),
                "es": example.get("es", ""),
                "it": example.get("it", ""),
                "en": field_value,
            }
            translated["pt"] = translate_text(
                field_value, client, prompt_template, **context
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
            translated[field_name] = translate_text(
                field_value, client, prompt_template
            )
        else:
            translated[field_name] = field_value

    return translated


def translate_single(args):
    """Translate a single example - used for parallel processing."""
    example, client, prompt_template, dataset_type, index = args
    start_time = time.time()

    try:
        translated_example = translate_example(
            example, client, prompt_template, dataset_type
        )
        processing_time = time.time() - start_time
        return {
            "success": True,
            "data": translated_example,
            "processing_time": processing_time,
            "index": index,
            "error": None,
        }
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "success": False,
            "data": example,  # Return original on failure
            "processing_time": processing_time,
            "index": index,
            "error": str(e),
        }


def process_dataset(input_file: str, output_file: str, max_workers: int = 4):
    """Process and translate a dataset file."""
    # Start translation tracking
    start_translation()

    try:
        # Initialize Ollama
        client = init_ollama()

        # Load dataset
        try:
            data = load_json_file(input_file)
        except Exception as e:
            _LOGGER.error(f"Error loading dataset: {e}")
            return

        if not data:
            _LOGGER.error("Empty dataset")
            return

        # Detect dataset type from first example
        dataset_type = detect_dataset_type(data[0])
        _LOGGER.info(f"Detected dataset type: {dataset_type}")

        # Load appropriate prompt
        prompt_template = load_prompt(dataset_type)

        # Setup output files
        ensure_directory_exists(os.path.dirname(output_file))
        jsonl_file = output_file.replace(".json", ".jsonl")

        # Check for existing progress
        processed_count = 0
        if os.path.exists(jsonl_file):
            with open(jsonl_file, encoding="utf-8") as f:
                processed_count = sum(1 for line in f if line.strip())
            _LOGGER.info(
                f"Found {processed_count} already processed examples, resuming..."
            )

        _LOGGER.info(
            f"Processing {len(data)} examples (starting from {processed_count}) with {max_workers} workers..."
        )

        # Prepare data for parallel processing
        remaining_data = data[processed_count:]
        task_args = [
            (example, client, prompt_template, dataset_type, i + processed_count)
            for i, example in enumerate(remaining_data)
        ]

        # Process data with parallel processing and line-by-line saving
        with open(jsonl_file, "a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(translate_single, args): args[4]
                    for args in task_args
                }

                # Process completed tasks as they finish
                for future in tqdm(
                    as_completed(future_to_index),
                    desc="Translating",
                    total=len(task_args),
                    initial=processed_count,
                ):
                    result = future.result()

                    # Save immediately to JSONL
                    save_jsonl_line(result["data"], jsonl_file)

                    # Track translation result
                    add_translation_result(
                        result["success"], result["processing_time"], result["error"]
                    )

                    if not result["success"]:
                        _LOGGER.error(
                            f"Translation failed for example {result['index']}: {result['error']}"
                        )

        # Convert JSONL to final JSON format
        _LOGGER.info("Converting JSONL to final JSON format...")
        translated_data = load_jsonl_file(jsonl_file)
        save_json_file(translated_data, output_file)

        _LOGGER.info(f"Saved translated data to {output_file}")
        _LOGGER.info(f"Line-by-line backup saved to {jsonl_file}")

    finally:
        # End translation tracking and generate report
        end_translation()
        report_file = generate_translation_report(input_file, output_file, dataset_type)
        if report_file:
            _LOGGER.info(f"Translation report generated: {report_file}")


def main():
    """Main CLI entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Dataset Translator - Translate datasets using Ollama Gemma3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python translator.py dataset.json
  python translator.py dataset.json --workers=8
  python translator.py dataset.json --output=output.json
  python translator.py dataset.json --output=output.json --workers=8
        """,
    )

    parser.add_argument("input_file", help="Input JSON file to translate")
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file (default: auto-generated from input file)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, recommended: 2-8)",
    )

    args = parser.parse_args()

    # Validate input file
    if not validate_file_exists(args.input_file):
        _LOGGER.error(f"File not found: {args.input_file}")
        return

    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        output_file = generate_output_filename(args.input_file)

    # Validate workers count
    if args.workers < 1 or args.workers > 16:
        _LOGGER.error(f"Workers must be between 1 and 16, got: {args.workers}")
        return

    _LOGGER.info(f"Input file: {args.input_file}")
    _LOGGER.info(f"Output file: {output_file}")
    _LOGGER.info(f"Workers: {args.workers}")

    process_dataset(args.input_file, output_file, args.workers)


if __name__ == "__main__":
    main()
