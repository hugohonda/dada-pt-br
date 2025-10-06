"""
Common utility functions used across all scripts in the dadaptbr project.
"""

import json
import os
from datetime import datetime
from typing import Any


def get_timestamp() -> str:
    """Get standardized timestamp for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)


def load_json_file(file_path: str) -> list[dict[str, Any]]:
    """Load JSON file and return data."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Any, file_path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    ensure_directory_exists(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def save_jsonl_line(data: dict[str, Any], file_path: str) -> None:
    """Append a single line to JSONL file."""
    ensure_directory_exists(os.path.dirname(file_path))
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl_file(file_path: str) -> list[dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    if not os.path.exists(file_path):
        return []

    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def detect_dataset_type(example: dict[str, Any]) -> str:
    """Detect dataset type based on example structure."""
    keys = set(example.keys())

    # M-ALERT multilingual dataset
    if "fr" in keys and "de" in keys and "es" in keys and "it" in keys and "en" in keys:
        return "multilingual"

    # AgentHarm safety dataset
    if "category" in keys and "prompt" in keys and "name" in keys:
        return "safety"

    # ALERT dataset
    if "category" in keys and "prompt" in keys:
        return "safety"

    # Default to single language
    return "single"


def extract_texts(example: dict[str, Any], dataset_type: str) -> tuple[str, str]:
    """Extract source and translation texts based on dataset type."""
    if dataset_type == "multilingual":
        return example.get("en", ""), example.get("pt", "")
    elif dataset_type == "safety":
        return example.get("prompt", ""), example.get("prompt", "")
    else:
        return example.get("text", example.get("content", "")), example.get(
            "translation", ""
        )


def generate_output_filename(input_file: str, output_type: str = "translated") -> str:
    """Generate output filename with timestamp pattern."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = get_timestamp()
    output_dir = (
        "datasets/processed" if output_type == "translated" else "datasets/evaluation"
    )
    ensure_directory_exists(output_dir)
    return os.path.join(output_dir, f"{base_name}_{output_type}_{timestamp}.json")


def generate_evaluation_filename(input_file: str) -> str:
    """Generate evaluation output filename."""
    return generate_output_filename(input_file, "evaluated")


def validate_file_exists(file_path: str) -> bool:
    """Validate that file exists."""
    return os.path.exists(file_path)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
