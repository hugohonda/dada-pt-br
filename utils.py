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
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}") from e


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


def generate_output_filename(
    input_file: str,
    output_type: str = "translated",
    output_dir: str = "datasets/processed",
) -> str:
    """Generate output filename with timestamp pattern."""
    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = get_timestamp()

    # Create output directory
    ensure_directory_exists(output_dir)

    return os.path.join(output_dir, f"{base_name}_{output_type}_{timestamp}.json")


def generate_evaluation_filename(input_file: str) -> str:
    """Generate evaluation output filename."""
    return generate_output_filename(input_file, "evaluated", "datasets/evaluation")


def generate_report_filename(input_file: str, report_type: str = "translation") -> str:
    """Generate report filename."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = get_timestamp()

    ensure_directory_exists("reports")
    return f"reports/{base_name}_{report_type}_report_{timestamp}.json"


def validate_file_exists(file_path: str) -> bool:
    """Validate that file exists."""
    return os.path.exists(file_path)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    if not os.path.exists(file_path):
        return 0.0
    return os.path.getsize(file_path) / (1024 * 1024)


def count_json_lines(file_path: str) -> int:
    """Count lines in JSONL file."""
    if not os.path.exists(file_path):
        return 0

    count = 0
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


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


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and newlines."""
    if not isinstance(text, str):
        return str(text)
    return " ".join(text.split())


def is_valid_text(text: str) -> bool:
    """Check if text is valid for processing."""
    return isinstance(text, str) and len(text.strip()) > 0


def get_example_id(example: dict[str, Any], index: int) -> Any:
    """Get example ID, using 'id' field or index as fallback."""
    return example.get("id", index)


def create_progress_info(current: int, total: int, prefix: str = "Progress") -> str:
    """Create progress information string."""
    percentage = (current / total * 100) if total > 0 else 0
    return f"{prefix}: {current}/{total} ({percentage:.1f}%)"


def validate_dataset(data: list[dict[str, Any]]) -> tuple[bool, str]:
    """Validate dataset structure."""
    if not data:
        return False, "Empty dataset"

    if not isinstance(data, list):
        return False, "Dataset must be a list of examples"

    if not isinstance(data[0], dict):
        return False, "Each example must be a dictionary"

    return True, "Valid dataset"


def filter_valid_examples(
    data: list[dict[str, Any]], dataset_type: str
) -> tuple[list[dict[str, Any]], list[int]]:
    """Filter examples that have valid source and translation texts."""
    valid_examples = []
    valid_indices = []

    for i, example in enumerate(data):
        source, translation = extract_texts(example, dataset_type)
        if is_valid_text(source) and is_valid_text(translation):
            valid_examples.append(example)
            valid_indices.append(i)

    return valid_examples, valid_indices


def create_batch_indices(total_items: int, batch_size: int) -> list[tuple[int, int]]:
    """Create batch start/end indices for processing."""
    batches = []
    for i in range(0, total_items, batch_size):
        end = min(i + batch_size, total_items)
        batches.append((i, end))
    return batches


def calculate_statistics(values: list[float]) -> dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def create_summary_dict(
    input_file: str,
    output_file: str,
    dataset_type: str,
    total_examples: int,
    processed_examples: int,
    success_rate: float,
    processing_time: float,
) -> dict[str, Any]:
    """Create a standardized summary dictionary."""
    return {
        "input_file": input_file,
        "output_file": output_file,
        "dataset_type": dataset_type,
        "total_examples": total_examples,
        "processed_examples": processed_examples,
        "success_rate": success_rate,
        "processing_time_seconds": processing_time,
        "processing_time_formatted": format_duration(processing_time),
        "timestamp": datetime.now().isoformat(),
    }
