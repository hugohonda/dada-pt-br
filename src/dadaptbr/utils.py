import json
import os
from datetime import datetime
from pathlib import Path
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


def extract_texts(example: dict[str, Any]) -> tuple[str, str]:
    """Extract source and translation texts using simplified logic."""
    # Standardized field extraction
    source = example.get("en", "")
    translation = example.get("pt-br", "")

    # Fallback to other common field names
    if not source:
        source = example.get("prompt", example.get("text", example.get("content", "")))
    if not translation:
        translation = example.get("pt", example.get("translation", ""))

    return source, translation


def get_dataset_id(input_file: str) -> str:
    """Extract dataset ID from input filename, mapping to config keys."""
    import re

    from .config.datasets import DATASETS, FILENAME_MAPPINGS, FILENAME_PATTERNS

    filename = os.path.basename(input_file)

    # Check direct mapping first
    if filename in FILENAME_MAPPINGS:
        return FILENAME_MAPPINGS[filename]

    # Check for new simplified pattern: {pipeline_id}_{dataset_id}.json
    # Extract dataset_id from pattern like "20251011_195003_m_alert.json"
    pattern = FILENAME_PATTERNS["pipeline_dataset"]
    match = re.match(pattern, filename)
    if match:
        return match.group(1)

    # Fallback: try to match with config values
    for config_key, config_value in DATASETS.items():
        if config_value.replace("/", "_").replace(":", "_") in filename:
            return config_key

    # Last resort: use filename without extensions
    return (
        filename.replace(".json", "").replace("_train", "").replace("_test", "").lower()
    )


def get_model_key_from_name(model_name: str) -> str:
    """Convert model name to config key."""
    from .config.datasets import MODEL_NAME_MAPPINGS
    
    # Check direct mapping first
    if model_name in MODEL_NAME_MAPPINGS:
        return MODEL_NAME_MAPPINGS[model_name]
    
    # Check partial matches
    for key, value in MODEL_NAME_MAPPINGS.items():
        if key in model_name:
            return value
    
    # Fallback: return model_name as-is
    return model_name


def get_output_dir_name(output_type: str) -> str:
    """Get the numbered output directory name for a given operation type."""
    dir_mapping = {
        "translated": "01-translated",
        "evaluated": "02-evaluated", 
        "merged": "03-merged",
        "reviewed": "04-reviewed"
    }
    return dir_mapping.get(output_type, output_type)


def generate_output_filename(
    input_file: str,
    output_type: str = "translated",
    model_name: str = None,
    dataset_id: str = None,
    pipeline_id: str = None,
) -> str:
    """Generate output filename with clean, consistent pattern."""
    if dataset_id is None:
        dataset_id = get_dataset_id(input_file)

    # Use provided pipeline_id or generate new one
    if pipeline_id is None:
        pipeline_id = get_timestamp()

    # Clean pattern: {timestamp}_{dataset}_{model}_{operation}.json
    output_dir = f"output/{get_output_dir_name(output_type)}"

    if model_name:
        model_key = get_model_key_from_name(model_name)
        filename = f"{pipeline_id}_{dataset_id}_{model_key}_{output_type}.json"
    else:
        filename = f"{pipeline_id}_{dataset_id}_{output_type}.json"

    ensure_directory_exists(output_dir)
    return os.path.join(output_dir, filename)


def extract_pipeline_id(filename: str) -> str:
    """Extract pipeline_id from filename pattern: {pipeline_id}_{dataset_id}.json"""
    import re

    # Pattern: pipelineid_dataset.json
    pattern = r"^(\d{8}_\d{6})_[a-z_]+\.json$"
    match = re.match(pattern, os.path.basename(filename))
    if match:
        return match.group(1)
    return None


def generate_evaluation_filename(input_file: str, model_name: str = None) -> str:
    """Generate evaluation output filename."""
    return generate_output_filename(input_file, "evaluated", model_name)


def generate_review_filename(input_file: str, model_name: str = None) -> str:
    """Generate review output filename."""
    return generate_output_filename(input_file, "reviewed", model_name)


def generate_merge_filename(file1: str, file2: str) -> str:
    """Generate output filename for merged evaluations."""
    # Extract dataset name from first file
    file1_name = Path(file1).stem
    if "_" in file1_name:
        # Extract dataset from pattern: {timestamp}_{dataset}_{model}_{operation}
        parts = file1_name.split("_")
        if len(parts) >= 2:
            dataset_name = parts[1]  # Second part is dataset
        else:
            dataset_name = "merged"
    else:
        dataset_name = "merged"

    timestamp = get_timestamp()
    output_dir = f"output/{get_output_dir_name('merged')}"
    ensure_directory_exists(output_dir)
    return f"{output_dir}/{timestamp}_{dataset_name}_merged.json"


def generate_report_filename(
    dataset_id: str,
    operation: str,
    model_name: str = None,
    extension: str = "json",
    pipeline_id: str = None,
) -> str:
    """Generate report filename with simple, consistent logic."""
    if pipeline_id is None:
        pipeline_id = get_timestamp()

    # Reports go in main output folder
    report_dir = "output"
    ensure_directory_exists(report_dir)

    # Simple filename: pipeline_id + dataset_id + operation (same pattern as data files)
    filename = f"{pipeline_id}_{dataset_id}_{operation}.{extension}"

    return os.path.join(report_dir, filename)


# Removed unused functions: generate_log_filename, generate_visualization_path


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
