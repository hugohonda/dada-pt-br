"""
Common utility functions
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


def detect_dataset_type(example: dict[str, Any], dataset_id: str = None) -> str:
    """Detect dataset type based on example structure and dataset config."""
    from .config.datasets import DATASET_CONFIGS

    # If dataset_id is provided, use config
    if dataset_id and dataset_id in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_id]["type"]

    # Fallback: detect from example structure
    keys = set(example.keys())

    # Check against all dataset configs
    for config in DATASET_CONFIGS.values():
        if all(key in keys for key in config["detection_keys"]):
            return config["type"]

    # Default to single language
    return "single"


def extract_texts(
    example: dict[str, Any], dataset_type: str, dataset_id: str = None
) -> tuple[str, str]:
    """Extract source and translation texts based on dataset type and config."""
    from .config.datasets import DATASET_CONFIGS

    # If dataset_id is provided, use config
    if dataset_id and dataset_id in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_id]
        source_field = config["source_field"]
        target_field = config["target_field"]
        return example.get(source_field, ""), example.get(target_field, "")

    # Fallback: use dataset type
    if dataset_type == "multilingual":
        return example.get("en", ""), example.get("pt", "")
    elif dataset_type == "safety":
        return example.get("prompt", ""), example.get("translation", "")
    else:
        return example.get("text", example.get("content", "")), example.get(
            "translation", ""
        )


def get_dataset_id(input_file: str) -> str:
    """Extract dataset ID from input filename, mapping to config keys."""
    from .config.datasets import DATASETS

    filename = os.path.basename(input_file)

    # Map downloaded filenames back to config keys
    filename_mappings = {
        # M-ALERT dataset
        "felfri_M-ALERT_train.json": "m_alert",
        "felfri_M-ALERT_test.json": "m_alert",
        # ALERT datasets
        "Babelscape_ALERT_alert_train.json": "alert",
        "Babelscape_ALERT_alert_test.json": "alert",
        "Babelscape_ALERT_alert_adversarial_train.json": "alert_adversarial",
        "Babelscape_ALERT_alert_adversarial_test.json": "alert_adversarial",
        # AgentHarm datasets
        "ai-safety-institute_AgentHarm_chat_train.json": "agent_harm_chat",
        "ai-safety-institute_AgentHarm_chat_test.json": "agent_harm_chat",
        "ai-safety-institute_AgentHarm_harmful_train.json": "agent_harm_harmful",
        "ai-safety-institute_AgentHarm_harmful_test.json": "agent_harm_harmful",
        "ai-safety-institute_AgentHarm_harmless_benign_train.json": "agent_harm_harmless",
        "ai-safety-institute_AgentHarm_harmless_benign_test.json": "agent_harm_harmless",
    }

    # Check direct mapping first
    if filename in filename_mappings:
        return filename_mappings[filename]

    # Check for new simplified pattern: {pipeline_id}_{dataset_id}.json
    # Extract dataset_id from pattern like "20251011_195003_m_alert.json"
    import re

    pattern = r"^\d{8}_\d{6}_([a-z_]+)\.json$"
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


def get_dataset_id_from_data(data: list) -> str:
    """Extract dataset ID from data content when filename is not available."""
    if not data:
        return "unknown"

    # Try to detect dataset type from data structure
    first_item = data[0]

    # Check for M-ALERT structure
    if "instruction" in first_item and "output" in first_item:
        return "m_alert"

    # Check for ALERT structure
    if "text" in first_item and "label" in first_item:
        return "alert"

    # Check for AgentHarm structure
    if "messages" in first_item:
        return "agent_harm_chat"

    # Check for AgentHarm harmful/harmless structure
    if "prompt" in first_item and "category" in first_item and "name" in first_item:
        # For now, default to harmful since we can't distinguish easily
        # This could be improved by checking the filename or other metadata
        return "agent_harm_harmful"

    # Default fallback
    return "unknown"


def get_category_translation(category: str, dataset_id: str) -> str:
    """Get translated category name from config."""
    from .config.datasets import DATASET_CONFIGS

    if dataset_id in DATASET_CONFIGS:
        categories = DATASET_CONFIGS[dataset_id]["categories"]
        return categories.get(category, category)

    return category


def get_model_key_from_name(model_name: str) -> str:
    """Convert model name to config key."""
    if "gemma3" in model_name:
        return "gemma3"
    elif "towerinstruct" in model_name:
        return "towerinstruct"
    else:
        return "unknown"


def generate_output_filename(
    input_file: str,
    output_type: str = "translated",
    model_name: str = None,
    dataset_id: str = None,
    pipeline_id: str = None,
) -> str:
    """Generate output filename with simple, consistent logic."""
    if dataset_id is None:
        dataset_id = get_dataset_id(input_file)

    # Use provided pipeline_id or generate new one
    if pipeline_id is None:
        pipeline_id = get_timestamp()

    # Simple structure: data/{output_type}/{model_key}/
    if model_name:
        model_key = get_model_key_from_name(model_name)
        output_dir = f"data/{output_type}/{model_key}"
        filename = f"{pipeline_id}_{dataset_id}.json"
    else:
        output_dir = f"data/{output_type}"
        filename = f"{pipeline_id}_{dataset_id}.json"

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

    # Reports go directly in analysis folder, no dataset subdirectory
    report_dir = "data/analysis"
    ensure_directory_exists(report_dir)

    # Simple filename: pipeline_id + dataset_id + operation (same pattern as data files)
    filename = f"{pipeline_id}_{dataset_id}_{operation}.{extension}"

    return os.path.join(report_dir, filename)


def generate_log_filename(
    dataset_id: str, operation: str, model_name: str = None
) -> str:
    """Generate log filename with organized structure."""
    timestamp = get_timestamp()

    if model_name:
        model_key = get_model_key_from_name(model_name)
        log_dir = f"outputs/logs/{operation}"
        filename = f"{model_key}_{dataset_id}_{operation}_{timestamp}.log"
    else:
        log_dir = f"outputs/logs/{operation}"
        filename = f"{dataset_id}_{operation}_{timestamp}.log"

    ensure_directory_exists(log_dir)
    return os.path.join(log_dir, filename)


def generate_visualization_path(dataset_id: str, visualization_name: str) -> str:
    """Generate visualization file path following data folder structure."""
    viz_dir = f"data/analysis/{dataset_id}"
    ensure_directory_exists(viz_dir)
    return os.path.join(viz_dir, f"{visualization_name}.png")


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
