import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_timestamp() -> str:
    """Get standardized UTC timestamp for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)


def load_json_file(file_path: str) -> tuple[list[dict], dict]:
    """Load JSON file and return (data, metadata) tuple.
    Handles both old format (list) and new format (dict with data/metadata).
    """
    with open(file_path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "data" in raw:
        return raw["data"], raw.get("metadata", {})
    return raw, {}


def resolve_dataset_file(input_path_or_key: str) -> str:
    """Resolve dataset file path.

    If path exists, return it. Otherwise, lookup filename from DATASET_FILES dict.
    """
    if os.path.exists(input_path_or_key):
        return input_path_or_key

    from .config.datasets import DATASET_FILES

    if input_path_or_key not in DATASET_FILES:
        raise FileNotFoundError(
            f"Dataset '{input_path_or_key}' not found. Available: {list(DATASET_FILES.keys())}"
        )

    filename = DATASET_FILES[input_path_or_key]
    file_path = os.path.join("datasets", filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    return file_path


def save_json_file(data: Any, file_path: str, indent: int = 2) -> None:
    """Atomically save data to JSON file (write temp + fsync + replace)."""
    directory = os.path.dirname(file_path)
    ensure_directory_exists(directory)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def extract_texts(example: dict[str, Any]) -> tuple[str, str]:
    """Extract source and translation texts using simplified logic."""
    source = example.get("en", "")
    translation = example.get("pt-br", "")

    # Fallback to other common field names
    if not source:
        source = example.get("prompt", example.get("text", example.get("content", "")))
    if not translation:
        translation = example.get("pt", example.get("translation", ""))

    return source, translation


def get_dataset_id(input_file: str) -> str:
    """Extract dataset ID from filename."""
    import re

    from .config.datasets import FILENAME_MAPPINGS

    filename = os.path.basename(input_file)

    if filename in FILENAME_MAPPINGS:
        return FILENAME_MAPPINGS[filename]

    # Pattern: {timestamp}_{dataset_id}_{model}_{phase}.json
    match = re.search(r"\d{8}_\d{6}Z_([a-z_]+)", filename)
    if match:
        return match.group(1)

    # Fallback: clean filename
    return (
        filename.replace(".json", "").replace("_train", "").replace("_test", "").lower()
    )


def get_model_key(model_name: str) -> str:
    """Simplify model name to key (gemma3:latest -> gemma3)."""
    if not model_name:
        return "unknown"
    # Extract base name before : or /
    name = model_name.split(":")[0].split("/")[-1]
    return name.replace("towerinstruct-mistral", "tower").replace("gemma3", "gemma3")


OUTPUT_DIRS = {
    "translated": "01-translated",
    "evaluated": "02-evaluated",
    "merged": "03-merged",
    "reviewed": "04-reviewed",
}


def generate_output_filename(
    input_file: str,
    phase: str,
    model_name: str = None,
    dataset_id: str = None,
    pipeline_id: str = None,
) -> str:
    """Generate output filename: {pipeline_id}_{dataset}_{model?}_{phase}.json"""
    dataset_id = dataset_id or get_dataset_id(input_file)
    pipeline_id = pipeline_id or get_timestamp()
    model_key = get_model_key(model_name) if model_name else None

    output_dir = f"output/{OUTPUT_DIRS.get(phase, phase)}"
    ensure_directory_exists(output_dir)

    parts = [pipeline_id, dataset_id]
    if model_key:
        parts.append(model_key)
    parts.append(phase)

    filename = "_".join(parts) + ".json"
    return os.path.join(output_dir, filename)


def extract_pipeline_id(filename: str) -> str:
    """Extract pipeline_id from filename (first part before _)."""
    import re

    match = re.match(r"^(\d{8}_\d{6}Z)", os.path.basename(filename))
    return match.group(1) if match else get_timestamp()


def generate_evaluation_filename(f, m=None):
    return generate_output_filename(f, "evaluated", m)


def generate_review_filename(f, m=None):
    return generate_output_filename(f, "reviewed", m)


def generate_merge_filename(file1: str, file2: str) -> str:
    """Generate merged output filename."""
    dataset_id = get_dataset_id(file1)
    pipeline_id = extract_pipeline_id(file1)
    return generate_output_filename(file1, "merged", None, dataset_id, pipeline_id)


def resolve_workers(workers: int | str, phase: str) -> int:
    """Resolve workers value; supports 'auto' and clamps by phase caps."""
    from .config.datasets import PHASE_WORKERS

    if isinstance(workers, str) and workers == "auto":
        cpu = os.cpu_count() or 4
        auto = min(32, max(4, cpu * 2))
        return min(auto, PHASE_WORKERS.get(phase, {}).get("max", auto))
    return int(workers)


def resolve_device(device: str) -> str:
    """Resolve 'auto' device to cuda if available else cpu."""
    if device != "auto":
        return device
    try:
        import torch  # noqa: WPS433

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def write_run_manifest(
    pipeline_id: str,
    dataset_id: str,
    config: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> str:
    """Write a compact manifest for the full pipeline run and return its path."""
    run_dir = Path("output") / "runs" / pipeline_id
    ensure_directory_exists(str(run_dir))
    manifest_path = run_dir / "manifest.json"

    manifest = {
        "pipeline_id": pipeline_id,
        "dataset_id": dataset_id,
        "created_at": get_timestamp(),
        "config": config,
        "artifacts": artifacts,
    }

    save_json_file(manifest, str(manifest_path))
    return str(manifest_path)


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
