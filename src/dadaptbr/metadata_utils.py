"""
Consolidated metadata utilities for all pipeline phases.
Provides consistent, meaningful, and non-redundant metadata generation.
"""

from typing import Any

from .utils import get_timestamp


def create_base_metadata(
    operation: str,
    pipeline_id: str,
    dataset_id: str,
    total_examples: int,
    processing_time: float,
    model_name: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Create base metadata structure shared across all phases."""
    base = {
        "operation": operation,
        "pipeline_id": pipeline_id,
        "dataset_id": dataset_id,
        "total_examples": total_examples,
        "timestamp": get_timestamp(),
        "processing_time": {
            "total_seconds": round(processing_time, 2),
            "avg_per_example": round(processing_time / total_examples, 3)
            if total_examples > 0
            else 0,
        },
    }

    # Add model info if provided
    if model_name:
        base["model_name"] = model_name

    # Add any additional fields
    base.update(kwargs)

    return base


def create_processing_time_metadata(
    total_seconds: float, total_examples: int, **additional_metrics
) -> dict[str, Any]:
    """Create standardized processing time metadata."""
    return {
        "total_seconds": round(total_seconds, 2),
        "avg_per_example": round(total_seconds / total_examples, 3)
        if total_examples > 0
        else 0,
        **additional_metrics,
    }


def create_translation_metadata(
    pipeline_id: str,
    dataset_id: str,
    total_examples: int,
    processing_time: float,
    model_name: str,
    max_workers: int,
    translated_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create metadata for translation phase."""
    # Calculate text length statistics
    source_lengths = [len(item.get("en", "")) for item in translated_data]
    translation_lengths = [len(item.get("pt-br", "")) for item in translated_data]

    # Get unique categories
    categories = list({item.get("category", "unknown") for item in translated_data})

    analysis = {
        "categories": categories,
        "category_count": len(categories),
        "avg_source_length": round(sum(source_lengths) / len(source_lengths), 2)
        if source_lengths
        else 0,
        "avg_translation_length": round(
            sum(translation_lengths) / len(translation_lengths), 2
        )
        if translation_lengths
        else 0,
    }

    processing_time_meta = create_processing_time_metadata(
        processing_time, total_examples, workers_used=max_workers
    )

    metadata = create_base_metadata(
        operation="translation",
        pipeline_id=pipeline_id,
        dataset_id=dataset_id,
        total_examples=total_examples,
        processing_time=processing_time,
        model_name=model_name,
        analysis=analysis,
    )

    # Override processing_time with detailed metrics
    metadata["processing_time"] = processing_time_meta


    return metadata


def create_evaluation_metadata(
    pipeline_id: str,
    dataset_id: str,
    total_examples: int,
    processing_time: float,
    results: list[dict[str, Any]],
    batch_size: int,
    total_batches: int,
    model_name: str = "xcomet-xl",
) -> dict[str, Any]:
    """Create metadata for evaluation phase."""
    # Calculate score statistics
    scores = [item.get("score", 0) for item in results if item.get("score") is not None]

    if scores:
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        std_score = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5

        score_distribution = {
            "high_quality": len([s for s in scores if s >= 0.8]),
            "medium_quality": len([s for s in scores if 0.6 <= s < 0.8]),
            "low_quality": len([s for s in scores if s < 0.6]),
        }
    else:
        avg_score = min_score = max_score = std_score = 0
        score_distribution = {"high_quality": 0, "medium_quality": 0, "low_quality": 0}

    # Count error spans
    error_spans_count = sum(len(item.get("error_spans", [])) for item in results)

    analysis = {
        "avg_score": round(avg_score, 4),
        "min_score": round(min_score, 4),
        "max_score": round(max_score, 4),
        "std_score": round(std_score, 4),
        "score_distribution": score_distribution,
        "error_spans_count": error_spans_count,
        "evaluation_model": model_name,
    }

    processing_time_meta = create_processing_time_metadata(
        processing_time,
        total_examples,
        batch_size=batch_size,
        total_batches=total_batches,
    )

    metadata = create_base_metadata(
        operation="evaluation",
        pipeline_id=pipeline_id,
        dataset_id=dataset_id,
        total_examples=total_examples,
        processing_time=processing_time,
        model_name=model_name,
        analysis=analysis,
    )

    # Override processing_time with detailed metrics
    metadata["processing_time"] = processing_time_meta


    return metadata


def create_merge_metadata(
    pipeline_id: str,
    dataset_id: str,
    total_examples: int,
    processing_time: float,
    model1: str,
    model2: str,
    stats: dict[str, int],
    source_files: dict[str, str],
) -> dict[str, Any]:
    """Create metadata for merge phase."""
    analysis = {
        "model1": model1,
        "model2": model2,
        "from_model1_count": stats["from_file1"],
        "from_model2_count": stats["from_file2"],
        "tie_count": stats["tie"],
        "model1_win_rate": round(stats["from_file1"] / total_examples, 4)
        if total_examples > 0
        else 0,
        "model2_win_rate": round(stats["from_file2"] / total_examples, 4)
        if total_examples > 0
        else 0,
        "tie_rate": round(stats["tie"] / total_examples, 4)
        if total_examples > 0
        else 0,
    }

    processing_time_meta = create_processing_time_metadata(
        processing_time, total_examples
    )

    metadata = create_base_metadata(
        operation="merge",
        pipeline_id=pipeline_id,
        dataset_id=dataset_id,
        total_examples=total_examples,
        processing_time=processing_time,
        analysis=analysis,
    )
    
    # Add source files info
    metadata["source_files"] = source_files

    # Override processing_time with detailed metrics
    metadata["processing_time"] = processing_time_meta

    return metadata


def create_review_metadata(
    pipeline_id: str,
    dataset_id: str,
    total_examples: int,
    processing_time: float,
    model_name: str,
    reviewed_data: list[dict[str, Any]],
    review_stats: dict[str, int],
) -> dict[str, Any]:
    """Create metadata for review phase."""
    # Calculate review statistics
    reviewed_items = [
        item for item in reviewed_data if item.get("reviewed_translation", "").strip()
    ]
    improvement_rate = (
        review_stats["reviewed"] / total_examples if total_examples > 0 else 0
    )

    # Calculate average score of reviewed items
    avg_score = 0
    if reviewed_items:
        scores = [item.get("score", 0) for item in reviewed_items]
        avg_score = round(sum(scores) / len(scores), 4) if scores else 0

    analysis = {
        "reviewed_count": review_stats["reviewed"],
        "skipped_count": review_stats["skipped"],
        "error_count": review_stats["errors"],
        "improvement_rate": round(improvement_rate, 4),
        "avg_score": avg_score,
    }

    processing_time_meta = create_processing_time_metadata(
        processing_time, total_examples
    )

    metadata = create_base_metadata(
        operation="review",
        pipeline_id=pipeline_id,
        dataset_id=dataset_id,
        total_examples=total_examples,
        processing_time=processing_time,
        model_name=model_name,
        analysis=analysis,
    )

    # Override processing_time with detailed metrics
    metadata["processing_time"] = processing_time_meta


    return metadata


def extract_metadata_from_file(file_path: str) -> dict[str, Any] | None:
    """Extract metadata from a JSON file if it exists."""
    try:
        import json

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("metadata") if isinstance(data, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def validate_metadata_consistency(metadata: dict[str, Any]) -> list[str]:
    """Validate metadata structure and return any issues found."""
    issues = []

    required_fields = [
        "operation",
        "pipeline_id",
        "dataset_id",
        "total_examples",
        "timestamp",
        "processing_time",
    ]
    for field in required_fields:
        if field not in metadata:
            issues.append(f"Missing required field: {field}")

    # Validate processing_time structure
    if "processing_time" in metadata:
        pt = metadata["processing_time"]
        if not isinstance(pt, dict):
            issues.append("processing_time should be a dictionary")
        else:
            if "total_seconds" not in pt:
                issues.append("processing_time missing total_seconds")
            if "avg_per_example" not in pt:
                issues.append("processing_time missing avg_per_example")

    return issues
