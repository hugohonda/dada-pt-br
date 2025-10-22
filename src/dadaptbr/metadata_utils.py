from typing import Any

from .utils import get_timestamp


def create_metadata(
    operation: str,
    pipeline_id: str,
    dataset_id: str,
    total_examples: int,
    processing_time: float,
    model_name: str = None,
    **analysis,
) -> dict[str, Any]:
    """Create metadata for any pipeline phase."""
    meta = {
        "operation": operation,
        "pipeline_id": pipeline_id,
        "dataset_id": dataset_id,
        "total_examples": total_examples,
        "timestamp": get_timestamp(),
        "processing_seconds": round(processing_time, 2),
    }
    if model_name:
        meta["model"] = model_name
    if analysis:
        meta["analysis"] = analysis
    return meta


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
    categories = len(
        {item.get("category") for item in translated_data if item.get("category")}
    )
    return create_metadata(
        "translation",
        pipeline_id,
        dataset_id,
        total_examples,
        processing_time,
        model_name,
        categories=categories,
        workers=max_workers,
    )


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
    scores = [item.get("score", 0) for item in results if item.get("score")]
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0
    return create_metadata(
        "evaluation",
        pipeline_id,
        dataset_id,
        total_examples,
        processing_time,
        model_name,
        avg_score=avg_score,
        batch_size=batch_size,
    )


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
    return create_metadata(
        "merge",
        pipeline_id,
        dataset_id,
        total_examples,
        processing_time,
        None,
        model1=model1,
        model2=model2,
        from_model1=stats["from_file1"],
        from_model2=stats["from_file2"],
        ties=stats["tie"],
    )


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
    return create_metadata(
        "review",
        pipeline_id,
        dataset_id,
        total_examples,
        processing_time,
        model_name,
        reviewed=review_stats["reviewed"],
        skipped=review_stats["skipped"],
        errors=review_stats["errors"],
    )
