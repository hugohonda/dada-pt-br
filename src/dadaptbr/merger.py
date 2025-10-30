import argparse
import time

from .config.datasets import DEFAULT_MODELS
from .config.logging import setup_logger
from .metadata_utils import create_merge_metadata
from .utils import (
    generate_merge_filename,
    get_dataset_id,
    get_timestamp,
    load_json_file,
    save_json_file,
)

_LOGGER = setup_logger(__name__)


def merge_evaluations(
    file1_path: str, file2_path: str, output_path: str, limit: int = None
):
    """Merge two evaluation files, keeping the best translation for each example."""
    start_time = time.time()
    _LOGGER.info(f"Merging evaluations from: {file1_path} and {file2_path}")

    # Load both evaluation files
    data1, metadata1 = load_json_file(file1_path)
    data2, metadata2 = load_json_file(file2_path)

    if limit:
        data1 = data1[:limit]
        data2 = data2[:limit]
        _LOGGER.info(f"Limited to {limit} examples")

    # Ensure both files have the same number of examples
    if len(data1) != len(data2):
        _LOGGER.warning(f"Different number of examples: {len(data1)} vs {len(data2)}")
        min_length = min(len(data1), len(data2))
        data1 = data1[:min_length]
        data2 = data2[:min_length]
        _LOGGER.info(f"Using first {min_length} examples from each file")

    merged_data = []
    stats = {"total": len(data1), "from_file1": 0, "from_file2": 0, "tie": 0}

    for i, (example1, example2) in enumerate[tuple[dict, dict]](
        zip[tuple[dict, dict]](data1, data2, strict=False)
    ):
        # Ensure both examples have the same source and index
        if example1.get("source") != example2.get("source"):
            _LOGGER.warning(f"Example {i}: Different sources, skipping")
            continue

        if example1.get("index") != example2.get("index"):
            _LOGGER.warning(f"Example {i}: Different indices, skipping")
            continue

        # Compare scores and pick the best
        score1: float = example1.get("score", 0.0)
        score2: float = example2.get("score", 0.0)
        model1: str = metadata1.get("model", "unknown")
        model2: str = metadata2.get("model", "unknown")

        # Pick best translation (or tie-breaker)
        if score1 > score2 or (
            score1 == score2 and model1 == DEFAULT_MODELS["tie_breaker"]
        ):
            first: dict = example1
            second: dict = example2
            first_model: str = model1
            second_model: str = model2
            first_score, second_score = score1, score2
            stats["from_file1"] += 1 if score1 > score2 else 0
            stats["from_file2"] += 1 if score2 > score1 else 0
            stats["tie"] += 1 if score1 == score2 else 0
        else:
            first: dict = example2
            second: dict = example1
            first_model: str = model2
            second_model: str = model1
            first_score: float = score2
            second_score: float = score1
            stats["from_file2"] += 1 if score2 > score1 else 0
            stats["from_file1"] += 1 if score1 > score2 else 0
            stats["tie"] += 1 if score1 == score2 else 0

        # Create merged entry with both translations for review
        best_example: dict = {
            "index": first.get("index", i),
            "id": first.get("id", i),
            "source": first.get("source", ""),
            "translation": first.get("translation", ""),
            "score": round(first_score, 4),
            "merged_from": first_model,
            "second_translation": second.get("translation", ""),
            "second_score": round(second_score, 4),
            "second_model": second_model,
            "error_spans": first.get("error_spans", []),
        }

        merged_data.append(best_example)

    # Calculate total processing time
    total_processing_time = time.time() - start_time

    # Extract model names for analysis
    model1: str = metadata1.get("model_name", "unknown")
    model2: str = metadata2.get("model_name", "unknown")

    # Create merge metadata
    merge_metadata: dict = create_merge_metadata(
        pipeline_id=get_timestamp(),
        dataset_id=get_dataset_id(file1_path),
        total_examples=len(merged_data),
        processing_time=total_processing_time,
        model1=model1,
        model2=model2,
        stats=stats,
        source_files={"file1": file1_path, "file2": file2_path},
    )

    # Create final data structure
    final_data: dict = {"metadata": merge_metadata, "data": merged_data}

    # Save merged data
    save_json_file(final_data, output_path)

    # Log statistics
    _LOGGER.info("Merge completed:")
    _LOGGER.info(f"  Total examples: {stats['total']}")
    _LOGGER.info(f"  From file1: {stats['from_file1']}")
    _LOGGER.info(f"  From file2: {stats['from_file2']}")
    _LOGGER.info(f"  Ties: {stats['tie']}")
    _LOGGER.info(f"  Merged results saved to: {output_path}")

    return merged_data


def main():
    """CLI entry point for merger."""
    parser = argparse.ArgumentParser(description="Merge evaluation results")
    parser.add_argument("file1", help="First evaluation JSON file")
    parser.add_argument("file2", help="Second evaluation JSON file")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--limit", "-l", type=int, help="Limit examples")

    args = parser.parse_args()

    output = args.output or generate_merge_filename(args.file1, args.file2)
    merge_evaluations(args.file1, args.file2, output, args.limit)


if __name__ == "__main__":
    main()
