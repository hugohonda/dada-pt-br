import argparse

from .config.datasets import DEFAULT_MODELS
from .config.logging import setup_logger
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
    import time

    start_time = time.time()
    _LOGGER.info(f"Merging evaluations from: {file1_path} and {file2_path}")

    # Load both evaluation files
    raw_data1 = load_json_file(file1_path)
    raw_data2 = load_json_file(file2_path)

    # Handle new data structure with metadata
    if isinstance(raw_data1, dict) and "data" in raw_data1:
        data1 = raw_data1["data"]
        metadata1 = raw_data1.get("metadata", {})
        _LOGGER.info(f"Loaded file1 with metadata: {metadata1}")
    else:
        data1 = raw_data1
        metadata1 = {}

    if isinstance(raw_data2, dict) and "data" in raw_data2:
        data2 = raw_data2["data"]
        metadata2 = raw_data2.get("metadata", {})
        _LOGGER.info(f"Loaded file2 with metadata: {metadata2}")
    else:
        data2 = raw_data2
        metadata2 = {}

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

    for i, (example1, example2) in enumerate(zip(data1, data2, strict=False)):
        # Ensure both examples have the same source and index
        if example1.get("source") != example2.get("source"):
            _LOGGER.warning(f"Example {i}: Different sources, skipping")
            continue

        if example1.get("index") != example2.get("index"):
            _LOGGER.warning(f"Example {i}: Different indices, skipping")
            continue

        # Compare scores to determine which translation to keep
        score1 = example1.get("score", 0.0)
        score2 = example2.get("score", 0.0)

        # Extract model names from metadata
        model1 = metadata1.get("source_metadata", {}).get("model_name", "unknown")
        model2 = metadata2.get("source_metadata", {}).get("model_name", "unknown")

        # Helper function to optimize error spans
        def optimize_error_spans(error_spans):
            """Keep only essential error span information."""
            if not error_spans:
                return []

            optimized = []
            for span in error_spans:
                # Keep only essential fields: text, severity, and position
                optimized_span = {
                    "text": span.get("text", ""),
                    "severity": span.get("severity", "minor"),
                    "start": span.get("start", 0),
                    "end": span.get("end", 0),
                }
                # Only include confidence if it's high (significant error)
                if span.get("confidence", 0) > 0.5:
                    optimized_span["confidence"] = round(span.get("confidence", 0), 2)
                optimized.append(optimized_span)

            return optimized

        # Helper function to create optimized merged example
        def create_optimized_example(
            selected_example,
            selected_model,
            selected_score,
            alternative_example,
            alternative_model,
            alternative_score,
            merge_reason,
        ):
            """Create a clean merged example with only essential information."""
            # Only include alternative information if scores are close (within 0.1)
            score_diff = abs(selected_score - alternative_score)
            include_alternative = score_diff < 0.1

            result = {
                # Core fields
                "index": selected_example.get("index", 0),
                "id": selected_example.get("id", 0),
                "source": selected_example.get("source", ""),
                "translation": selected_example.get("translation", ""),
                "score": round(selected_score, 4),  # Round to 4 decimal places
                # Merge metadata
                "merged_from": merge_reason,
                "original_model": selected_model,
                # Optimized error spans (only from selected translation)
                "error_spans": optimize_error_spans(
                    selected_example.get("error_spans", [])
                ),
            }

            # Only include alternative info if scores are close
            if include_alternative:
                result.update(
                    {
                        "alternative_score": round(alternative_score, 4),
                        "alternative_translation": alternative_example.get(
                            "translation", ""
                        ),
                        "alternative_model": alternative_model,
                    }
                )

            # Only include system score if it's significantly different from main score
            system_score = selected_example.get("system_score", selected_score)
            if abs(system_score - selected_score) > 0.05:
                result["system_score"] = round(system_score, 4)

            return result

        if score1 > score2:
            # Model 1 has better score
            best_example = create_optimized_example(
                example1, model1, score1, example2, model2, score2, model1
            )
            stats["from_file1"] += 1
        elif score2 > score1:
            # Model 2 has better score
            best_example = create_optimized_example(
                example2, model2, score2, example1, model1, score1, model2
            )
            stats["from_file2"] += 1
        else:
            # Tie - prefer configured tie-breaker model if available, otherwise prefer model1
            tie_breaker = DEFAULT_MODELS["tie_breaker"]
            if model1 == tie_breaker:
                best_example = create_optimized_example(
                    example1, model1, score1, example2, model2, score2, f"{model1}_tie"
                )
            elif model2 == tie_breaker:
                best_example = create_optimized_example(
                    example2, model2, score2, example1, model1, score1, f"{model2}_tie"
                )
            else:
                # Neither is Tower, prefer model1
                best_example = create_optimized_example(
                    example1, model1, score1, example2, model2, score2, f"{model1}_tie"
                )
            stats["tie"] += 1

        merged_data.append(best_example)

    # Calculate total processing time
    total_processing_time = time.time() - start_time

    # Extract model names for analysis
    model1 = metadata1.get("source_metadata", {}).get("model_name", "unknown")
    model2 = metadata2.get("source_metadata", {}).get("model_name", "unknown")

    # Create merge metadata with model analysis using consolidated approach
    from .metadata_utils import create_merge_metadata

    merge_metadata = create_merge_metadata(
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
    final_data = {"metadata": merge_metadata, "data": merged_data}

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
