"""
Standardized report generation utilities.

Ensures consistent structure and format across all reports (translation, evaluation, analysis).
"""

import platform
import time
from datetime import datetime
from typing import Any

from .config.logging import setup_logger
from .llm_client import get_model_info
from .utils import (
    extract_pipeline_id,
    format_duration,
    generate_report_filename,
    get_dataset_id,
    save_json_file,
)

_LOGGER = setup_logger("report_generator", log_to_file=True, log_prefix="report")

# Global tracking variables for translation
_translation_stats = {
    "start_time": None,
    "end_time": None,
    "total_examples": 0,
    "successful_translations": 0,
    "failed_translations": 0,
    "errors": [],
    "processing_times": [],
}


def get_system_info() -> dict[str, Any]:
    """Get basic system information."""
    try:
        return {
            "platform": platform.system(),
            "python_version": platform.python_version(),
        }
    except Exception as e:
        return {"error": str(e)}


def get_performance_metrics(
    processing_times: list[float] = None, total_wall_time: float = None
) -> dict[str, Any]:
    """Get processing performance metrics."""
    if not processing_times:
        return {}

    try:
        # For parallel processing (like translation), use wall time as total
        # For sequential processing (like evaluation), use sum of individual times
        total_time = (
            total_wall_time if total_wall_time is not None else sum(processing_times)
        )

        return {
            "processing_times_stats": {
                "min_seconds": round(min(processing_times), 2),
                "max_seconds": round(max(processing_times), 2),
                "median_seconds": round(
                    sorted(processing_times)[len(processing_times) // 2], 2
                ),
                "total_processing_time": round(total_time, 2),
            }
        }
    except Exception as e:
        return {"error": str(e)}


def create_standard_report(
    operation: str,
    input_file: str,
    output_file: str,
    dataset_type: str,
    model_name: str,
    pipeline_id: str = None,
    **kwargs,
) -> dict[str, Any]:
    """Create a standardized report structure."""

    # Common fields for all reports
    timestamp = datetime.now().isoformat()
    if pipeline_id:
        timestamp = (
            pipeline_id.replace("_", "T") + ":00"
        )  # Convert pipeline_id to ISO format

    report = {
        "operation_summary": {
            "timestamp": timestamp,
            "operation": operation,
            "input_file": input_file,
            "output_file": output_file,
            "dataset_type": dataset_type,
            "pipeline_id": pipeline_id,
        },
        "model_information": get_model_info(model_name)
        if operation == "translation"
        else {"name": model_name},
        "system_information": get_system_info(),
    }

    # Add operation-specific fields
    if operation == "translation":
        report["translation_metrics"] = {
            "total_examples": kwargs.get("total_examples", 0),
            "successful_translations": kwargs.get("successful_translations", 0),
            "failed_translations": kwargs.get("failed_translations", 0),
            "success_rate_percent": kwargs.get("success_rate_percent", 0.0),
            "total_time_seconds": kwargs.get("total_time_seconds", 0.0),
            "total_time_formatted": kwargs.get("total_time_formatted", "0s"),
            "average_processing_time_seconds": kwargs.get(
                "average_processing_time_seconds", 0.0
            ),
            "examples_per_minute": kwargs.get("examples_per_minute", 0.0),
        }
        report["performance_metrics"] = get_performance_metrics(
            kwargs.get("processing_times", []), kwargs.get("total_time_seconds")
        )
        report["errors"] = kwargs.get("errors", [])

    elif operation == "evaluation":
        report["evaluation_metrics"] = {
            "total_examples": kwargs.get("total_examples", 0),
            "evaluated_examples": kwargs.get("evaluated_examples", 0),
            "limit": kwargs.get("limit", None),
            "mean_score": kwargs.get("mean_score", 0.0),
            "min_score": kwargs.get("min_score", 0.0),
            "max_score": kwargs.get("max_score", 0.0),
            "std_score": kwargs.get("std_score", 0.0),
            "batch_size": kwargs.get("batch_size", None),
            "total_batches": kwargs.get("total_batches", 0),
        }
        report["performance_metrics"] = get_performance_metrics(
            kwargs.get("processing_times", [])
        )

    elif operation == "analysis":
        report["analysis_metrics"] = {
            "total_examples": kwargs.get("total_examples", 0),
            "models_analyzed": kwargs.get("models_analyzed", []),
            "visualizations_created": kwargs.get("visualizations_created", []),
        }
        report["performance_metrics"] = get_performance_metrics(
            kwargs.get("processing_times", [])
        )

    return report


def generate_standard_summary_report(report: dict[str, Any], summary_file: str):
    """Generate a standardized human-readable summary report."""

    operation = report["operation_summary"]["operation"]
    operation_title = operation.replace("_", " ").title()

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"{operation_title.upper()} REPORT SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # Operation Summary
        summary = report["operation_summary"]
        f.write(f"{operation_title.upper()} SUMMARY:\n")
        f.write(f"  Timestamp: {summary['timestamp']}\n")
        f.write(f"  Input File: {summary['input_file']}\n")
        f.write(f"  Output File: {summary['output_file']}\n")
        f.write(f"  Dataset Type: {summary['dataset_type']}\n")
        if summary.get("pipeline_id"):
            f.write(f"  Pipeline ID: {summary['pipeline_id']}\n")
        f.write("\n")

        # Operation-specific metrics
        if operation == "translation":
            metrics = report["translation_metrics"]
            f.write("TRANSLATION METRICS:\n")
            f.write(f"  Total Examples: {metrics['total_examples']}\n")
            f.write(
                f"  Successful Translations: {metrics['successful_translations']}\n"
            )
            f.write(f"  Failed Translations: {metrics['failed_translations']}\n")
            f.write(f"  Success Rate: {metrics['success_rate_percent']}%\n")
            f.write(f"  Total Time: {metrics['total_time_formatted']}\n")
            f.write(f"  Examples/Minute: {metrics['examples_per_minute']}\n")
            f.write(
                f"  Avg Processing Time: {metrics['average_processing_time_seconds']}s\n\n"
            )

        elif operation == "evaluation":
            metrics = report["evaluation_metrics"]
            f.write("EVALUATION METRICS:\n")
            f.write(f"  Total Examples: {metrics['total_examples']}\n")
            f.write(f"  Evaluated Examples: {metrics['evaluated_examples']}\n")
            f.write(f"  Limit: {metrics['limit'] or 'All'}\n")
            f.write(f"  Mean Score: {metrics['mean_score']}\n")
            f.write(f"  Min Score: {metrics['min_score']}\n")
            f.write(f"  Max Score: {metrics['max_score']}\n")
            if metrics.get("std_score"):
                f.write(f"  Std Score: {metrics['std_score']}\n")
            if metrics.get("batch_size") and metrics.get("total_batches"):
                f.write(f"  Batch Size: {metrics['batch_size']}\n")
                f.write(f"  Total Batches: {metrics['total_batches']}\n")
            f.write("\n")

        elif operation == "analysis":
            metrics = report["analysis_metrics"]
            f.write("ANALYSIS METRICS:\n")
            f.write(f"  Total Examples: {metrics['total_examples']}\n")
            f.write(f"  Models Analyzed: {', '.join(metrics['models_analyzed'])}\n")
            f.write(
                f"  Visualizations Created: {', '.join(metrics['visualizations_created'])}\n"
            )
            f.write("\n")

        # Model Information
        f.write("MODEL INFORMATION:\n")
        model = report["model_information"]
        for key, value in model.items():
            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")

        # System Information
        f.write("SYSTEM INFORMATION:\n")
        sys_info = report["system_information"]
        if "error" not in sys_info:
            f.write(f"  Platform: {sys_info.get('platform', 'Unknown')}\n")
            f.write(f"  Python Version: {sys_info.get('python_version', 'Unknown')}\n")
        else:
            f.write(f"  Error getting system info: {sys_info['error']}\n")
        f.write("\n")

        # Performance Metrics (only if available)
        perf = report["performance_metrics"]
        if perf and "error" not in perf and "processing_times_stats" in perf:
            if operation == "evaluation":
                f.write("BATCH PERFORMANCE METRICS:\n")
            else:
                f.write("PERFORMANCE METRICS:\n")
            stats = perf["processing_times_stats"]
            f.write(f"  Min Processing Time: {stats.get('min_seconds', 'Unknown')}s\n")
            f.write(f"  Max Processing Time: {stats.get('max_seconds', 'Unknown')}s\n")
            f.write(
                f"  Median Processing Time: {stats.get('median_seconds', 'Unknown')}s\n"
            )
            f.write(
                f"  Total Processing Time: {stats.get('total_processing_time', 'Unknown')}s\n"
            )
            f.write("\n")

        # Errors (for translation)
        if operation == "translation" and report.get("errors"):
            f.write("ERRORS:\n")
            for error in report["errors"]:
                f.write(f"  - {error}\n")
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("End of Report\n")
        f.write("=" * 60 + "\n")


def save_standard_report(report: dict[str, Any], report_file: str):
    """Save a standardized report to JSON file."""
    save_json_file(report, report_file)
    _LOGGER.info(f"Standard report saved to: {report_file}")


def generate_standard_reports(
    operation: str,
    input_file: str,
    output_file: str,
    dataset_type: str,
    model_name: str,
    pipeline_id: str = None,
    report_file: str = None,
    summary_file: str = None,
    **kwargs,
) -> tuple[str, str]:
    """Generate both JSON and TXT standardized reports."""

    # Create standardized report
    report = create_standard_report(
        operation,
        input_file,
        output_file,
        dataset_type,
        model_name,
        pipeline_id,
        **kwargs,
    )

    # Save JSON report
    if report_file:
        save_standard_report(report, report_file)

    # Generate and save TXT summary
    if summary_file:
        generate_standard_summary_report(report, summary_file)
        _LOGGER.info(f"Standard summary saved to: {summary_file}")

    return report_file, summary_file


# Translation tracking functions (for backward compatibility)
def start_translation():
    """Mark the start of translation process."""
    _translation_stats["start_time"] = time.time()
    _LOGGER.info("Translation process started")


def end_translation():
    """Mark the end of translation process."""
    _translation_stats["end_time"] = time.time()
    _LOGGER.info("Translation process completed")


def add_translation_result(success: bool, processing_time: float, error: str = None):
    """Add a translation result to statistics."""
    _translation_stats["total_examples"] += 1
    if success:
        _translation_stats["successful_translations"] += 1
        _translation_stats["processing_times"].append(processing_time)
    else:
        _translation_stats["failed_translations"] += 1
        if error:
            _translation_stats["errors"].append(error)


def generate_translation_report(
    input_file: str,
    output_file: str,
    dataset_type: str,
    model_name: str = "gemma3:latest",
) -> str:
    """Generate a comprehensive translation report using standardized format."""
    if not _translation_stats["start_time"] or not _translation_stats["end_time"]:
        _LOGGER.warning("Translation timing not properly recorded")
        return None

    # Calculate statistics
    total_time = _translation_stats["end_time"] - _translation_stats["start_time"]
    avg_processing_time = (
        sum(_translation_stats["processing_times"])
        / len(_translation_stats["processing_times"])
        if _translation_stats["processing_times"]
        else 0
    )

    success_rate = (
        _translation_stats["successful_translations"]
        / _translation_stats["total_examples"]
        * 100
        if _translation_stats["total_examples"] > 0
        else 0
    )

    # Extract pipeline_id from output_file for consistent naming

    dataset_id = get_dataset_id(input_file)
    pipeline_id = extract_pipeline_id(output_file)

    report_file = generate_report_filename(
        dataset_id, "translation", model_name, "json", pipeline_id
    )
    summary_file = generate_report_filename(
        dataset_id, "translation", model_name, "txt", pipeline_id
    )

    # Generate standardized reports
    generate_standard_reports(
        operation="translation",
        input_file=input_file,
        output_file=output_file,
        dataset_type=dataset_type,
        model_name=model_name,
        pipeline_id=pipeline_id,
        report_file=report_file,
        summary_file=summary_file,
        total_examples=_translation_stats["total_examples"],
        successful_translations=_translation_stats["successful_translations"],
        failed_translations=_translation_stats["failed_translations"],
        success_rate_percent=round(success_rate, 2),
        total_time_seconds=round(total_time, 2),
        total_time_formatted=format_duration(total_time),
        average_processing_time_seconds=round(avg_processing_time, 2),
        examples_per_minute=round(
            _translation_stats["total_examples"] / (total_time / 60), 2
        ),
        processing_times=_translation_stats["processing_times"],
        errors=_translation_stats["errors"],
    )

    _LOGGER.info(f"Translation report saved to: {report_file}")
    _LOGGER.info(f"Summary report saved to: {summary_file}")

    return report_file
