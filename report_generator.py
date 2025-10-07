#!/usr/bin/env python3
"""
Translation report generator with comprehensive statistics.
"""

import platform
import time
from datetime import datetime
from typing import Any

import ollama
import psutil

from config.logging import setup_logger
from utils import ensure_directory_exists, format_duration, save_json_file

_LOGGER = setup_logger("report_generator", log_to_file=True, log_prefix="report")

# Global tracking variables
_translation_stats = {
    "start_time": None,
    "end_time": None,
    "total_examples": 0,
    "successful_translations": 0,
    "failed_translations": 0,
    "errors": [],
    "processing_times": [],
}


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
    _translation_stats["processing_times"].append(processing_time)

    if success:
        _translation_stats["successful_translations"] += 1
    else:
        _translation_stats["failed_translations"] += 1
        if error:
            _translation_stats["errors"].append(error)


def get_model_info():
    """Get information about the Ollama model being used."""
    try:
        client = ollama.Client()
        models = client.list()

        # Find gemma3:latest model
        for model in models.models:
            if model.model == "gemma3:latest":
                return {
                    "name": model.model,
                    "size": getattr(model, "size", "Unknown"),
                    "modified_at": str(getattr(model, "modified_at", "Unknown")),
                    "digest": getattr(model, "digest", "Unknown")[:12] + "..."
                    if hasattr(model, "digest")
                    else "Unknown",
                }

        return {"name": "gemma3:latest", "status": "Model not found in Ollama list"}

    except Exception as e:
        _LOGGER.warning(f"Could not get model info: {e}")
        return {"name": "gemma3:latest", "error": str(e)}


def get_system_info():
    """Get basic system information."""
    try:
        return {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        }
    except Exception as e:
        return {"error": str(e)}


def generate_recommendations(
    success_rate: float, avg_time: float, total_time: float
) -> list[str]:
    """Generate recommendations based on performance metrics."""
    recommendations = []

    if success_rate < 95:
        recommendations.append("Consider checking error logs for translation failures")

    if avg_time > 15:
        recommendations.append(
            "Translation is slow - consider using a faster model or reducing prompt complexity"
        )

    if total_time > 3600:  # More than 1 hour
        recommendations.append(
            "Long translation time - consider processing in smaller batches"
        )

    if not recommendations:
        recommendations.append("Translation performance looks good!")

    return recommendations


def generate_summary_report(report: dict[str, Any], summary_file: str):
    """Generate a human-readable summary report."""
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("TRANSLATION REPORT SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # Translation Summary
        summary = report["translation_summary"]
        f.write("TRANSLATION SUMMARY:\n")
        f.write(f"  Timestamp: {summary['timestamp']}\n")
        f.write(f"  Input File: {summary['input_file']}\n")
        f.write(f"  Output File: {summary['output_file']}\n")
        f.write(f"  Dataset Type: {summary['dataset_type']}\n")
        f.write(f"  Total Time: {summary['total_time_formatted']}\n")
        f.write(f"  Examples Processed: {summary['total_examples']}\n")
        f.write(f"  Success Rate: {summary['success_rate_percent']}%\n")
        f.write(f"  Examples/Minute: {summary['examples_per_minute']}\n")
        f.write(
            f"  Avg Processing Time: {summary['average_processing_time_seconds']}s\n\n"
        )

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
            f.write(f"  Python: {sys_info.get('python_version', 'Unknown')}\n")
        else:
            f.write(f"  Error getting system info: {sys_info['error']}\n")
        f.write("\n")

        # Performance Metrics
        f.write("PERFORMANCE METRICS:\n")
        perf = report["performance_metrics"]
        f.write(f"  Current CPU Usage: {perf['current_cpu_percent']}%\n")
        f.write(f"  Current Memory Usage: {perf['current_memory_percent']}%\n")
        f.write(
            f"  Min Processing Time: {perf['processing_times_stats']['min_seconds']}s\n"
        )
        f.write(
            f"  Max Processing Time: {perf['processing_times_stats']['max_seconds']}s\n"
        )
        f.write(
            f"  Median Processing Time: {perf['processing_times_stats']['median_seconds']}s\n\n"
        )

        # Errors
        if report["errors"]:
            f.write("ERRORS (first 10):\n")
            for i, error in enumerate(report["errors"], 1):
                f.write(f"  {i}. {error}\n")
            f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        for i, rec in enumerate(report["recommendations"], 1):
            f.write(f"  {i}. {rec}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("End of Report\n")
        f.write("=" * 60 + "\n")


def generate_translation_report(
    input_file: str, output_file: str, dataset_type: str
) -> str:
    """Generate a comprehensive translation report."""
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

    # Get current system resources
    current_memory = psutil.virtual_memory()
    current_cpu = psutil.cpu_percent(interval=1)

    report = {
        "translation_summary": {
            "timestamp": datetime.now().isoformat(),
            "input_file": input_file,
            "output_file": output_file,
            "dataset_type": dataset_type,
            "total_time_seconds": round(total_time, 2),
            "total_time_formatted": format_duration(total_time),
            "total_examples": _translation_stats["total_examples"],
            "successful_translations": _translation_stats["successful_translations"],
            "failed_translations": _translation_stats["failed_translations"],
            "success_rate_percent": round(success_rate, 2),
            "average_processing_time_seconds": round(avg_processing_time, 2),
            "examples_per_minute": round(
                _translation_stats["total_examples"] / (total_time / 60), 2
            ),
        },
        "model_information": get_model_info(),
        "system_information": get_system_info(),
        "performance_metrics": {
            "current_cpu_percent": current_cpu,
            "current_memory_percent": current_memory.percent,
            "current_memory_available_gb": round(
                current_memory.available / (1024**3), 2
            ),
            "processing_times_stats": {
                "min_seconds": round(min(_translation_stats["processing_times"]), 2)
                if _translation_stats["processing_times"]
                else 0,
                "max_seconds": round(max(_translation_stats["processing_times"]), 2)
                if _translation_stats["processing_times"]
                else 0,
                "median_seconds": round(
                    sorted(_translation_stats["processing_times"])[
                        len(_translation_stats["processing_times"]) // 2
                    ],
                    2,
                )
                if _translation_stats["processing_times"]
                else 0,
            },
        },
        "errors": _translation_stats["errors"][:10]
        if _translation_stats["errors"]
        else [],  # Limit to first 10 errors
        "recommendations": generate_recommendations(
            success_rate, avg_processing_time, total_time
        ),
    }

    # Save report to file
    ensure_directory_exists("reports")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"reports/translation_report_{timestamp}.json"
    save_json_file(report, report_file)

    # Generate human-readable summary
    summary_file = f"reports/translation_summary_{timestamp}.txt"
    generate_summary_report(report, summary_file)

    _LOGGER.info(f"Translation report saved to: {report_file}")
    _LOGGER.info(f"Summary report saved to: {summary_file}")

    return report_file
