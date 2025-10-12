"""
Comparison Analyzer

Compares multiple evaluation files and generates comparative visualizations and reports.
"""

import json
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config.datasets import TRANSLATION_MODELS
from .utils import (
    ensure_directory_exists,
    extract_pipeline_id,
    generate_report_filename,
    get_dataset_id,
)

_LOGGER = logging.getLogger(__name__)


class ComparisonAnalyzer:
    """Analyzer for comparing multiple evaluation files."""

    def __init__(self, evaluation_files: list[str]):
        """Initialize analyzer with multiple evaluation files."""
        self.evaluation_files = evaluation_files
        self.dataset_id = get_dataset_id(evaluation_files[0])  # Assume all same dataset
        self.pipeline_ids = [extract_pipeline_id(f) for f in evaluation_files]

        # Load all evaluation data
        self.models_data = self._load_all_evaluations()

        # Chart configuration
        self.chart_config = {
            "figsize": (12, 8),
            "dpi": 300,
            "style": "whitegrid",
            "palette": "husl",
        }

        _LOGGER.info(
            f"Analyzing {len(evaluation_files)} evaluations for dataset: {self.dataset_id}"
        )

    def _load_all_evaluations(self) -> dict[str, list[dict[str, Any]]]:
        """Load all evaluation data from files."""
        models_data = {}

        for eval_file in self.evaluation_files:
            # Extract model key from file path
            model_key = self._extract_model_key(eval_file)

            try:
                with open(eval_file, encoding="utf-8") as f:
                    data = json.load(f)
                models_data[model_key] = data
                _LOGGER.info(f"Loaded {len(data)} examples for model: {model_key}")
            except Exception as e:
                _LOGGER.error(f"Failed to load evaluation data from {eval_file}: {e}")
                raise

        return models_data

    def _extract_model_key(self, file_path: str) -> str:
        """Extract model key from file path."""
        # Extract from path like data/evaluated/gemma3/m_alert_20251011_195938.json
        path_parts = file_path.split(os.sep)
        for part in path_parts:
            if part in TRANSLATION_MODELS.keys():
                return part
        return "unknown"

    def _extract_scores_by_model(self) -> dict[str, dict[str, list[float]]]:
        """Extract scores for each model."""
        all_scores = {}

        for model_key, data in self.models_data.items():
            scores = {"xcomet": [], "bleu": [], "chrf": [], "ter": []}

            for item in data:
                if "xcomet_score" in item and item["xcomet_score"] is not None:
                    scores["xcomet"].append(item["xcomet_score"])
                if "bleu_score" in item and item["bleu_score"] is not None:
                    scores["bleu"].append(item["bleu_score"])
                if "chrf_score" in item and item["chrf_score"] is not None:
                    scores["chrf"].append(item["chrf_score"])
                if "ter_score" in item and item["ter_score"] is not None:
                    scores["ter"].append(item["ter_score"])

            all_scores[model_key] = scores

        return all_scores

    def _calculate_statistics_by_model(
        self, scores_by_model: dict[str, dict[str, list[float]]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Calculate statistics for each model and metric."""
        stats_by_model = {}

        for model_key, scores in scores_by_model.items():
            stats = {}
            for metric, values in scores.items():
                if values:
                    stats[metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values),
                        "count": len(values),
                    }
                else:
                    stats[metric] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "median": 0.0,
                        "count": 0,
                    }
            stats_by_model[model_key] = stats

        return stats_by_model

    def create_comparison_visualizations(self) -> list[str]:
        """Create comparison visualizations."""
        scores_by_model = self._extract_scores_by_model()
        stats_by_model = self._calculate_statistics_by_model(scores_by_model)

        # Set style
        plt.style.use("default")
        sns.set_palette(self.chart_config["palette"])

        created_files = []

        # 1. Performance Comparison
        self._create_performance_comparison(stats_by_model)
        created_files.append("performance_comparison")

        # 2. Score Distribution Comparison
        self._create_distribution_comparison(scores_by_model)
        created_files.append("distribution_comparison")

        # 3. Quality Comparison
        self._create_quality_comparison(scores_by_model)
        created_files.append("quality_comparison")

        return created_files

    def _create_performance_comparison(
        self, stats_by_model: dict[str, dict[str, dict[str, float]]]
    ):
        """Create performance comparison bar chart."""
        metrics = ["xcomet", "bleu", "chrf", "ter"]
        metric_names = ["XCOMET", "BLEU", "chrF", "TER"]
        models = list(stats_by_model.keys())

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Performance Comparison - {self.dataset_id.upper()}",
            fontsize=16,
            fontweight="bold",
        )

        for i, (metric, metric_name) in enumerate(
            zip(metrics, metric_names, strict=False)
        ):
            ax = axes[i // 2, i % 2]

            means = [stats_by_model[model][metric]["mean"] for model in models]
            stds = [stats_by_model[model][metric]["std"] for model in models]

            bars = ax.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(f"{metric_name} Comparison")
            ax.set_ylabel("Score")
            ax.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.01,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                )

            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to data/analysis/ (no dataset subfolder)
        output_path = os.path.join("data", "analysis", "performance_comparison.png")
        ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.chart_config["dpi"], bbox_inches="tight")
        plt.close()

        _LOGGER.info(f"Performance comparison saved to: {output_path}")

    def _create_distribution_comparison(
        self, scores_by_model: dict[str, dict[str, list[float]]]
    ):
        """Create score distribution comparison."""
        if "xcomet" not in scores_by_model[list(scores_by_model.keys())[0]]:
            _LOGGER.warning("No XCOMET scores available for distribution comparison")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        for model_key, scores in scores_by_model.items():
            if scores["xcomet"]:
                ax.hist(
                    scores["xcomet"], bins=30, alpha=0.6, label=model_key, density=True
                )

        ax.set_title(
            f"XCOMET Score Distribution Comparison - {self.dataset_id.upper()}",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("XCOMET Score")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to data/analysis/ (no dataset subfolder)
        output_path = os.path.join("data", "analysis", "distribution_comparison.png")
        ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.chart_config["dpi"], bbox_inches="tight")
        plt.close()

        _LOGGER.info(f"Distribution comparison saved to: {output_path}")

    def _create_quality_comparison(
        self, scores_by_model: dict[str, dict[str, list[float]]]
    ):
        """Create quality tiers comparison."""
        if "xcomet" not in scores_by_model[list(scores_by_model.keys())[0]]:
            _LOGGER.warning("No XCOMET scores available for quality comparison")
            return

        # Define quality tiers
        tiers = ["Excellent (≥0.8)", "Good (0.6-0.8)", "Fair (0.4-0.6)", "Poor (<0.4)"]
        tier_ranges = [(0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.0, 0.4)]

        fig, ax = plt.subplots(figsize=(12, 8))

        model_tier_counts = {}
        for model_key, scores in scores_by_model.items():
            if scores["xcomet"]:
                tier_counts = []
                for min_score, max_score in tier_ranges:
                    count = len(
                        [s for s in scores["xcomet"] if min_score <= s < max_score]
                    )
                    tier_counts.append(count)
                model_tier_counts[model_key] = tier_counts

        # Create stacked bar chart
        bottom = np.zeros(len(tiers))
        colors = ["green", "lightgreen", "orange", "red"]

        for i, (model_key, tier_counts) in enumerate(model_tier_counts.items()):
            ax.bar(
                tiers,
                tier_counts,
                bottom=bottom,
                label=model_key,
                color=colors[i % len(colors)],
                alpha=0.7,
            )
            bottom += tier_counts

        ax.set_title(
            f"Quality Tiers Comparison - {self.dataset_id.upper()}",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_ylabel("Number of Examples")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to data/analysis/ (no dataset subfolder)
        output_path = os.path.join("data", "analysis", "quality_comparison.png")
        ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.chart_config["dpi"], bbox_inches="tight")
        plt.close()

        _LOGGER.info(f"Quality comparison saved to: {output_path}")

    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report using standardized format."""
        scores_by_model = self._extract_scores_by_model()
        stats_by_model = self._calculate_statistics_by_model(scores_by_model)

        # Calculate quality metrics for each model
        quality_metrics = {}
        for model_key, scores in scores_by_model.items():
            if scores["xcomet"]:
                excellent = len([s for s in scores["xcomet"] if s >= 0.8])
                good = len([s for s in scores["xcomet"] if 0.6 <= s < 0.8])
                fair = len([s for s in scores["xcomet"] if 0.4 <= s < 0.6])
                poor = len([s for s in scores["xcomet"] if s < 0.4])
                quality_metrics[model_key] = {
                    "excellent_count": excellent,
                    "good_count": good,
                    "fair_count": fair,
                    "poor_count": poor,
                }

        # Generate standardized reports
        from .report_generator import generate_standard_reports

        report_file = generate_report_filename(
            self.dataset_id, "comparison", extension="json"
        )
        summary_file = generate_report_filename(
            self.dataset_id, "comparison", extension="txt"
        )

        generate_standard_reports(
            operation="analysis",  # Use analysis for comparison
            input_file=", ".join(self.evaluation_files),
            output_file="",  # Comparison doesn't have output file
            dataset_type="comparison",
            model_name="comparison_analyzer",
            pipeline_id=None,  # Comparison uses current timestamp
            report_file=report_file,
            summary_file=summary_file,
            total_examples=sum(len(data) for data in self.models_data.values()),
            models_analyzed=list(stats_by_model.keys()),
            visualizations_created=[
                "performance_comparison",
                "distribution_comparison",
                "quality_comparison",
            ],
            quality_metrics=quality_metrics,
            score_statistics=stats_by_model,
            pipeline_ids=self.pipeline_ids,
        )

        _LOGGER.info(f"Comparison report saved to: {report_file}")
        return report_file


def main(evaluation_files: list[str]):
    """Main function for comparison analysis."""
    try:
        analyzer = ComparisonAnalyzer(evaluation_files)

        print(f"Iniciando Análise Comparativa - {analyzer.dataset_id.upper()}")
        print("=" * 60)
        print(f"Modelos: {', '.join(analyzer.models_data.keys())}")

        # Create comparison visualizations
        print("Criando visualizações comparativas...")
        created_viz = analyzer.create_comparison_visualizations()

        # Generate comparison report
        print("Gerando relatório comparativo...")
        report_file = analyzer.generate_comparison_report()

        print("=" * 60)
        print("ANÁLISE COMPARATIVA CONCLUÍDA")
        print("=" * 60)
        print(f"📊 Relatório: {report_file}")
        print("📈 Visualizações:")
        for viz in created_viz:
            print(f"  • {viz.replace('_', ' ').title()}: data/analysis/{viz}.png")

    except Exception as e:
        _LOGGER.error(f"Comparison analysis failed: {e}")
        raise
