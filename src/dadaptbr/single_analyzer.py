"""
Single Evaluation Analyzer

Analyzes a single evaluation file and generates visualizations and reports.
"""

import json
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import (
    ensure_directory_exists,
    extract_pipeline_id,
    generate_report_filename,
    get_dataset_id,
)

_LOGGER = logging.getLogger(__name__)


class SingleEvaluationAnalyzer:
    """Analyzer for single evaluation files."""

    def __init__(self, evaluation_file: str):
        """Initialize analyzer with evaluation file."""
        self.evaluation_file = evaluation_file
        self.dataset_id = get_dataset_id(evaluation_file)
        self.pipeline_id = extract_pipeline_id(evaluation_file)

        # Load evaluation data
        self.data = self._load_evaluation_data()

        # Chart configuration
        self.chart_config = {
            "figsize": (12, 8),
            "dpi": 300,
            "style": "whitegrid",
            "palette": "husl",
        }

        _LOGGER.info(
            f"Analyzing single evaluation: {self.dataset_id} (pipeline: {self.pipeline_id})"
        )

    def _load_evaluation_data(self) -> list[dict[str, Any]]:
        """Load evaluation data from file."""
        try:
            with open(self.evaluation_file, encoding="utf-8") as f:
                data = json.load(f)
            _LOGGER.info(f"Loaded {len(data)} evaluation examples")
            return data
        except Exception as e:
            _LOGGER.error(f"Failed to load evaluation data: {e}")
            raise

    def _extract_scores(self) -> dict[str, list[float]]:
        """Extract scores from evaluation data."""
        scores = {"xcomet": [], "bleu": [], "chrf": [], "ter": []}

        for item in self.data:
            if "xcomet_score" in item and item["xcomet_score"] is not None:
                scores["xcomet"].append(item["xcomet_score"])
            if "bleu_score" in item and item["bleu_score"] is not None:
                scores["bleu"].append(item["bleu_score"])
            if "chrf_score" in item and item["chrf_score"] is not None:
                scores["chrf"].append(item["chrf_score"])
            if "ter_score" in item and item["ter_score"] is not None:
                scores["ter"].append(item["ter_score"])

        return scores

    def _calculate_statistics(
        self, scores: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """Calculate statistics for each metric."""
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

        return stats

    def create_visualizations(self) -> list[str]:
        """Create visualizations for the evaluation."""
        scores = self._extract_scores()
        stats = self._calculate_statistics(scores)

        # Set style
        plt.style.use("default")
        sns.set_palette(self.chart_config["palette"])

        created_files = []

        # 1. Score Distribution
        self._create_score_distribution(scores, stats)
        created_files.append("score_distribution")

        # 2. Quality Tiers
        self._create_quality_tiers(scores)
        created_files.append("quality_tiers")

        # 3. Score Correlation
        self._create_score_correlation(scores)
        created_files.append("score_correlation")

        return created_files

    def _create_score_distribution(
        self, scores: dict[str, list[float]], stats: dict[str, dict[str, float]]
    ):
        """Create score distribution visualization."""
        fig, axes = plt.subplots(2, 2, figsize=self.chart_config["figsize"])
        fig.suptitle(
            f"Score Distribution - {self.dataset_id.upper()}",
            fontsize=16,
            fontweight="bold",
        )

        metrics = ["xcomet", "bleu", "chrf", "ter"]
        titles = ["XCOMET", "BLEU", "chrF", "TER"]

        for i, (metric, title) in enumerate(zip(metrics, titles, strict=False)):
            ax = axes[i // 2, i % 2]

            if scores[metric]:
                ax.hist(scores[metric], bins=30, alpha=0.7, edgecolor="black")
                ax.axvline(
                    stats[metric]["mean"],
                    color="red",
                    linestyle="--",
                    label=f"Mean: {stats[metric]['mean']:.3f}",
                )
                ax.axvline(
                    stats[metric]["median"],
                    color="orange",
                    linestyle="--",
                    label=f"Median: {stats[metric]['median']:.3f}",
                )
                ax.legend()
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            ax.set_title(f"{title} Distribution")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to data/analysis/ (no dataset subfolder)
        output_path = os.path.join("data", "analysis", "score_distribution.png")
        ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.chart_config["dpi"], bbox_inches="tight")
        plt.close()

        _LOGGER.info(f"Score distribution saved to: {output_path}")

    def _create_quality_tiers(self, scores: dict[str, list[float]]):
        """Create quality tiers visualization."""
        if not scores["xcomet"]:
            _LOGGER.warning("No XCOMET scores available for quality tiers")
            return

        # Define quality tiers based on XCOMET scores
        tiers = {
            "Excellent (≥0.8)": [s for s in scores["xcomet"] if s >= 0.8],
            "Good (0.6-0.8)": [s for s in scores["xcomet"] if 0.6 <= s < 0.8],
            "Fair (0.4-0.6)": [s for s in scores["xcomet"] if 0.4 <= s < 0.6],
            "Poor (<0.4)": [s for s in scores["xcomet"] if s < 0.4],
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            f"Quality Tiers - {self.dataset_id.upper()}", fontsize=16, fontweight="bold"
        )

        # Bar chart
        tier_counts = [len(tier_scores) for tier_scores in tiers.values()]
        tier_labels = list(tiers.keys())

        bars = ax1.bar(
            tier_labels, tier_counts, color=["green", "lightgreen", "orange", "red"]
        )
        ax1.set_title("Distribution by Quality Tier")
        ax1.set_ylabel("Number of Examples")
        ax1.tick_params(axis="x", rotation=45)

        # Add count labels on bars
        for bar, count in zip(bars, tier_counts, strict=False):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
            )

        # Pie chart
        non_zero_counts = [count for count in tier_counts if count > 0]
        non_zero_labels = [
            label
            for label, count in zip(tier_labels, tier_counts, strict=False)
            if count > 0
        ]

        if non_zero_counts:
            ax2.pie(
                non_zero_counts,
                labels=non_zero_labels,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax2.set_title("Quality Distribution")

        plt.tight_layout()

        # Save to data/analysis/ (no dataset subfolder)
        output_path = os.path.join("data", "analysis", "quality_tiers.png")
        ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.chart_config["dpi"], bbox_inches="tight")
        plt.close()

        _LOGGER.info(f"Quality tiers saved to: {output_path}")

    def _create_score_correlation(self, scores: dict[str, list[float]]):
        """Create score correlation visualization."""
        # Create correlation matrix
        metrics = ["xcomet", "bleu", "chrf", "ter"]
        available_metrics = [m for m in metrics if scores[m]]

        if len(available_metrics) < 2:
            _LOGGER.warning("Not enough metrics for correlation analysis")
            return

        # Create correlation matrix
        corr_data = []
        for metric in available_metrics:
            corr_data.append(scores[metric])

        corr_matrix = np.corrcoef(corr_data)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            xticklabels=[m.upper() for m in available_metrics],
            yticklabels=[m.upper() for m in available_metrics],
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".3f",
        )

        ax.set_title(
            f"Score Correlation Matrix - {self.dataset_id.upper()}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save to data/analysis/ (no dataset subfolder)
        output_path = os.path.join("data", "analysis", "score_correlation.png")
        ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.chart_config["dpi"], bbox_inches="tight")
        plt.close()

        _LOGGER.info(f"Score correlation saved to: {output_path}")

    def generate_report(self) -> str:
        """Generate comprehensive analysis report using standardized format."""
        scores = self._extract_scores()
        stats = self._calculate_statistics(scores)

        # Calculate quality metrics
        quality_metrics = {}
        if scores["xcomet"]:
            excellent = len([s for s in scores["xcomet"] if s >= 0.8])
            good = len([s for s in scores["xcomet"] if 0.6 <= s < 0.8])
            fair = len([s for s in scores["xcomet"] if 0.4 <= s < 0.6])
            poor = len([s for s in scores["xcomet"] if s < 0.4])
            quality_metrics = {
                "excellent_count": excellent,
                "good_count": good,
                "fair_count": fair,
                "poor_count": poor,
                "excellent_percent": round(excellent / len(scores["xcomet"]) * 100, 1),
                "good_percent": round(good / len(scores["xcomet"]) * 100, 1),
                "fair_percent": round(fair / len(scores["xcomet"]) * 100, 1),
                "poor_percent": round(poor / len(scores["xcomet"]) * 100, 1),
            }

        # Generate standardized reports
        from .report_generator import generate_standard_reports

        report_file = generate_report_filename(
            self.dataset_id, "analysis", extension="json", pipeline_id=self.pipeline_id
        )
        summary_file = generate_report_filename(
            self.dataset_id, "analysis", extension="txt", pipeline_id=self.pipeline_id
        )

        generate_standard_reports(
            operation="analysis",
            input_file=self.evaluation_file,
            output_file="",  # Analysis doesn't have output file
            dataset_type="evaluation",
            model_name="single_analyzer",
            pipeline_id=self.pipeline_id,
            report_file=report_file,
            summary_file=summary_file,
            total_examples=len(self.data),
            models_analyzed=[self.dataset_id],
            visualizations_created=[
                "score_distribution",
                "quality_tiers",
                "score_correlation",
            ],
            quality_metrics=quality_metrics,
            score_statistics=stats,
        )

        _LOGGER.info(f"Analysis report saved to: {report_file}")
        return report_file


def main(evaluation_file: str):
    """Main function for single evaluation analysis."""
    try:
        analyzer = SingleEvaluationAnalyzer(evaluation_file)

        print(f"Iniciando Análise de Avaliação Única - {analyzer.dataset_id.upper()}")
        print("=" * 60)

        # Create visualizations
        print("Criando visualizações...")
        created_viz = analyzer.create_visualizations()

        # Generate report
        print("Gerando relatório...")
        report_file = analyzer.generate_report()

        print("=" * 60)
        print("ANÁLISE CONCLUÍDA")
        print("=" * 60)
        print(f"📊 Relatório: {report_file}")
        print("📈 Visualizações:")
        for viz in created_viz:
            print(f"  • {viz.replace('_', ' ').title()}: data/analysis/{viz}.png")

    except Exception as e:
        _LOGGER.error(f"Analysis failed: {e}")
        raise
