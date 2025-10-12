#!/usr/bin/env python3
"""
Agnostic Translation Analysis - Works with any models and datasets
"""

import os
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import generate_visualization_path, load_json_file


class AgnosticAnalyser:
    """Agnostic analyser that works with any models and datasets."""

    def __init__(self, dataset_id: str = "m_alert"):
        self.dataset_id = dataset_id
        self.model_data = {}  # {model_key: data}
        self.model_stats = {}  # {model_key: stats}
        self.model_cats = {}  # {model_key: category_stats}
        self.model_tiers = {}  # {model_key: tier_stats}
        self.merged_data = None
        self.selected_data = None
        self.comparison = None
        self.merged_comparison = None
        self.selection_stats = None
        self.correlation_data = None

        # Chart configuration
        self.chart_config = {
            "figsize": (10, 6),
            "dpi": 150,
            "colors": [
                "#2E86AB",
                "#A23B72",
                "#F18F01",
                "#C73E1D",
                "#6A994E",
                "#7209B7",
            ],
            "style": "whitegrid",
            "font_size": 14,
            "title_size": 16,
            "label_size": 12,
            "legend_size": 12,
        }

    def detect_available_models(self) -> dict:
        """Detect available models and their files for a dataset."""
        from .config.datasets import TRANSLATION_MODELS

        available_models = {}

        # Look for evaluated files in data/evaluated/{model_key}/ directory
        evaluated_dir = "data/evaluated"
        if not os.path.exists(evaluated_dir):
            return available_models

        # Check each model subdirectory
        for model_key, model_config in TRANSLATION_MODELS.items():
            model_dir = os.path.join(evaluated_dir, model_key)
            if os.path.exists(model_dir):
                # Look for evaluated files in this model directory
                for filename in os.listdir(model_dir):
                    if filename.endswith(".json") and "_" in filename:
                        available_models[model_key] = {
                            "evaluated_file": os.path.join(model_dir, filename),
                            "display_name": model_config["display_name"],
                        }
                        break

        return available_models

    def load_data(self):
        """Load all necessary data files dynamically."""
        print(f"Loading data files for dataset: {self.dataset_id}...")

        # Detect available models
        available_models = self.detect_available_models()

        if not available_models:
            print("No evaluated model files found!")
            return

        print(f"Found {len(available_models)} models: {list(available_models.keys())}")

        # Load evaluation data for each model
        self.model_data = {}
        for model_key, model_info in available_models.items():
            try:
                data = load_json_file(model_info["evaluated_file"])
                self.model_data[model_key] = sorted(data, key=lambda x: x.get("id", ""))
                print(f"Loaded {len(data)} {model_info['display_name']} evaluations")
            except Exception as e:
                print(f"Error loading {model_key}: {e}")

        # Load merged dataset (find the latest one)
        merged_files = [
            f
            for f in os.listdir("data")
            if f.startswith(f"{self.dataset_id}_merged_best_")
            or f.startswith(f"dadaptbr_{self.dataset_id.upper()}_train_merged_best_")
        ]
        if merged_files:
            latest_merged = sorted(merged_files)[-1]
            self.merged_data = load_json_file(f"data/{latest_merged}")
            print(f"Loaded merged dataset: {latest_merged}")
        else:
            self.merged_data = None
            print("No merged dataset found")

        # Load original data for categories (try to find any original file)
        original_files = [
            f
            for f in os.listdir("data")
            if f.endswith(".json")
            and not f.endswith("_evaluated.json")
            and not f.startswith("merged")
        ]

        if original_files:
            # Use the first available original file
            original_file = original_files[0]
            try:
                original_data = load_json_file(f"data/{original_file}")
                # Add categories to all model data
                id_to_category = {
                    item.get("id"): item.get("category", "unknown")
                    for item in original_data
                    if item.get("id")
                }

                for model_key, data in self.model_data.items():
                    for item in data:
                        item["category"] = id_to_category.get(item.get("id"), "unknown")

                if self.merged_data:
                    for item in self.merged_data:
                        item["category"] = id_to_category.get(item.get("id"), "unknown")

                print(f"Added categories from: {original_file}")
            except Exception as e:
                print(f"Error loading original data: {e}")

        # Load selected translations if available
        selected_files = [
            f
            for f in os.listdir("data")
            if f.startswith(f"{self.dataset_id}_selected_")
            or f.startswith(f"dadaptbr_{self.dataset_id.upper()}_train_selected_")
        ]
        if selected_files:
            selected_file = sorted(selected_files)[-1]
            self.selected_data = load_json_file(f"data/{selected_file}")
            print(f"Loaded selected translations: {selected_file}")
        else:
            self.selected_data = None

    def calculate_stats(self):
        """Calculate all statistics and store in instance variables."""
        if not self.model_data:
            print("No model data available for statistics calculation")
            return

        # Calculate statistics for each model
        self.model_stats = {}
        for model_key, data in self.model_data.items():
            scores = [item["score"] for item in data]

            self.model_stats[model_key] = {
                "count": len(data),
                "mean_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "std_score": statistics.stdev(scores) if len(scores) > 1 else 0,
                "error_rate": sum(1 for item in data if item.get("error_spans"))
                / len(data)
                * 100,
                "excellent_rate": sum(1 for s in scores if s >= 0.95)
                / len(scores)
                * 100,
                "poor_rate": sum(1 for s in scores if s < 0.60) / len(scores) * 100,
            }

        # Merged dataset statistics
        if self.merged_data:
            merged_scores = [item["score"] for item in self.merged_data]
            self.merged_stats = {
                "count": len(self.merged_data),
                "mean_score": statistics.mean(merged_scores),
                "median_score": statistics.median(merged_scores),
                "std_score": statistics.stdev(merged_scores)
                if len(merged_scores) > 1
                else 0,
                "error_rate": sum(
                    1 for item in self.merged_data if item.get("error_spans")
                )
                / len(self.merged_data)
                * 100,
                "excellent_rate": sum(1 for s in merged_scores if s >= 0.95)
                / len(merged_scores)
                * 100,
                "poor_rate": sum(1 for s in merged_scores if s < 0.60)
                / len(merged_scores)
                * 100,
            }

            # Add model selection counts dynamically
            for model_key in self.model_data.keys():
                selected_count = sum(
                    1
                    for item in self.merged_data
                    if item.get("selected_model") == model_key
                )
                self.merged_stats[f"{model_key}_selected"] = selected_count
        else:
            self.merged_stats = None

        # Model comparison - work with first two models if available
        model_keys = list(self.model_data.keys())
        if len(model_keys) >= 2:
            model1_key, model2_key = model_keys[0], model_keys[1]
            model1_scores = [item["score"] for item in self.model_data[model1_key]]
            model2_scores = [item["score"] for item in self.model_data[model2_key]]

            model1_better = sum(
                1
                for m1, m2 in zip(model1_scores, model2_scores, strict=False)
                if m1 > m2
            )
            model2_better = sum(
                1
                for m1, m2 in zip(model1_scores, model2_scores, strict=False)
                if m2 > m1
            )
            ties = len(model1_scores) - model1_better - model2_better

            # Analyze ties in detail
            tie_scores = []
            for m1, m2 in zip(model1_scores, model2_scores, strict=False):
                if m1 == m2:  # Exact tie
                    tie_scores.append((m1, m2))

            self.comparison = {
                "model1_key": model1_key,
                "model2_key": model2_key,
                "model1_better": model1_better,
                "model2_better": model2_better,
                "ties": ties,
                "tie_scores": tie_scores,
                "model1_win_rate": model1_better / len(model1_scores) * 100,
                "model2_win_rate": model2_better / len(model1_scores) * 100,
                "tie_rate": ties / len(model1_scores) * 100,
                "mean_difference": statistics.mean(
                    [
                        m1 - m2
                        for m1, m2 in zip(model1_scores, model2_scores, strict=False)
                    ]
                ),
            }
        else:
            self.comparison = None

        # Selection results
        if self.selected_data:
            self.selection_stats = {
                "total": len(self.selected_data),
                "above_threshold": sum(
                    1 for item in self.selected_data if item.get("score", 0) >= 0.5
                ),
                "below_threshold": sum(
                    1 for item in self.selected_data if item.get("score", 0) < 0.5
                ),
            }

            # Add model selection counts dynamically
            for model_key in self.model_data.keys():
                selected_count = sum(
                    1
                    for item in self.selected_data
                    if item.get("selected_model") == model_key
                )
                self.selection_stats[f"{model_key}_selected"] = selected_count
        else:
            self.selection_stats = None

        # Category analysis for each model
        self.model_cats = {}
        for model_key, data in self.model_data.items():
            self.model_cats[model_key] = self.analyze_by_category(data)

        self.merged_cats = (
            self.analyze_by_category(self.merged_data) if self.merged_data else None
        )

        # Quality tiers for each model
        self.model_tiers = {}
        for model_key, data in self.model_data.items():
            scores = [item["score"] for item in data]
            self.model_tiers[model_key] = self.analyze_quality_tiers(scores)

        self.merged_tiers = (
            self.analyze_quality_tiers(merged_scores) if self.merged_data else None
        )

        # Correlation
        self.correlation_data = self.analyze_score_correlation()

        # Merged comparison
        if self.merged_data and self.model_stats:
            individual_scores = [
                stats["mean_score"] for stats in self.model_stats.values()
            ]
            best_individual = max(individual_scores)

            self.merged_comparison = {
                "best_individual": best_individual,
                "merged_vs_best": self.merged_stats["mean_score"] - best_individual,
            }

            # Add individual model improvements
            for model_key, stats in self.model_stats.items():
                self.merged_comparison[f"vs_{model_key}_improvement"] = (
                    self.merged_stats["mean_score"] - stats["mean_score"]
                )

    def analyze_by_category(self, data):
        """Analyze performance by category."""
        if not data:
            return {}

        category_stats = defaultdict(list)
        for item in data:
            category_stats[item.get("category", "unknown")].append(item["score"])

        results = {}
        for category, scores in category_stats.items():
            if scores:
                results[category] = {
                    "count": len(scores),
                    "mean_score": statistics.mean(scores),
                    "median_score": statistics.median(scores),
                    "std_score": statistics.stdev(scores) if len(scores) > 1 else 0,
                }
        return results

    def analyze_quality_tiers(self, scores):
        """Analyze performance across quality tiers."""
        if not scores:
            return {}

        tiers = {
            "Excelente (≥0.95)": sum(1 for s in scores if s >= 0.95),
            "Bom (0.80-0.94)": sum(1 for s in scores if 0.80 <= s < 0.95),
            "Regular (0.60-0.79)": sum(1 for s in scores if 0.60 <= s < 0.80),
            "Ruim (<0.60)": sum(1 for s in scores if s < 0.60),
        }

        total = len(scores)
        return {tier: (count, count / total * 100) for tier, count in tiers.items()}

    def analyze_score_correlation(self):
        """Analyze correlation between model scores."""
        if len(self.model_data) < 2:
            return None

        # Get first two models for correlation
        model_keys = list(self.model_data.keys())
        model1_key, model2_key = model_keys[0], model_keys[1]

        model1_scores_dict = {
            item["id"]: item["score"] for item in self.model_data[model1_key]
        }
        model2_scores_dict = {
            item["id"]: item["score"] for item in self.model_data[model2_key]
        }

        common_ids = set(model1_scores_dict.keys()) & set(model2_scores_dict.keys())

        if len(common_ids) < 2:
            return None

        model1_matched = [model1_scores_dict[id] for id in common_ids]
        model2_matched = [model2_scores_dict[id] for id in common_ids]

        correlation = np.corrcoef(model1_matched, model2_matched)[0, 1]

        return {
            "model1_key": model1_key,
            "model2_key": model2_key,
            "correlation": correlation,
            "model1_scores": model1_matched,
            "model2_scores": model2_matched,
            "common_count": len(common_ids),
        }

    def get_model_display_name(self, model_key: str) -> str:
        """Get display name for a model key."""
        from .config.datasets import TRANSLATION_MODELS

        return TRANSLATION_MODELS.get(model_key, {}).get("display_name", model_key)

    def create_histogram_chart(self):
        """Create histogram chart for score distribution."""
        plt.figure(figsize=self.chart_config["figsize"])

        if not self.model_data:
            plt.text(
                0.5,
                0.5,
                "No model data available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            return

        # Plot histogram for each model
        model_keys = list(self.model_data.keys())
        for i, model_key in enumerate(model_keys):
            scores = [item["score"] for item in self.model_data[model_key]]
            display_name = self.get_model_display_name(model_key)

            plt.hist(
                scores,
                bins=40,
                alpha=0.7,
                label=display_name,
                color=self.chart_config["colors"][i % len(self.chart_config["colors"])],
                density=True,
                edgecolor="white",
                linewidth=0.5,
            )

            # Add mean line
            plt.axvline(
                statistics.mean(scores),
                color=self.chart_config["colors"][i % len(self.chart_config["colors"])],
                linestyle="--",
                alpha=0.9,
                linewidth=3,
            )

        plt.xlabel("Pontuação de Qualidade XCOMET")
        plt.ylabel("Densidade")

        # Dynamic title based on available models
        model_names = [self.get_model_display_name(key) for key in model_keys]
        title = f"Distribuição de Pontuações de Qualidade de Tradução\n{' vs '.join(model_names)}"
        plt.title(title, fontweight="bold")

        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            generate_visualization_path(self.dataset_id, "score_distribution"),
            dpi=self.chart_config["dpi"],
            bbox_inches="tight",
        )
        plt.close()

    def create_visualizations(self):
        """Create all visualizations."""
        print("Creating visualizations...")

        # Set up matplotlib style
        sns.set_style(self.chart_config["style"])
        plt.rcParams.update(
            {
                "font.size": self.chart_config["font_size"],
                "axes.titlesize": self.chart_config["title_size"],
                "axes.labelsize": self.chart_config["label_size"],
                "xtick.labelsize": self.chart_config["label_size"],
                "ytick.labelsize": self.chart_config["label_size"],
                "legend.fontsize": self.chart_config["legend_size"],
                "figure.titlesize": self.chart_config["title_size"],
            }
        )

        # Create charts
        self.create_histogram_chart()

        print("Visualizations saved to:")
        print(
            f"  📊 Score Distribution: data/analysis/{self.dataset_id}/score_distribution.png"
        )

    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("Generating comprehensive report...")

        report = []
        report.append("=" * 80)
        report.append(
            f"RELATÓRIO UNIFICADO DE ANÁLISE DE TRADUÇÃO {self.dataset_id.upper()}"
        )
        report.append("=" * 80)
        report.append("")

        # Basic Statistics for each model
        for model_key, stats in self.model_stats.items():
            display_name = self.get_model_display_name(model_key)
            report.append(f"{display_name}:")
            report.append(f"  Contagem: {stats['count']:,}")
            report.append(f"  Pontuação Média: {stats['mean_score']:.4f}")
            report.append(f"  Pontuação Mediana: {stats['median_score']:.4f}")
            report.append(f"  Desvio Padrão: {stats['std_score']:.4f}")
            report.append(f"  Taxa Excelente (≥0.95): {stats['excellent_rate']:.1f}%")
            report.append(f"  Taxa Ruim (<0.60): {stats['poor_rate']:.1f}%")
            report.append(f"  Taxa de Erro: {stats['error_rate']:.1f}%")
            report.append("")

        # Model Comparison
        if self.comparison:
            report.append("COMPARAÇÃO DE MODELOS:")
            report.append("-" * 40)
            model1_name = self.get_model_display_name(self.comparison["model1_key"])
            model2_name = self.get_model_display_name(self.comparison["model2_key"])

            report.append(
                f"{model1_name} melhor: {self.comparison['model1_better']:,} ({self.comparison['model1_win_rate']:.1f}%)"
            )
            report.append(
                f"{model2_name} melhor: {self.comparison['model2_better']:,} ({self.comparison['model2_win_rate']:.1f}%)"
            )
            report.append(
                f"Empates: {self.comparison['ties']:,} ({self.comparison['tie_rate']:.1f}%)"
            )
            report.append(
                f"Diferença média ({model1_name} - {model2_name}): {self.comparison['mean_difference']:.4f}"
            )
            report.append("")

        # Winner determination
        if self.comparison:
            winner_key = (
                self.comparison["model1_key"]
                if self.comparison["mean_difference"] > 0
                else self.comparison["model2_key"]
            )
            winner_name = self.get_model_display_name(winner_key)
            advantage = abs(self.comparison["mean_difference"])
            report.append(f"VENCEDOR GERAL: {winner_name}")
            report.append(f"Vantagem: {advantage:.4f} pontos")
            report.append("")

        report.append("=" * 80)
        report.append("Fim do Relatório")
        report.append("=" * 80)

        # Save report
        report_text = "\n".join(report)
        from .utils import generate_report_filename

        report_file = generate_report_filename(
            self.dataset_id, "analysis", extension="txt"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"Report saved to: {report_file}")

    def print_summary(self):
        """Print analysis summary."""
        print("\n" + "=" * 60)
        print("ANÁLISE CONCLUÍDA")
        print("=" * 60)
        print("Arquivos gerados:")

        from .utils import generate_report_filename

        report_file = generate_report_filename(
            self.dataset_id, "analysis", extension="txt"
        )
        print(f"📊 Relatório: {report_file}")
        print("📈 Visualizações:")
        print(
            f"  • Distribuição de Pontuações: data/analysis/{self.dataset_id}/score_distribution.png"
        )

        print("\nPrincipais descobertas:")

        if self.comparison:
            winner_key = (
                self.comparison["model1_key"]
                if self.comparison["mean_difference"] > 0
                else self.comparison["model2_key"]
            )
            winner_name = self.get_model_display_name(winner_key)
            advantage = abs(self.comparison["mean_difference"])

            print(
                f"  • {winner_name} vence no geral (vantagem de {advantage:.4f} pontos)"
            )

            for model_key, stats in self.model_stats.items():
                display_name = self.get_model_display_name(model_key)
                print(f"  • {display_name}: pontuação média {stats['mean_score']:.4f}")

            if self.correlation_data:
                print(
                    f"  • Correlação entre modelos: {self.correlation_data['correlation']:.3f}"
                )

        print("=" * 60)

    def run_analysis(self):
        """Run complete analysis."""
        print(f"Iniciando Análise Unificada de Tradução {self.dataset_id.upper()}")
        print("=" * 60)

        self.load_data()
        self.calculate_stats()
        self.create_visualizations()
        self.generate_report()
        self.print_summary()


def main(dataset_id: str = None):
    """Main analysis function."""
    if dataset_id is None:
        dataset_id = "m_alert"  # Default for current data

    analyser = AgnosticAnalyser(dataset_id)
    analyser.run_analysis()


if __name__ == "__main__":
    main()
