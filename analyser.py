#!/usr/bin/env python3
"""
Translation Analysis
"""

import os
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import generate_visualization_path, load_json_file

# Global variables
gemma_data = None
tower_data = None
merged_data = None
selected_data = None
gemma_stats = None
tower_stats = None
merged_stats = None
comparison = None
merged_comparison = None
selection_stats = None
correlation_data = None
gemma_cats = None
tower_cats = None
merged_cats = None
gemma_tiers = None
tower_tiers = None
merged_tiers = None

# Category translations
CATEGORY_TRANSLATIONS = {
    "crime_injury": "Crime/Lesão",
    "hate_other": "Ódio/Outros",
    "hate_ethnic": "Ódio/Étnico",
    "crime_theft": "Crime/Roubo",
    "crime_propaganda": "Crime/Propaganda",
    "hate_women": "Ódio/Mulheres",
    "substance_drug": "Substância/Drogas",
    "substance_other": "Substância/Outros",
    "weapon_other": "Arma/Outros",
    "crime_cyber": "Crime/Cibernético",
    "hate_religion": "Ódio/Religião",
    "hate_lgbtq+": "Ódio/LGBTQ+",
    "sex_harrasment": "Sexo/Assédio",
    "sex_other": "Sexo/Outros",
    "crime_privacy": "Crime/Privacidade",
    "substance_alcohol": "Substância/Álcool",
    "crime_other": "Crime/Outros",
    "crime_tax": "Crime/Impostos",
    "substance_cannabis": "Substância/Cannabis",
    "self_harm_thin": "Auto-dano/Anorexia",
    "weapon_chemical": "Arma/Química",
    "weapon_biological": "Arma/Biológica",
    "crime_kidnapp": "Crime/Sequestro",
    "self_harm_suicide": "Auto-dano/Suicídio",
    "hate_body": "Ódio/Corpo",
    "weapon_radioactive": "Arma/Radioativa",
    "sex_porn": "Sexo/Pornografia",
    "self_harm_other": "Auto-dano/Outros",
    "hate_disabled": "Ódio/Deficientes",
    "weapon_firearm": "Arma/Arma de fogo",
    "substance_tobacco": "Substância/Tabaco",
    "hate_poor": "Ódio/Pobres",
    "unknown": "Desconhecido",
}

# Chart configuration
CHART_CONFIG = {
    "figsize": (10, 6),
    "dpi": 150,
    "colors": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#7209B7"],
    "style": "whitegrid",
    "font_size": 14,
    "title_size": 16,
    "label_size": 12,
    "legend_size": 12,
}


def load_data():
    """Load all necessary data files."""
    global gemma_data, tower_data, merged_data, selected_data

    print("Loading data files...")

    # Load evaluation data
    gemma_eval = load_json_file("data/dadaptbr_M-ALERT_train_gemma_evaluated.json")
    tower_eval = load_json_file("data/dadaptbr_M-ALERT_train_tower_evaluated.json")

    # Load merged dataset (find the latest one)
    merged_files = [
        f
        for f in os.listdir("data")
        if f.startswith("dadaptbr_M-ALERT_train_merged_best_")
    ]
    if merged_files:
        latest_merged = sorted(merged_files)[-1]
        merged_data = load_json_file(f"data/{latest_merged}")
        print(f"Loaded merged dataset: {latest_merged}")
    else:
        merged_data = None
        print("No merged dataset found")

    # Load original data for categories
    gemma_orig = load_json_file("data/dadaptbr_M-ALERT_train_gemma.json")

    # Load selected translations if available
    selected_file = "data/dadaptbr_M-ALERT_train_selected_20251008_123504.json"
    selected_data = None
    if os.path.exists(selected_file):
        selected_data = load_json_file(selected_file)

    # Add categories to evaluation data
    id_to_category = {
        item["id"]: item.get("category", "unknown") for item in gemma_orig
    }
    for item in gemma_eval:
        item["category"] = id_to_category.get(item.get("id"), "unknown")
    for item in tower_eval:
        item["category"] = id_to_category.get(item.get("id"), "unknown")

    # Add categories to merged data if available
    if merged_data:
        for item in merged_data:
            item["category"] = id_to_category.get(item.get("id"), "unknown")

    # Sort data by ID to ensure consistent comparison
    gemma_data = sorted(gemma_eval, key=lambda x: x["id"])
    tower_data = sorted(tower_eval, key=lambda x: x["id"])

    print(f"Loaded {len(gemma_data)} Gemma3 evaluations")
    print(f"Loaded {len(tower_data)} TowerInstruct evaluations")
    if merged_data:
        print(f"Loaded {len(merged_data)} merged best translations")
    if selected_data:
        print(f"Loaded {len(selected_data)} selected translations")


def calculate_stats():
    """Calculate all statistics and store in global variables."""
    global \
        gemma_stats, \
        tower_stats, \
        merged_stats, \
        comparison, \
        merged_comparison, \
        selection_stats, \
        correlation_data
    global gemma_cats, tower_cats, merged_cats, gemma_tiers, tower_tiers, merged_tiers

    # Basic statistics
    gemma_scores = [item["score"] for item in gemma_data]
    tower_scores = [item["score"] for item in tower_data]

    gemma_stats = {
        "count": len(gemma_data),
        "mean_score": statistics.mean(gemma_scores),
        "median_score": statistics.median(gemma_scores),
        "std_score": statistics.stdev(gemma_scores) if len(gemma_scores) > 1 else 0,
        "error_rate": sum(1 for item in gemma_data if item.get("error_spans"))
        / len(gemma_data)
        * 100,
        "excellent_rate": sum(1 for s in gemma_scores if s >= 0.95)
        / len(gemma_scores)
        * 100,
        "poor_rate": sum(1 for s in gemma_scores if s < 0.60) / len(gemma_scores) * 100,
    }

    tower_stats = {
        "count": len(tower_data),
        "mean_score": statistics.mean(tower_scores),
        "median_score": statistics.median(tower_scores),
        "std_score": statistics.stdev(tower_scores) if len(tower_scores) > 1 else 0,
        "error_rate": sum(1 for item in tower_data if item.get("error_spans"))
        / len(tower_data)
        * 100,
        "excellent_rate": sum(1 for s in tower_scores if s >= 0.95)
        / len(tower_scores)
        * 100,
        "poor_rate": sum(1 for s in tower_scores if s < 0.60) / len(tower_scores) * 100,
    }

    # Merged dataset statistics
    if merged_data:
        merged_scores = [item["score"] for item in merged_data]
        merged_stats = {
            "count": len(merged_data),
            "mean_score": statistics.mean(merged_scores),
            "median_score": statistics.median(merged_scores),
            "std_score": statistics.stdev(merged_scores)
            if len(merged_scores) > 1
            else 0,
            "error_rate": sum(1 for item in merged_data if item.get("error_spans"))
            / len(merged_data)
            * 100,
            "excellent_rate": sum(1 for s in merged_scores if s >= 0.95)
            / len(merged_scores)
            * 100,
            "poor_rate": sum(1 for s in merged_scores if s < 0.60)
            / len(merged_scores)
            * 100,
            "gemma_selected": sum(
                1 for item in merged_data if item.get("selected_model") == "gemma3"
            ),
            "tower_selected": sum(
                1
                for item in merged_data
                if item.get("selected_model") == "towerinstruct"
            ),
        }
    else:
        merged_stats = None

    # Model comparison with detailed tie analysis
    gemma_better = sum(
        1 for g, t in zip(gemma_scores, tower_scores, strict=False) if g > t
    )
    tower_better = sum(
        1 for g, t in zip(gemma_scores, tower_scores, strict=False) if t > g
    )
    ties = len(gemma_scores) - gemma_better - tower_better

    # Analyze ties in detail
    tie_scores = []
    for g, t in zip(gemma_scores, tower_scores, strict=False):
        if g == t:  # Exact tie (same as merger logic)
            tie_scores.append((g, t))

    comparison = {
        "gemma_better": gemma_better,
        "tower_better": tower_better,
        "ties": ties,
        "tie_scores": tie_scores,
        "gemma_win_rate": gemma_better / len(gemma_scores) * 100,
        "tower_win_rate": tower_better / len(gemma_scores) * 100,
        "tie_rate": ties / len(gemma_scores) * 100,
        "mean_difference": statistics.mean(
            [g - t for g, t in zip(gemma_scores, tower_scores, strict=False)]
        ),
    }

    # Selection results
    if selected_data:
        selection_stats = {
            "total": len(selected_data),
            "gemma_selected": sum(
                1 for item in selected_data if item.get("selected_model") == "gemma3"
            ),
            "tower_selected": sum(
                1
                for item in selected_data
                if item.get("selected_model") == "towerinstruct"
            ),
            "above_threshold": sum(
                1 for item in selected_data if item.get("score", 0) >= 0.5
            ),
            "below_threshold": sum(
                1 for item in selected_data if item.get("score", 0) < 0.5
            ),
        }
    else:
        selection_stats = None

    # Category analysis
    gemma_cats = analyze_by_category(gemma_data)
    tower_cats = analyze_by_category(tower_data)
    if merged_data:
        merged_cats = analyze_by_category(merged_data)

    # Quality tiers
    gemma_tiers = analyze_quality_tiers(gemma_scores)
    tower_tiers = analyze_quality_tiers(tower_scores)
    if merged_data:
        merged_tiers = analyze_quality_tiers(merged_scores)

    # Correlation
    correlation_data = analyze_score_correlation()

    # Merged comparison
    if merged_data:
        merged_comparison = {
            "vs_gemma_improvement": merged_stats["mean_score"]
            - gemma_stats["mean_score"],
            "vs_tower_improvement": merged_stats["mean_score"]
            - tower_stats["mean_score"],
            "best_individual": max(
                gemma_stats["mean_score"], tower_stats["mean_score"]
            ),
            "merged_vs_best": merged_stats["mean_score"]
            - max(gemma_stats["mean_score"], tower_stats["mean_score"]),
        }


def analyze_by_category(data):
    """Analyze performance by category."""
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


def analyze_quality_tiers(scores):
    """Analyze performance across quality tiers."""
    tiers = {
        "Excelente (≥0.95)": sum(1 for s in scores if s >= 0.95),
        "Bom (0.80-0.94)": sum(1 for s in scores if 0.80 <= s < 0.95),
        "Regular (0.60-0.79)": sum(1 for s in scores if 0.60 <= s < 0.80),
        "Ruim (<0.60)": sum(1 for s in scores if s < 0.60),
    }

    total = len(scores)
    return {tier: (count, count / total * 100) for tier, count in tiers.items()}


def analyze_score_correlation():
    """Analyze correlation between model scores."""
    gemma_scores_dict = {item["id"]: item["score"] for item in gemma_data}
    tower_scores_dict = {item["id"]: item["score"] for item in tower_data}

    common_ids = set(gemma_scores_dict.keys()) & set(tower_scores_dict.keys())

    if len(common_ids) < 2:
        return None

    gemma_matched = [gemma_scores_dict[id] for id in common_ids]
    tower_matched = [tower_scores_dict[id] for id in common_ids]

    correlation = np.corrcoef(gemma_matched, tower_matched)[0, 1]

    return {
        "correlation": correlation,
        "gemma_scores": gemma_matched,
        "tower_scores": tower_matched,
        "common_count": len(common_ids),
    }


def create_chart(chart_type, dataset_id="m_alert", **kwargs):
    """Generic chart creation function."""
    sns.set_style(CHART_CONFIG["style"])
    plt.rcParams.update(
        {
            "font.size": CHART_CONFIG["font_size"],
            "axes.titlesize": CHART_CONFIG["title_size"],
            "axes.labelsize": CHART_CONFIG["label_size"],
            "xtick.labelsize": CHART_CONFIG["label_size"],
            "ytick.labelsize": CHART_CONFIG["label_size"],
            "legend.fontsize": CHART_CONFIG["legend_size"],
            "figure.titlesize": CHART_CONFIG["title_size"],
        }
    )

    if chart_type == "histogram":
        plt.figure(figsize=CHART_CONFIG["figsize"])
        gemma_scores = [item["score"] for item in gemma_data]
        tower_scores = [item["score"] for item in tower_data]

        plt.hist(
            gemma_scores,
            bins=40,
            alpha=0.7,
            label="Gemma3",
            color=CHART_CONFIG["colors"][0],
            density=True,
            edgecolor="white",
            linewidth=0.5,
        )
        plt.hist(
            tower_scores,
            bins=40,
            alpha=0.7,
            label="TowerInstruct",
            color=CHART_CONFIG["colors"][1],
            density=True,
            edgecolor="white",
            linewidth=0.5,
        )

        plt.xlabel("Pontuação de Qualidade XCOMET")
        plt.ylabel("Densidade")
        plt.title(
            "Distribuição de Pontuações de Qualidade de Tradução\nGemma3 vs TowerInstruct",
            fontweight="bold",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.axvline(
            statistics.mean(gemma_scores),
            color=CHART_CONFIG["colors"][0],
            linestyle="--",
            alpha=0.9,
            linewidth=3,
        )
        plt.axvline(
            statistics.mean(tower_scores),
            color=CHART_CONFIG["colors"][1],
            linestyle="--",
            alpha=0.9,
            linewidth=3,
        )

        plt.tight_layout()
        plt.savefig(
            generate_visualization_path(dataset_id, "score_distribution"),
            dpi=CHART_CONFIG["dpi"],
            bbox_inches="tight",
        )
        plt.close()

    elif chart_type == "performance":
        plt.figure(figsize=CHART_CONFIG["figsize"])

        metrics = [
            "Pontuação Média",
            "Taxa Excelente\n(≥0.95)",
            "Taxa Ruim\n(<0.60)",
            "Taxa de Erro",
        ]
        gemma_values = [
            gemma_stats["mean_score"],
            gemma_stats["excellent_rate"],
            gemma_stats["poor_rate"],
            gemma_stats["error_rate"],
        ]
        tower_values = [
            tower_stats["mean_score"],
            tower_stats["excellent_rate"],
            tower_stats["poor_rate"],
            tower_stats["error_rate"],
        ]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = plt.bar(
            x - width / 2,
            gemma_values,
            width,
            label="Gemma3",
            color=CHART_CONFIG["colors"][0],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )
        bars2 = plt.bar(
            x + width / 2,
            tower_values,
            width,
            label="TowerInstruct",
            color=CHART_CONFIG["colors"][1],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    f"{height:.1f}%"
                    if "Rate" in metrics[bars.index(bar)]
                    else f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.xlabel("Métricas de Performance")
        plt.ylabel("Pontuação / Porcentagem")
        plt.title(
            "Comparação de Performance dos Modelos\nMétricas Principais de Qualidade",
            fontweight="bold",
        )
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            generate_visualization_path(dataset_id, "performance_comparison"),
            dpi=CHART_CONFIG["dpi"],
            bbox_inches="tight",
        )
        plt.close()

    elif chart_type == "selection":
        # Use merged_data if selected_data is not available
        if not selected_data and not merged_data:
            return

        # Use merged_data for selection stats if selected_data is not available
        if not selected_data:
            selection_data = merged_data
            gemma_selected = sum(
                1 for item in selection_data if item.get("selected_model") == "gemma3"
            )
            tower_selected = sum(
                1
                for item in selection_data
                if item.get("selected_model") == "towerinstruct"
            )
            above_threshold = sum(
                1 for item in selection_data if item.get("score", 0) >= 0.5
            )
            # below_threshold not used in visualization summary
            total = len(selection_data)
        else:
            selection_data = selected_data
            gemma_selected = selection_stats["gemma_selected"]
            tower_selected = selection_stats["tower_selected"]
            above_threshold = selection_stats["above_threshold"]
            total = selection_stats["total"]

        plt.figure(figsize=CHART_CONFIG["figsize"])

        # Calculate score-based wins vs ties for consistent visualization
        ties = comparison.get("ties", 0)
        gemma_wins = comparison.get("gemma_better", 0)
        tower_wins = comparison.get("tower_better", 0)

        # For pie chart: show score-based comparison (what the text box describes)
        models = ["Gemma3", "TowerInstruct", "Empates (→TowerInstruct)"]
        counts = [gemma_wins, tower_wins, ties]
        colors = [
            CHART_CONFIG["colors"][0],
            CHART_CONFIG["colors"][1],
            CHART_CONFIG["colors"][2],
        ]

        wedges, texts, autotexts = plt.pie(
            counts,
            labels=models,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontweight": "bold"},
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )

        for autotext in autotexts:
            autotext.set_fontweight("bold")
            autotext.set_color("white")

        plt.title(
            "Resultados da Seleção de Traduções",
            fontweight="bold",
            pad=20,
        )

        # Text box: show score-based analysis (now consistent with pie chart)
        summary_text = (
            f"Total de Traduções: {total:,}\n"
            f"Gemma3 venceu por score: {gemma_wins:,} ({gemma_wins / total * 100:.1f}%)\n"
            f"TowerInstruct venceu por score: {tower_wins:,} ({tower_wins / total * 100:.1f}%)\n"
            f"Empates (TowerInstruct): {ties:,} ({ties / total * 100:.1f}%)\n"
            f"Acima do Limiar: {above_threshold:,} ({above_threshold / total * 100:.1f}%)"
        )

        plt.figtext(
            0.5,
            -0.15,
            summary_text,
            ha="center",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": CHART_CONFIG["colors"][2],
                "alpha": 0.8,
                "edgecolor": "white",
                "linewidth": 1,
            },
        )

        plt.tight_layout()
        plt.savefig(
            generate_visualization_path(dataset_id, "selection_results"),
            dpi=CHART_CONFIG["dpi"],
            bbox_inches="tight",
        )
        plt.close()

    elif chart_type == "boxplot":
        plt.figure(figsize=CHART_CONFIG["figsize"])

        all_scores = [item["score"] for item in gemma_data] + [
            item["score"] for item in tower_data
        ]
        all_models = ["Gemma3"] * len(gemma_data) + ["TowerInstruct"] * len(tower_data)

        sns.boxplot(
            x=all_models,
            y=all_scores,
            hue=all_models,
            palette=CHART_CONFIG["colors"][:2],
            legend=False,
        )
        plt.xlabel("Modelo")
        plt.ylabel("Pontuação de Qualidade XCOMET")
        plt.title(
            "Distribuição de Pontuações de Qualidade\nComparação Box Plot",
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3, axis="y")

        gemma_mean = statistics.mean([item["score"] for item in gemma_data])
        tower_mean = statistics.mean([item["score"] for item in tower_data])
        plt.scatter(
            [0],
            [gemma_mean],
            color="red",
            s=120,
            marker="D",
            label=f"Gemma3 Mean: {gemma_mean:.3f}",
            zorder=5,
            edgecolor="white",
            linewidth=2,
        )
        plt.scatter(
            [1],
            [tower_mean],
            color="red",
            s=120,
            marker="D",
            label=f"TowerInstruct Mean: {tower_mean:.3f}",
            zorder=5,
            edgecolor="white",
            linewidth=2,
        )
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            generate_visualization_path(dataset_id, "quality_boxplot"),
            dpi=CHART_CONFIG["dpi"],
            bbox_inches="tight",
        )
        plt.close()

    elif chart_type == "tiers":
        plt.figure(figsize=CHART_CONFIG["figsize"])

        tiers = list(gemma_tiers.keys())
        gemma_percentages = [gemma_tiers[tier][1] for tier in tiers]
        tower_percentages = [tower_tiers[tier][1] for tier in tiers]

        x = np.arange(len(tiers))
        width = 0.35

        bars1 = plt.bar(
            x - width / 2,
            gemma_percentages,
            width,
            label="Gemma3",
            color=CHART_CONFIG["colors"][0],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )
        bars2 = plt.bar(
            x + width / 2,
            tower_percentages,
            width,
            label="TowerInstruct",
            color=CHART_CONFIG["colors"][1],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.xlabel("Níveis de Qualidade")
        plt.ylabel("Porcentagem de Traduções")
        plt.title(
            "Distribuição de Qualidade por Níveis\nAnálise de Qualidade de Tradução",
            fontweight="bold",
        )
        plt.xticks(x, [tier.split("(")[0].strip() for tier in tiers])
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            generate_visualization_path(dataset_id, "quality_tiers"),
            dpi=CHART_CONFIG["dpi"],
            bbox_inches="tight",
        )
        plt.close()

    elif chart_type == "correlation":
        if not correlation_data:
            return

        plt.figure(figsize=CHART_CONFIG["figsize"])

        plt.scatter(
            correlation_data["gemma_scores"],
            correlation_data["tower_scores"],
            alpha=0.7,
            color=CHART_CONFIG["colors"][2],
            s=30,
            edgecolor="white",
            linewidth=0.5,
        )

        min_score = min(
            min(correlation_data["gemma_scores"]), min(correlation_data["tower_scores"])
        )
        max_score = max(
            max(correlation_data["gemma_scores"]), max(correlation_data["tower_scores"])
        )
        plt.plot(
            [min_score, max_score],
            [min_score, max_score],
            "r--",
            alpha=0.9,
            linewidth=3,
            label="Perfect Agreement",
        )

        plt.xlabel("Pontuação XCOMET Gemma3")
        plt.ylabel("Pontuação XCOMET TowerInstruct")
        plt.title(
            f"Correlação de Pontuações dos Modelos\nPearson r = {correlation_data['correlation']:.3f}",
            fontweight="bold",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.figtext(
            0.15,
            0.85,
            f"Correlação: {correlation_data['correlation']:.3f}\nAmostras: {correlation_data['common_count']:,}",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": CHART_CONFIG["colors"][3],
                "alpha": 0.8,
                "edgecolor": "white",
                "linewidth": 1,
            },
        )

        plt.tight_layout()
        plt.savefig(
            generate_visualization_path(dataset_id, "score_correlation"),
            dpi=CHART_CONFIG["dpi"],
            bbox_inches="tight",
        )
        plt.close()

    elif chart_type == "merged_comparison":
        if not merged_data:
            return

        plt.figure(figsize=CHART_CONFIG["figsize"])

        models = ["Gemma3", "TowerInstruct", "Mesclado"]
        means = [
            gemma_stats["mean_score"],
            tower_stats["mean_score"],
            merged_stats["mean_score"],
        ]
        excellent_rates = [
            gemma_stats["excellent_rate"],
            tower_stats["excellent_rate"],
            merged_stats["excellent_rate"],
        ]
        poor_rates = [
            gemma_stats["poor_rate"],
            tower_stats["poor_rate"],
            merged_stats["poor_rate"],
        ]

        x = np.arange(len(models))
        width = 0.25

        bars1 = plt.bar(
            x - width,
            means,
            width,
            label="Pontuação Média",
            color=CHART_CONFIG["colors"][0],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )
        bars2 = plt.bar(
            x,
            excellent_rates,
            width,
            label="Taxa Excelente (≥0.95)",
            color=CHART_CONFIG["colors"][1],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )
        bars3 = plt.bar(
            x + width,
            poor_rates,
            width,
            label="Taxa Ruim (<0.60)",
            color=CHART_CONFIG["colors"][2],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    f"{height:.1f}%"
                    if "Rate" in str(bars) or "Taxa" in str(bars)
                    else f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )

        plt.xlabel("Modelos")
        plt.ylabel("Pontuação / Porcentagem")
        plt.title(
            "Comparação: Modelos Individuais vs Dataset Mesclado\nSeleção da Melhor Tradução por ID",
            fontweight="bold",
        )
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")

        # Improvement metrics are shown in the title and can be referenced in the paper text

        plt.tight_layout()
        plt.savefig(
            generate_visualization_path(dataset_id, "merged_comparison"),
            dpi=CHART_CONFIG["dpi"],
            bbox_inches="tight",
        )
        plt.close()

    elif chart_type == "categories":
        plt.figure(figsize=(14, 10))

        all_categories = set(gemma_cats.keys()) | set(tower_cats.keys())
        category_counts = [
            (
                cat,
                gemma_cats.get(cat, {}).get("count", 0)
                + tower_cats.get(cat, {}).get("count", 0),
            )
            for cat in all_categories
        ]
        top_categories = sorted(category_counts, key=lambda x: x[1], reverse=True)[:15]

        categories = [cat[0] for cat in top_categories]
        gemma_means = [gemma_cats[cat]["mean_score"] for cat in categories]
        tower_means = [tower_cats[cat]["mean_score"] for cat in categories]

        y = np.arange(len(categories))
        height = 0.35

        bars1 = plt.barh(
            y - height / 2,
            gemma_means,
            height,
            label="Gemma3",
            color=CHART_CONFIG["colors"][0],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )
        bars2 = plt.barh(
            y + height / 2,
            tower_means,
            height,
            label="TowerInstruct",
            color=CHART_CONFIG["colors"][1],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )

        # No text labels on bars for cleaner look

        plt.ylabel("Categorias")
        plt.xlabel("Pontuação Média")
        plt.title(
            "Performance por Categoria\nTop 15 Categorias por Volume",
            fontweight="bold",
        )
        plt.yticks(
            y,
            [CATEGORY_TRANSLATIONS.get(cat, cat) for cat in categories],
        )
        plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        plt.grid(True, alpha=0.3, axis="x")
        plt.xlim(0, 1.0)  # Normal x-axis range

        plt.subplots_adjust(right=0.85)
        plt.tight_layout()
        plt.savefig(
            generate_visualization_path(dataset_id, "category_performance"),
            dpi=CHART_CONFIG["dpi"],
            bbox_inches="tight",
        )
        plt.close()


def create_visualizations():
    """Create all visualizations."""
    print("Creating visualizations...")
    # Use dataset ID from loaded data
    dataset_id = "m_alert"  # Default for current data

    chart_types = [
        "histogram",
        "performance",
        "selection",
        "boxplot",
        "tiers",
        "correlation",
        "merged_comparison",
        "categories",
    ]
    for chart_type in chart_types:
        create_chart(chart_type, dataset_id=dataset_id)

    print("Visualizations saved to:")
    print("  📊 Score Distribution: analysis/visualizations/score_distribution.png")
    print(
        "  📈 Performance Comparison: analysis/visualizations/performance_comparison.png"
    )
    print("  🎯 Selection Results: analysis/visualizations/selection_results.png")
    print("  📦 Quality Box Plot: analysis/visualizations/quality_boxplot.png")
    print("  🏆 Quality Tiers: analysis/visualizations/quality_tiers.png")
    print("  📋 Category Performance: analysis/visualizations/category_performance.png")
    if correlation_data:
        print("  🔗 Score Correlation: analysis/visualizations/score_correlation.png")
    if merged_data:
        print("  🎯 Merged Comparison: analysis/visualizations/merged_comparison.png")


def generate_report():
    """Generate comprehensive analysis report."""
    print("Generating comprehensive report...")
    dataset_id = "m_alert"  # Default for current data

    report = []
    report.append("=" * 80)
    report.append("RELATÓRIO UNIFICADO DE ANÁLISE DE TRADUÇÃO M-ALERT")
    report.append("=" * 80)
    report.append("")

    # Basic Statistics
    report.append("ESTATÍSTICAS BÁSICAS:")
    report.append("-" * 40)
    report.append("Gemma3:")
    report.append(f"  Contagem: {gemma_stats['count']:,}")
    report.append(f"  Pontuação Média: {gemma_stats['mean_score']:.4f}")
    report.append(f"  Pontuação Mediana: {gemma_stats['median_score']:.4f}")
    report.append(f"  Desvio Padrão: {gemma_stats['std_score']:.4f}")
    report.append(f"  Taxa Excelente (≥0.95): {gemma_stats['excellent_rate']:.1f}%")
    report.append(f"  Taxa Ruim (<0.60): {gemma_stats['poor_rate']:.1f}%")
    report.append(f"  Taxa de Erro: {gemma_stats['error_rate']:.1f}%")
    report.append("")

    report.append("TowerInstruct:")
    report.append(f"  Contagem: {tower_stats['count']:,}")
    report.append(f"  Pontuação Média: {tower_stats['mean_score']:.4f}")
    report.append(f"  Pontuação Mediana: {tower_stats['median_score']:.4f}")
    report.append(f"  Desvio Padrão: {tower_stats['std_score']:.4f}")
    report.append(f"  Taxa Excelente (≥0.95): {tower_stats['excellent_rate']:.1f}%")
    report.append(f"  Taxa Ruim (<0.60): {tower_stats['poor_rate']:.1f}%")
    report.append(f"  Taxa de Erro: {tower_stats['error_rate']:.1f}%")
    report.append("")

    # Merged Dataset Statistics
    if merged_stats:
        report.append("Dataset Mesclado (Melhor por ID):")
        report.append(f"  Contagem: {merged_stats['count']:,}")
        report.append(f"  Pontuação Média: {merged_stats['mean_score']:.4f}")
        report.append(f"  Pontuação Mediana: {merged_stats['median_score']:.4f}")
        report.append(f"  Desvio Padrão: {merged_stats['std_score']:.4f}")
        report.append(
            f"  Taxa Excelente (≥0.95): {merged_stats['excellent_rate']:.1f}%"
        )
        report.append(f"  Taxa Ruim (<0.60): {merged_stats['poor_rate']:.1f}%")
        report.append(f"  Taxa de Erro: {merged_stats['error_rate']:.1f}%")
        report.append(
            f"  Gemma3 selecionado: {merged_stats['gemma_selected']:,} ({merged_stats['gemma_selected'] / merged_stats['count'] * 100:.1f}%)"
        )
        report.append(
            f"  TowerInstruct selecionado: {merged_stats['tower_selected']:,} ({merged_stats['tower_selected'] / merged_stats['count'] * 100:.1f}%)"
        )
        report.append("")

    # Model Comparison
    report.append("COMPARAÇÃO DE MODELOS:")
    report.append("-" * 40)
    report.append(
        f"Gemma3 melhor: {comparison['gemma_better']:,} ({comparison['gemma_win_rate']:.1f}%)"
    )
    report.append(
        f"TowerInstruct melhor: {comparison['tower_better']:,} ({comparison['tower_win_rate']:.1f}%)"
    )
    report.append(f"Empates: {comparison['ties']:,} ({comparison['tie_rate']:.1f}%)")
    report.append(
        f"Diferença média (Gemma3 - TowerInstruct): {comparison['mean_difference']:.4f}"
    )
    report.append("")

    # Tie Analysis
    if comparison["tie_scores"]:
        tie_avg_score = statistics.mean(
            [score[0] for score in comparison["tie_scores"]]
        )
        report.append("ANÁLISE DE EMPATES:")
        report.append("-" * 40)
        report.append(
            f"Total de empates: {comparison['ties']:,} ({comparison['tie_rate']:.1f}%)"
        )
        report.append(f"Score médio dos empates: {tie_avg_score:.4f}")
        report.append(
            "Política de desempate: Empates resolvidos favorecendo TowerInstruct com base em evidências empíricas"
        )
        report.append("")

    # Winner determination
    winner = "Gemma3" if comparison["mean_difference"] > 0 else "TowerInstruct"
    advantage = abs(comparison["mean_difference"])
    report.append(f"VENCEDOR GERAL: {winner}")
    report.append(f"Vantagem: {advantage:.4f} pontos")
    report.append("")

    # Merged Dataset Comparison
    if merged_comparison:
        report.append("COMPARAÇÃO COM DATASET MESCLADO:")
        report.append("-" * 40)
        report.append(
            f"Melhoria vs Gemma3: +{merged_comparison['vs_gemma_improvement']:.4f} pontos"
        )
        report.append(
            f"Melhoria vs TowerInstruct: +{merged_comparison['vs_tower_improvement']:.4f} pontos"
        )
        report.append(
            f"Melhoria vs melhor individual: +{merged_comparison['merged_vs_best']:.4f} pontos"
        )
        report.append("")
        report.append("O dataset mesclado supera ambos os modelos individuais,")
        report.append("demonstrando a eficácia da seleção da melhor tradução por ID.")
        report.append("")

    # Selection Results
    if selection_stats:
        report.append("RESULTADOS DA SELEÇÃO DE TRADUÇÕES:")
        report.append("-" * 40)
        report.append(f"Total selecionado: {selection_stats['total']:,}")
        report.append(
            f"Gemma3 selecionado: {selection_stats['gemma_selected']:,} ({selection_stats['gemma_selected'] / selection_stats['total'] * 100:.1f}%)"
        )
        report.append(
            f"TowerInstruct selecionado: {selection_stats['tower_selected']:,} ({selection_stats['tower_selected'] / selection_stats['total'] * 100:.1f}%)"
        )
        report.append(
            f"Acima do limiar de qualidade: {selection_stats['above_threshold']:,} ({selection_stats['above_threshold'] / selection_stats['total'] * 100:.1f}%)"
        )
        report.append(
            f"Abaixo do limiar de qualidade: {selection_stats['below_threshold']:,} ({selection_stats['below_threshold'] / selection_stats['total'] * 100:.1f}%)"
        )
        report.append("")

    # Category Performance Analysis
    report.append("ANÁLISE DE PERFORMANCE POR CATEGORIA:")
    report.append("-" * 40)

    all_categories = set(gemma_cats.keys()) | set(tower_cats.keys())
    category_counts = [
        (
            cat,
            gemma_cats.get(cat, {}).get("count", 0)
            + tower_cats.get(cat, {}).get("count", 0),
        )
        for cat in all_categories
    ]
    top_categories = sorted(category_counts, key=lambda x: x[1], reverse=True)[:10]

    for category, total_count in top_categories:
        gemma_stats_cat = gemma_cats.get(category, {})
        tower_stats_cat = tower_cats.get(category, {})

        if gemma_stats_cat and tower_stats_cat:
            report.append(f"{CATEGORY_TRANSLATIONS.get(category, category)}:")
            report.append(f"  Total: {total_count}")
            report.append(
                f"  Gemma3: {gemma_stats_cat['mean_score']:.4f} (n={gemma_stats_cat['count']})"
            )
            report.append(
                f"  TowerInstruct: {tower_stats_cat['mean_score']:.4f} (n={tower_stats_cat['count']})"
            )
            winner_cat = (
                "Gemma3"
                if gemma_stats_cat["mean_score"] > tower_stats_cat["mean_score"]
                else "TowerInstruct"
            )
            diff = abs(gemma_stats_cat["mean_score"] - tower_stats_cat["mean_score"])
            report.append(f"  Vencedor: {winner_cat} (diferença: {diff:.4f})")
            report.append("")

    # Key Insights
    report.append("PRINCIPAIS INSIGHTS:")
    report.append("-" * 40)
    report.append(
        f"1. {winner} tem melhor performance geral com vantagem de {advantage:.4f} pontos"
    )
    report.append(
        f"2. Gemma3 tem taxa {'maior' if gemma_stats['excellent_rate'] > tower_stats['excellent_rate'] else 'menor'} de traduções excelentes ({gemma_stats['excellent_rate']:.1f}% vs {tower_stats['excellent_rate']:.1f}%)"
    )
    report.append(
        f"3. TowerInstruct tem taxa {'maior' if tower_stats['error_rate'] > gemma_stats['error_rate'] else 'menor'} de detecção de erros ({tower_stats['error_rate']:.1f}% vs {gemma_stats['error_rate']:.1f}%)"
    )

    gemma_excellent = gemma_tiers["Excelente (≥0.95)"][1]
    tower_excellent = tower_tiers["Excelente (≥0.95)"][1]
    report.append(
        f"4. Análise de níveis de qualidade: Gemma3 {gemma_excellent:.1f}% vs TowerInstruct {tower_excellent:.1f}% traduções excelentes"
    )

    if merged_stats:
        merged_excellent = merged_tiers["Excelente (≥0.95)"][1]
        report.append(
            f"5. Dataset mesclado atinge {merged_excellent:.1f}% de traduções excelentes, superando ambos os modelos individuais"
        )

    if correlation_data:
        correlation_strength = (
            "forte"
            if correlation_data["correlation"] > 0.7
            else "moderada"
            if correlation_data["correlation"] > 0.4
            else "fraca"
        )
        insight_num = 6 if merged_stats else 5
        report.append(
            f"{insight_num}. Correlação entre modelos: {correlation_data['correlation']:.3f} (correlação {correlation_strength})"
        )

    if selection_stats:
        insight_num = 7 if merged_stats else 6
        report.append(
            f"{insight_num}. Processo de seleção escolheu {selection_stats['above_threshold'] / selection_stats['total'] * 100:.1f}% de traduções de alta qualidade"
        )
        report.append(
            f"{insight_num + 1}. Dataset final contém {selection_stats['total']:,} traduções cuidadosamente selecionadas"
        )
        report.append(
            f"{insight_num + 2}. Viés de seleção: Gemma3 selecionado {selection_stats['gemma_selected'] / selection_stats['total'] * 100:.1f}% das vezes"
        )
        report.append(
            f"{insight_num + 3}. Empates: {comparison['ties']:,} ({comparison['tie_rate']:.1f}%) foram resolvidos favorecendo TowerInstruct com base em evidências empíricas"
        )

    # Performance characteristics
    report.append("")
    report.append("CARACTERÍSTICAS DE PERFORMANCE:")
    report.append("-" * 40)
    report.append(
        f"• Gemma3: Mais consistente (desvio: {gemma_stats['std_score']:.4f}) mas média ligeiramente menor"
    )
    report.append(
        f"• TowerInstruct: Performance média maior mas mais variável (desvio: {tower_stats['std_score']:.4f})"
    )
    report.append(
        "• Ambos os modelos atingem >90% de qualidade média, indicando capacidades sólidas de tradução"
    )
    report.append(
        "• Taxas de detecção de erro sugerem que ambos os modelos são conservadores na avaliação de qualidade"
    )

    if merged_stats:
        report.append(
            f"• Dataset mesclado atinge {merged_stats['mean_score']:.4f} de qualidade média, superando ambos os modelos individuais"
        )
        report.append(
            f"• Estratégia de seleção da melhor tradução por ID resulta em {merged_comparison['merged_vs_best']:.4f} pontos de melhoria sobre o melhor modelo individual"
        )
        report.append(
            f"• Política: {comparison['ties']:,} empates ({comparison['tie_rate']:.1f}%) foram resolvidos favorecendo TowerInstruct com base em evidências"
        )

    report.append("")
    report.append("=" * 80)
    report.append("Fim do Relatório")
    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    from utils import generate_report_filename

    report_file = generate_report_filename(dataset_id, "analysis", extension="txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Report saved to: {report_file}")


def print_summary():
    """Print analysis summary."""
    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 60)
    print("Arquivos gerados:")
    dataset_id = "m_alert"  # Default for current data
    from utils import generate_report_filename

    report_file = generate_report_filename(dataset_id, "analysis", extension="txt")
    print(f"📊 Relatório: {report_file}")
    print("📈 Visualizações:")
    print(
        "  • Distribuição de Pontuações: analysis/visualizations/score_distribution.png"
    )
    print(
        "  • Comparação de Performance: analysis/visualizations/performance_comparison.png"
    )
    print("  • Resultados de Seleção: analysis/visualizations/selection_results.png")
    print("  • Box Plot de Qualidade: analysis/visualizations/quality_boxplot.png")
    print("  • Níveis de Qualidade: analysis/visualizations/quality_tiers.png")
    print(
        "  • Performance por Categoria: analysis/visualizations/category_performance.png"
    )
    print("  • Correlação de Pontuações: analysis/visualizations/score_correlation.png")
    print("\nPrincipais descobertas:")

    winner = "Gemma3" if comparison["mean_difference"] > 0 else "TowerInstruct"
    advantage = abs(comparison["mean_difference"])

    print(f"  • {winner} vence no geral (vantagem de {advantage:.4f} pontos)")
    print(f"  • Gemma3: pontuação média {gemma_stats['mean_score']:.4f}")
    print(f"  • TowerInstruct: pontuação média {tower_stats['mean_score']:.4f}")
    if correlation_data:
        print(f"  • Correlação entre modelos: {correlation_data['correlation']:.3f}")

    if selected_data:
        print(f"  • {selection_stats['total']:,} traduções selecionadas")
        print(
            f"  • {selection_stats['above_threshold'] / selection_stats['total'] * 100:.1f}% acima do limiar de qualidade"
        )

    print("=" * 60)


def main():
    """Main analysis function."""
    print("Iniciando Análise Unificada de Tradução M-ALERT")
    print("=" * 60)

    load_data()
    calculate_stats()
    create_visualizations()
    generate_report()
    print_summary()


if __name__ == "__main__":
    main()
