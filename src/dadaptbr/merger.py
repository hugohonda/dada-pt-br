#!/usr/bin/env python3
"""
Dataset merger that selects the best translation based on scores from evaluations.
"""

from .utils import (
    ensure_directory_exists,
    get_timestamp,
    load_json_file,
    save_json_file,
)


def load_evaluated_data():
    """Load both evaluated datasets and original data for categories."""
    print("Loading evaluated datasets...")

    gemma_data = load_json_file("data/dadaptbr_M-ALERT_train_gemma_evaluated.json")
    tower_data = load_json_file("data/dadaptbr_M-ALERT_train_tower_evaluated.json")
    original_data = load_json_file("data/dadaptbr_M-ALERT_train_gemma.json")

    print(f"Loaded {len(gemma_data)} Gemma3 evaluations")
    print(f"Loaded {len(tower_data)} TowerInstruct evaluations")
    print(f"Loaded {len(original_data)} original entries")

    return gemma_data, tower_data, original_data


def create_id_mapping(data, model_name):
    """Create ID to item mapping for a dataset."""
    id_map = {}
    for item in data:
        item_id = item["id"]
        if item_id in id_map:
            print(f"Warning: Duplicate ID {item_id} found in {model_name}")
        id_map[item_id] = item
    return id_map


def merge_best_translations(gemma_data, tower_data, original_data):
    """Merge datasets by selecting the best translation for each ID."""
    print("Creating ID mappings...")

    gemma_map = create_id_mapping(gemma_data, "Gemma3")
    tower_map = create_id_mapping(tower_data, "TowerInstruct")
    original_map = create_id_mapping(original_data, "Original")

    # Get all unique IDs
    all_ids = set(gemma_map.keys()) | set(tower_map.keys())
    print(f"Found {len(all_ids)} unique IDs")

    merged_data = []
    gemma_wins = 0
    tower_wins = 0
    ties = 0

    print("Merging translations...")
    for item_id in sorted(all_ids):
        gemma_item = gemma_map.get(item_id)
        tower_item = tower_map.get(item_id)

        if gemma_item and tower_item:
            # Both models have this ID - compare scores
            gemma_score = gemma_item["score"]
            tower_score = tower_item["score"]

            if gemma_score > tower_score:
                best_item = gemma_item.copy()
                best_item["selected_model"] = "gemma3"
                gemma_wins += 1
            elif tower_score > gemma_score:
                best_item = tower_item.copy()
                best_item["selected_model"] = "towerinstruct"
                tower_wins += 1
            else:
                # Tie - prefer TowerInstruct (based on empirical superiority)
                best_item = tower_item.copy()
                best_item["selected_model"] = "towerinstruct"
                ties += 1

        elif gemma_item:
            # Only Gemma3 has this ID
            best_item = gemma_item.copy()
            best_item["selected_model"] = "gemma3"
            gemma_wins += 1
        elif tower_item:
            # Only TowerInstruct has this ID
            best_item = tower_item.copy()
            best_item["selected_model"] = "towerinstruct"
            tower_wins += 1
        else:
            print(f"Warning: No data found for ID {item_id}")
            continue

        # Add metadata
        best_item["gemma_score"] = gemma_item["score"] if gemma_item else None
        best_item["tower_score"] = tower_item["score"] if tower_item else None
        best_item["score_difference"] = (
            (gemma_item["score"] - tower_item["score"])
            if gemma_item and tower_item
            else None
        )

        # Add category from original data
        original_item = original_map.get(item_id)
        if original_item and "category" in original_item:
            best_item["category"] = original_item["category"]
        else:
            best_item["category"] = "unknown"

        merged_data.append(best_item)

    print("Merging complete:")
    print(f"  Gemma3 wins: {gemma_wins}")
    print(f"  TowerInstruct wins: {tower_wins}")
    print(f"  Ties: {ties}")
    print(f"  Total merged: {len(merged_data)}")

    return merged_data


def save_merged_dataset(merged_data):
    """Save the merged dataset with timestamp."""
    timestamp = get_timestamp()
    filename = f"data/dadaptbr_M-ALERT_train_merged_best_{timestamp}.json"

    ensure_directory_exists("data")
    save_json_file(merged_data, filename)

    print(f"Merged dataset saved to: {filename}")
    return filename


def print_summary(merged_data):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("MERGED DATASET SUMMARY")
    print("=" * 60)

    gemma_selected = sum(
        1 for item in merged_data if item["selected_model"] == "gemma3"
    )
    tower_selected = sum(
        1 for item in merged_data if item["selected_model"] == "towerinstruct"
    )

    scores = [item["score"] for item in merged_data]
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)

    print(f"Total translations: {len(merged_data):,}")
    print(
        f"Gemma3 selected: {gemma_selected:,} ({gemma_selected / len(merged_data) * 100:.1f}%)"
    )
    print(
        f"TowerInstruct selected: {tower_selected:,} ({tower_selected / len(merged_data) * 100:.1f}%)"
    )
    print(f"Average score: {avg_score:.4f}")
    print(f"Score range: {min_score:.4f} - {max_score:.4f}")

    # Quality distribution
    excellent = sum(1 for s in scores if s >= 0.95)
    good = sum(1 for s in scores if 0.80 <= s < 0.95)
    fair = sum(1 for s in scores if 0.60 <= s < 0.80)
    poor = sum(1 for s in scores if s < 0.60)

    print("\nQuality distribution:")
    print(f"  Excellent (≥0.95): {excellent:,} ({excellent / len(scores) * 100:.1f}%)")
    print(f"  Good (0.80-0.94): {good:,} ({good / len(scores) * 100:.1f}%)")
    print(f"  Fair (0.60-0.79): {fair:,} ({fair / len(scores) * 100:.1f}%)")
    print(f"  Poor (<0.60): {poor:,} ({poor / len(scores) * 100:.1f}%)")


def main():
    """Main function."""
    print("Merging Best Translations Dataset")
    print("=" * 60)

    # Load data
    gemma_data, tower_data, original_data = load_evaluated_data()

    # Merge datasets
    merged_data = merge_best_translations(gemma_data, tower_data, original_data)

    # Save result
    filename = save_merged_dataset(merged_data)

    # Print summary
    print_summary(merged_data)

    print(f"\n✅ Merged dataset created successfully: {filename}")


if __name__ == "__main__":
    main()
