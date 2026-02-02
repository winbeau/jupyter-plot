"""
Plot: Disagreement Rate by Season (Model II)
Extracted from task2/scripts/T2main.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# Paper style settings
FIGSIZE = (6.5, 3.5)
DPI = 300

def plot_disagreement_rate(data_path: str, output_path: str):
    """Generate disagreement rate bar chart."""
    by_season = pd.read_csv(data_path)

    x = by_season["season"].astype(int).astype(str).to_list()
    y = by_season["disagreement_rate"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x, y, color="#4C78A8", edgecolor="black", linewidth=0.5)
    plt.xticks(rotation=45, ha="right")
    ax.set_xlabel("Season")
    ax.set_ylabel("Disagreement rate")
    ax.set_title("Model II: Percentage vs Rank Disagreement by Season")
    ax.set_ylim(0.0, 1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_disagreement_rate(
        "method_compare_by_season.csv",
        "disagreement_by_season.png"
    )
