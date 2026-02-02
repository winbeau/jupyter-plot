"""
Plot: Season Consistency (Model I)
Extracted from task1/model1.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# Paper style settings
FIGSIZE = (6.5, 3.5)  # Single column width
DPI = 300

def plot_season_consistency(data_path: str, output_path: str):
    """Generate season consistency bar chart."""
    season_cons = pd.read_csv(data_path)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(
        season_cons["season"].astype(str),
        season_cons["consistency"].astype(float),
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.5
    )
    ax.set_xlabel("Season")
    ax.set_ylabel("Elimination Consistency")
    ax.set_title("Model I: Season-level Elimination Consistency")
    ax.set_ylim(0, 1.05)

    # Rotate x labels if many seasons
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_season_consistency(
        "season_consistency.csv",
        "season_consistency_model1.png"
    )
