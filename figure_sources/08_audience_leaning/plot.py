"""
Plot: Audience Leaning Analysis (Model II)
Extracted from task2/scripts/T2main.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# Paper style settings
FIGSIZE = (6.5, 3.5)
DPI = 300

def plot_audience_leaning(data_path: str, output_path: str):
    """Generate audience leaning bar chart."""
    by_season = pd.read_csv(data_path)

    x = by_season["season"].astype(int).astype(str).to_list()
    y = by_season["pct_more_audience_rate"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x, y, color="#F58518", edgecolor="black", linewidth=0.5)
    plt.xticks(rotation=45, ha="right")
    ax.set_xlabel("Season")
    ax.set_ylabel("Rate (Percentage eliminates lower V_hat)")
    ax.set_title("Model II: Audience Leaning (in disagreement weeks)")
    ax.set_ylim(0.0, 1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_audience_leaning(
        "method_compare_by_season.csv",
        "audience_leaning_by_season.png"
    )
