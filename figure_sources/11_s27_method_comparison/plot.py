"""
Plot: S27 Method Comparison Case Study
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paper style settings
FIGSIZE = (6.5, 4.0)
DPI = 300

def plot_s27_method_comparison(data_path: str, output_path: str):
    """Generate S27 method comparison plot."""
    by_week = pd.read_csv(data_path)
    s27 = by_week[by_week["season"] == 27].copy()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    weeks = s27["week"].to_numpy()
    pct_match = s27["pct_match"].to_numpy()
    rank_match = s27["rank_match"].to_numpy()

    x = np.arange(len(weeks))
    width = 0.35

    ax.bar(x - width/2, pct_match, width, label="Percentage", color="#4C78A8", alpha=0.8)
    ax.bar(x + width/2, rank_match, width, label="Rank", color="#F58518", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"W{w}" for w in weeks])
    ax.set_xlabel("Week")
    ax.set_ylabel("Match with Actual (0/1)")
    ax.set_title("Season 27: Method Accuracy Comparison")
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_s27_method_comparison(
        "method_compare_by_week.csv",
        "s27_method_comparison.png"
    )
