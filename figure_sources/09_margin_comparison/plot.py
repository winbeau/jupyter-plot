"""
Plot: Margin Comparison (Model II)
Extracted from task2/scripts/T2main.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# Paper style settings
FIGSIZE = (6.5, 3.5)
DPI = 300

def plot_margin_comparison(data_path: str, output_path: str):
    """Generate margin comparison plot."""
    by_season = pd.read_csv(data_path)

    x = by_season["season"].astype(int).astype(str).to_list()
    m_pct = by_season["median_margin_pct"].to_numpy(dtype=float)
    m_rank = by_season["median_margin_rank"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, m_pct, marker="o", label="Percentage margin", color="#4C78A8", linewidth=2)
    ax.plot(x, m_rank, marker="o", label="Rank margin", color="#F58518", linewidth=2)
    plt.xticks(rotation=45, ha="right")
    ax.set_xlabel("Season")
    ax.set_ylabel("Median elimination margin")
    ax.set_title("Model II: Elimination Margin (smaller = closer)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_margin_comparison(
        "method_compare_by_season.csv",
        "margin_comparison.png"
    )
