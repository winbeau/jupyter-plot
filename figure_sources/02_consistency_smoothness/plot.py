"""
Plot: Consistency with Smoothness Heatmap (Model I)
Extracted from task1/scripts/run_consistency_smoothness.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paper style settings
FIGSIZE = (6.5, 2.8)  # Wide heatmap
DPI = 300
DELTA_THR = 0.15
LAMBDA = 0.5

def plot_consistency_smoothness(data_path: str, output_path: str):
    """Generate consistency-smoothness heatmap."""
    season_df = pd.read_csv(data_path)

    seasons = season_df["season"].astype(int).to_numpy()
    L = season_df["elim_consistency"].to_numpy(float)
    P = season_df["smooth_penalty"].to_numpy(float)
    Score = season_df["final_score"].to_numpy(float)

    P_filled = np.where(np.isfinite(P), P, 0.0)
    M = np.vstack([L, P_filled, Score])

    row_labels = [r"$L_s$ (consistency)", r"$P_s$ (penalty)", r"$Score_s$"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(M, aspect="auto", cmap="viridis")

    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(seasons))[::2])
    ax.set_xticklabels([str(s) for s in seasons[::2]], rotation=0)
    ax.set_xlabel("Season")
    ax.set_title(f"Consistency & Smoothness (δ={DELTA_THR}, λ={LAMBDA})")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Value")

    # Annotate values
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_consistency_smoothness(
        "q1_consistency_with_smoothness_by_season.csv",
        "q1_consistency_with_smoothness.png"
    )
