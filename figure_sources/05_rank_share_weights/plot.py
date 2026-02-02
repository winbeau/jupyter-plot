"""
Plot: Rank-Share Weights (Model II)
Extracted from task2/scripts/T2main.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Paper style settings
FIGSIZE = (6.5, 3.5)
DPI = 300

def plot_rank_share_weights(N: int = 10, output_path: str = "math_rank_share_weights_N10.png"):
    """Generate rank-share weight structure plot."""
    ranks = np.arange(1, N + 1, dtype=float)
    weights = (N - ranks + 1.0)
    weights = weights / weights.sum()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(ranks, weights, marker="o", color="#4C78A8", linewidth=2)
    ax.invert_xaxis()  # rank=1 on right
    ax.set_xlabel("Rank (1 = best)")
    ax.set_ylabel("Rank-share weight")
    ax.set_title(f"Model II: Rank-share weights (N={N})")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_rank_share_weights()
