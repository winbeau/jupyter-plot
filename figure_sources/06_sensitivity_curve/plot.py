"""
Plot: Sensitivity to Vote Share (Model II)
Extracted from task2/scripts/T2main.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paper style settings
FIGSIZE = (6.5, 3.5)
DPI = 300

def rank_share_from_values(values: np.ndarray) -> np.ndarray:
    """Compute rank-share from values."""
    s = pd.Series(values)
    r = s.rank(ascending=False, method="average").to_numpy(dtype=float)
    N = len(r)
    score = (N - r + 1.0)
    return score / score.sum()

def plot_sensitivity_curve(N: int = 10, output_path: str = "math_sensitivity_vote_share_N10.png"):
    """Generate sensitivity comparison plot."""
    vs = np.linspace(0.01, 0.80, 200)
    fv_pct = []
    fv_rank = []

    for v in vs:
        rest = (1.0 - v) / (N - 1)
        V = np.array([v] + [rest] * (N - 1), dtype=float)
        fv_pct.append(V[0])
        FV = rank_share_from_values(V)
        fv_rank.append(FV[0])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(vs, fv_pct, label="Percentage: FV = vote share", linewidth=2, color="#4C78A8")
    ax.plot(vs, fv_rank, label="Rank: FV = rank-share", linewidth=2, color="#F58518")
    ax.set_xlabel("Target contestant vote share")
    ax.set_ylabel("FV assigned to target")
    ax.set_title(f"Model II: Sensitivity to magnitude (N={N})")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_sensitivity_curve()
