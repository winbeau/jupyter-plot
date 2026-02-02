"""
Plot: S27 Weekly Scores Case Study
"""

import pandas as pd
import matplotlib.pyplot as plt

# Paper style settings
FIGSIZE = (6.5, 4.0)
DPI = 300

def plot_s27_weekly_scores(data_path: str, output_path: str):
    """Generate S27 weekly scores trajectory plot."""
    df = pd.read_csv(data_path)
    s27 = df[df["season"] == 27].copy()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Plot top contestants
    for name in s27["celebrity_name"].unique()[:8]:  # Top 8 for readability
        subset = s27[s27["celebrity_name"] == name]
        ax.plot(subset["week"], subset["V_hat"], marker="o", label=name, linewidth=1.5)

    ax.set_xlabel("Week")
    ax.set_ylabel("Inferred Fan Vote (V_hat)")
    ax.set_title("Season 27: Weekly Score Trajectories")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_s27_weekly_scores(
        "inferred_votes_long.csv",
        "s27_weekly_scores.png"
    )
