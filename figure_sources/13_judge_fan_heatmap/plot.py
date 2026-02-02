"""
Plot: Judge vs Fan Attention Heatmap (Model III)
Extracted from task3/dwts/analysis_task3.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paper style settings
FIGSIZE = (8.6, 3.2)
DPI = 300
CMA_ATT = "YlGnBu"

DISPLAY_NAMES = {
    "ballroom_partner": "Partner",
    "celebrity_industry": "Industry",
    "celebrity_homestate": "Home State",
    "celebrity_homecountry/region": "Country",
    "age_z": "Age",
    "partner_score_z": "Partner Skill",
    "progress_z": "Progress",
    "popularity_z": "Popularity",
}

def plot_judge_fan_heatmap(judge_path: str, fan_path: str, output_path: str):
    """Generate judge vs fan attention heatmap."""
    head_j = pd.read_csv(judge_path)
    head_f = pd.read_csv(fan_path)

    token_order = [c for c in head_j.columns if c != "head"]
    x_labels = [DISPLAY_NAMES.get(t, t) for t in token_order]

    judge_data = head_j[token_order].values
    fan_data = head_f[token_order].values

    vmax = float(max(judge_data.max(), fan_data.max(), 1e-9))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    im1 = ax1.imshow(judge_data, vmin=0, vmax=vmax, cmap=CMA_ATT, aspect="auto")
    im2 = ax2.imshow(fan_data, vmin=0, vmax=vmax, cmap=CMA_ATT, aspect="auto")

    for ax, title in [(ax1, "Judges"), (ax2, "Fans")]:
        ax.set_xticks(np.arange(len(token_order)))
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
        ax.set_yticks(np.arange(judge_data.shape[0]))
        ax.set_yticklabels([f"Head {i}" for i in range(judge_data.shape[0])])
        ax.set_title(title)

    fig.colorbar(im2, ax=[ax1, ax2], fraction=0.03, pad=0.02)
    fig.suptitle("Judge vs Fan: Attention Allocation over Attributes", y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_judge_fan_heatmap(
        "head_attention_judge.csv",
        "head_attention_fan.csv",
        "judge_vs_fan_contrib_heatmap.png"
    )
