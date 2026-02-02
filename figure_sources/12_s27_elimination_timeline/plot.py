"""
Plot: S27 Elimination Timeline Case Study
"""

import pandas as pd
import matplotlib.pyplot as plt

# Paper style settings
FIGSIZE = (6.5, 4.5)
DPI = 300

def plot_s27_elimination_timeline(data_path: str, output_path: str):
    """Generate S27 elimination timeline plot."""
    df = pd.read_csv(data_path)
    s27 = df[(df["season"] == 27) & (df["is_eliminated_this_week"] == 1)].copy()
    s27 = s27.sort_values("week")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    weeks = s27["week"].to_numpy()
    names = s27["celebrity_name"].to_list()
    v_hats = s27["V_hat"].to_numpy()

    ax.barh(range(len(names)), weeks, color="#4C78A8", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Elimination Week")
    ax.set_title("Season 27: Elimination Timeline")

    # Annotate with V_hat
    for i, (w, v) in enumerate(zip(weeks, v_hats)):
        ax.text(w + 0.1, i, f"V={v:.2f}", va="center", fontsize=7)

    ax.invert_yaxis()  # First eliminated at top
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_s27_elimination_timeline(
        "inferred_votes_long.csv",
        "s27_elimination_timeline.png"
    )
