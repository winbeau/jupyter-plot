"""
Plot: Uncertainty Boxplot by Celebrity (Model I)
Extracted from task1/scripts/run_uncertainty_boxplot.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# Paper style settings
FIGSIZE = (6.5, 4.0)
DPI = 300

def plot_uncertainty_by_celebrity(data_path: str, output_path: str, top_n: int = 20):
    """Generate uncertainty boxplot by celebrity."""
    summary = pd.read_csv(data_path)

    # Select top N celebrities by active weeks
    counts = summary.groupby("celebrity_name")["week"].count().sort_values(ascending=False)
    top_names = counts.head(top_n).index.tolist()

    sub = summary[summary["celebrity_name"].isin(top_names)].copy()
    data_by_name = [
        sub.loc[sub["celebrity_name"] == nm, "V_iqr"].dropna().to_numpy()
        for nm in top_names
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot(data_by_name, labels=top_names, showfliers=False)
    plt.xticks(rotation=90)
    ax.set_xlabel(f"Celebrity (Top {top_n} by active weeks)")
    ax.set_ylabel("Uncertainty (IQR of V_hat)")
    ax.set_title("Model I: Uncertainty by Celebrity")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_uncertainty_by_celebrity(
        "uncertainty_summary_by_person_week.csv",
        "uncertainty_boxplot_by_celebrity_top20.png"
    )
