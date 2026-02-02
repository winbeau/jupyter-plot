"""
Plot: Uncertainty Boxplot by Week (Model I)
Extracted from task1/scripts/run_uncertainty_boxplot.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# Paper style settings
FIGSIZE = (6.5, 3.5)
DPI = 300

def plot_uncertainty_by_week(data_path: str, output_path: str):
    """Generate uncertainty boxplot by week."""
    summary = pd.read_csv(data_path)

    week_order = sorted(summary["week"].unique().tolist())
    data_by_week = [
        summary.loc[summary["week"] == w, "V_iqr"].dropna().to_numpy()
        for w in week_order
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot(data_by_week, labels=[str(w) for w in week_order], showfliers=False)
    ax.set_xlabel("Week")
    ax.set_ylabel("Uncertainty (IQR of V_hat)")
    ax.set_title("Model I: Uncertainty by Week")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_uncertainty_by_week(
        "uncertainty_summary_by_person_week.csv",
        "uncertainty_boxplot_by_week.png"
    )
