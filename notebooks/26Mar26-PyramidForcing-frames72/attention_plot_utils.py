"""Shared attention plotting helpers for 72-frame Pyramid Forcing notebooks."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch


NOTEBOOK_DIR = Path(__file__).resolve().parent


def resolve_notebook_path(path: str | os.PathLike[str]) -> Path:
    """Resolve paths the way the notebooks historically used them."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (NOTEBOOK_DIR / path_obj).resolve()


def apply_attention_plot_style() -> None:
    """Apply the shared publication-oriented Matplotlib style."""
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 6,
            "axes.labelsize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
        }
    )


def load_attention_data(data_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a saved attention `.pt` file and print a compact summary."""
    resolved_data_path = resolve_notebook_path(data_path)
    data = torch.load(resolved_data_path, map_location="cpu", weights_only=False)

    print("=" * 60)
    print(f"Layer: {data['layer_index']}")
    print(f"Prompt: {data.get('prompt', 'N/A')}")
    print(f"Num frames: {data.get('num_frames', 'N/A')}")
    print(f"Num heads: {data.get('num_heads', 'N/A')}")
    print(f"Block sizes: {data.get('block_sizes', 'N/A')}")
    print(f"Last block Q frames: {data.get('last_block_query_frames', 'N/A')}")

    full_frame_attn = data["full_frame_attention"].float().numpy()
    print(f"\nFull attention shape: {full_frame_attn.shape}")
    print(f"Range: [{full_frame_attn.min():.4f}, {full_frame_attn.max():.4f}]")

    last_block_attn = data["last_block_frame_attention"].float().numpy()
    print(f"\nLast block attention shape: {last_block_attn.shape}")

    return data


def get_attention_arrays(data: dict[str, Any]) -> tuple[int, int, int, np.ndarray, np.ndarray, Any]:
    """Extract common scalar metadata and NumPy attention arrays."""
    layer_idx = int(data["layer_index"])
    num_frames = int(data["num_frames"])
    num_heads = int(data["num_heads"])
    full_frame_attn = data["full_frame_attention"].float().numpy()
    last_block_attn = data["last_block_frame_attention"].float().numpy()
    last_block_q_frames = data.get("last_block_query_frames", [18, 19, 20])
    return layer_idx, num_frames, num_heads, full_frame_attn, last_block_attn, last_block_q_frames


def render_attention_heatmaps(
    data_path: str | os.PathLike[str],
    save_dir: str | os.PathLike[str],
    *,
    data: dict[str, Any] | None = None,
    figsize: tuple[float, float] = (5.81, 4.2),
    cmap: str = "RdBu_r",
    tick_positions: list[int] | tuple[int, ...] = (0, 36, 71),
    color_min: float = -18.0,
    color_max: float = 18.0,
    save_pdf: bool = True,
    save_png: bool = True,
    save_svg: bool = True,
    show: bool = True,
) -> dict[str, Path]:
    """Render the compact 3x4 all-head frame attention heatmap."""
    apply_attention_plot_style()
    if data is None:
        data = load_attention_data(data_path)
    layer_idx, num_frames, num_heads, full_frame_attn, _, _ = get_attention_arrays(data)

    ncols = 4
    nrows = math.ceil(num_heads / ncols)
    if nrows != 3:
        raise ValueError(f"Expected 12 heads for a 3x4 layout, got {num_heads}")

    vmin = color_min
    vmax = color_max
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        nrows,
        ncols + 1,
        left=0.080,
        right=0.895,
        bottom=0.135,
        top=0.900,
        width_ratios=[1, 1, 1, 1, 0.18],
        wspace=0.105,
        hspace=0.120,
    )
    axes = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            axes[row, col] = fig.add_subplot(grid[row, col])

    image = None
    for head_idx in range(num_heads):
        row = head_idx // ncols
        col = head_idx % ncols
        ax = axes[row, col]
        ax.set_anchor("S" if row == 0 else "N")

        image = ax.imshow(
            full_frame_attn[head_idx],
            cmap=cmap,
            norm=norm,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
        )

        ax.set_title(f"H{head_idx}", loc="center", pad=0.5, fontsize=5)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xlim(-0.5, num_frames - 0.5)
        ax.set_ylim(-0.5, num_frames - 0.5)
        ax.tick_params(axis="both", pad=0.5, labelsize=5)
        if row == nrows - 1:
            ax.set_xlabel("Key frame", labelpad=1, fontsize=6)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("Query frame", labelpad=1, fontsize=6)
        else:
            ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    fig.canvas.draw()
    plot_positions = [ax.get_position() for ax in axes.ravel()]
    plot_bottom = min(pos.y0 for pos in plot_positions)
    plot_top = max(pos.y1 for pos in plot_positions)
    colorbar_ax = fig.add_axes([0.925, plot_bottom, 0.016, plot_top - plot_bottom])
    cbar = fig.colorbar(image, cax=colorbar_ax, orientation="vertical")
    cbar.set_ticks([0.0])
    cbar.set_ticklabels(["0"])
    cbar.ax.tick_params(labelsize=5, pad=1)
    cbar.ax.text(
        -0.95,
        0.5,
        "Attention Logit",
        transform=cbar.ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=6,
    )
    cbar.ax.text(0.5, 1.02, "max", transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=5)
    cbar.ax.text(0.5, -0.035, "min", transform=cbar.ax.transAxes, ha="center", va="top", fontsize=5)

    resolved_save_dir = resolve_notebook_path(save_dir)
    resolved_save_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"layer{layer_idx}_2d_heatmap_all_heads"
    saved_paths: dict[str, Path] = {}
    if save_pdf:
        saved_paths["pdf"] = resolved_save_dir / f"{output_stem}.pdf"
        fig.savefig(saved_paths["pdf"], bbox_inches="tight", pad_inches=0)
        print(f"Saved PDF: {saved_paths['pdf']}")
    if save_png:
        saved_paths["png"] = resolved_save_dir / f"{output_stem}.png"
        fig.savefig(saved_paths["png"], bbox_inches="tight", pad_inches=0)
        print(f"Saved PNG: {saved_paths['png']}")
    if save_svg:
        saved_paths["svg"] = resolved_save_dir / f"{output_stem}.svg"
        fig.savefig(saved_paths["svg"], bbox_inches="tight", pad_inches=0)
        print(f"Saved SVG: {saved_paths['svg']}")

    print(f"Fixed range: [{vmin:.3f}, 0.000, {vmax:.3f}]")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_paths


def render_attention_lineplots(
    data_path: str | os.PathLike[str],
    save_dir: str | os.PathLike[str],
    *,
    data: dict[str, Any] | None = None,
    bar_color: str = "tab:blue",
    save_svg: bool = True,
    show: bool = True,
) -> dict[str, Path]:
    """Render the per-head last-block attention distribution grid."""
    apply_attention_plot_style()
    if data is None:
        data = load_attention_data(data_path)
    layer_idx, num_frames, num_heads, _, last_block_attn, last_block_q_frames = get_attention_arrays(data)

    key_indices = np.arange(num_frames)
    ncols = 4
    nrows = math.ceil(num_heads / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 7.5))
    axes = axes.flatten()

    if num_frames <= 30:
        tick_step = 5
    elif num_frames <= 60:
        tick_step = 10
    else:
        tick_step = 20
    tick_pos = list(range(0, num_frames, tick_step)) + [num_frames - 1]
    tick_pos = sorted(set(tick_pos))

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        head = last_block_attn[head_idx]
        first = head[0]
        middle = head[1:-1].mean() if len(head) > 2 else head.mean()
        sink_score = first - middle

        ax.fill_between(key_indices, 0, head, color=bar_color, alpha=0.75, linewidth=0)
        ax.plot(
            key_indices,
            head,
            "-",
            color="black",
            linewidth=0.75,
        )
        ax.set_title(f"H{head_idx} (sink={sink_score:.2f})", fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(tick_pos)
        ax.set_box_aspect(1)

    for extra_idx in range(num_heads, len(axes)):
        axes[extra_idx].axis("off")

    fig.suptitle(
        f"Layer {layer_idx}: Per-Head Attention Distribution\n"
        f"Query: frames {last_block_q_frames}",
        fontsize=12,
        fontweight="bold",
        y=1.00 + (0.01 * nrows),
    )
    plt.tight_layout()

    saved_paths: dict[str, Path] = {}
    if save_svg:
        resolved_save_dir = resolve_notebook_path(save_dir)
        resolved_save_dir.mkdir(parents=True, exist_ok=True)
        saved_paths["svg"] = resolved_save_dir / f"layer{layer_idx}_perhead_grid.svg"
        plt.savefig(saved_paths["svg"], format="svg", bbox_inches="tight")
        print(f"Saved: {saved_paths['svg']}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_paths


def render_attention_row(
    data_path: str | os.PathLike[str],
    save_dir: str | os.PathLike[str],
    *,
    data: dict[str, Any] | None = None,
    figsize: tuple[float, float] = (13.97 / 2.54, 2.15),
    cmap: str = "RdBu_r",
    tick_positions: list[int] | tuple[int, ...] = (0, 36, 71),
    color_min: float = -18.0,
    color_max: float = 18.0,
    bar_color: str = "tab:blue",
    save_pdf: bool = True,
    save_png: bool = True,
    save_svg: bool = True,
    show: bool = True,
) -> dict[str, Path]:
    """Render heatmap and per-head distribution as two equal-width panels in one row."""
    apply_attention_plot_style()
    if data is None:
        data = load_attention_data(data_path)
    layer_idx, num_frames, num_heads, full_frame_attn, last_block_attn, _ = get_attention_arrays(data)

    ncols = 4
    nrows = math.ceil(num_heads / ncols)
    if nrows != 3:
        raise ValueError(f"Expected 12 heads for a 3x4 layout, got {num_heads}")

    fig = plt.figure(figsize=figsize)
    outer_grid = fig.add_gridspec(
        1,
        2,
        left=0.032,
        right=0.992,
        bottom=0.108,
        top=0.975,
        width_ratios=[1, 1],
        wspace=0.060,
    )

    fig.text(0.262, 0.010, "(a) Frame attention logits", ha="center", va="bottom", fontsize=7)
    fig.text(0.745, 0.010, "(b) Last-block attention distribution", ha="center", va="bottom", fontsize=7)

    heat_grid = outer_grid[0, 0].subgridspec(
        nrows,
        ncols + 1,
        width_ratios=[1, 1, 1, 1, 0.10],
        wspace=0.18,
        hspace=0.18,
    )
    heat_axes = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            heat_axes[row, col] = fig.add_subplot(heat_grid[row, col])

    norm = TwoSlopeNorm(vmin=color_min, vcenter=0.0, vmax=color_max)
    image = None
    for head_idx in range(num_heads):
        row = head_idx // ncols
        col = head_idx % ncols
        ax = heat_axes[row, col]
        image = ax.imshow(
            full_frame_attn[head_idx],
            cmap=cmap,
            norm=norm,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
        )
        ax.text(
            0.04,
            0.96,
            f"H{head_idx}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.15},
        )
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xlim(-0.5, num_frames - 0.5)
        ax.set_ylim(-0.5, num_frames - 0.5)
        ax.tick_params(axis="both", pad=0.2, labelsize=6, length=1.2)
        ax.set_box_aspect(1)
        if row != nrows - 1:
            ax.set_xticklabels([])
        if col != 0:
            ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.35)

    cbar_ax = fig.add_subplot(heat_grid[:, -1])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation="vertical")
    cbar.set_ticks([0.0])
    cbar.set_ticklabels(["0"])
    cbar.ax.tick_params(labelsize=6, pad=0.5, length=1.2)
    cbar.outline.set_linewidth(0.35)

    line_grid = outer_grid[0, 1].subgridspec(
        nrows,
        ncols + 1,
        width_ratios=[1, 1, 1, 1, 0.10],
        wspace=0.18,
        hspace=0.18,
    )
    line_axes = np.empty((nrows, ncols), dtype=object)
    key_indices = np.arange(num_frames)
    line_tick_positions = [0, num_frames - 1]
    for row in range(nrows):
        for col in range(ncols):
            head_idx = row * ncols + col
            ax = fig.add_subplot(line_grid[row, col])
            line_axes[row, col] = ax
            head = last_block_attn[head_idx]
            first = head[0]
            middle = head[1:-1].mean() if len(head) > 2 else head.mean()
            sink_score = first - middle

            ax.fill_between(key_indices, 0, head, color=bar_color, alpha=0.75, linewidth=0)
            ax.plot(key_indices, head, "-", color="black", linewidth=0.38)
            ax.text(
                0.04,
                0.96,
                f"H{head_idx}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.15},
            )
            ax.set_xlim(-0.5, num_frames - 0.5)
            ax.set_xticks(line_tick_positions)
            ax.tick_params(axis="both", pad=0.2, labelsize=6, length=1.2)
            if row != nrows - 1:
                ax.set_xticklabels([])
            if col != 0:
                ax.set_yticklabels([])
            ax.grid(True, alpha=0.25, linewidth=0.35)
            ax.set_box_aspect(1)
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)

    resolved_save_dir = resolve_notebook_path(save_dir)
    resolved_save_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"layer{layer_idx}_attention_heatmap_lineplot_row"
    saved_paths: dict[str, Path] = {}
    with mpl.rc_context({"savefig.bbox": None}):
        if save_pdf:
            saved_paths["pdf"] = resolved_save_dir / f"{output_stem}.pdf"
            fig.savefig(saved_paths["pdf"], bbox_inches=None, pad_inches=0)
            print(f"Saved PDF: {saved_paths['pdf']}")
        if save_png:
            saved_paths["png"] = resolved_save_dir / f"{output_stem}.png"
            fig.savefig(saved_paths["png"], bbox_inches=None, pad_inches=0)
            print(f"Saved PNG: {saved_paths['png']}")
        if save_svg:
            saved_paths["svg"] = resolved_save_dir / f"{output_stem}.svg"
            fig.savefig(saved_paths["svg"], bbox_inches=None, pad_inches=0)
            print(f"Saved SVG: {saved_paths['svg']}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_paths


def print_attention_statistics(
    data_path: str | os.PathLike[str],
    *,
    data: dict[str, Any] | None = None,
) -> None:
    """Print global and per-head statistics for a saved attention file."""
    if data is None:
        data = load_attention_data(data_path)
    layer_idx, num_frames, num_heads, full_frame_attn, last_block_attn, _ = get_attention_arrays(data)

    print("=" * 60)
    print(f"Layer {layer_idx} Statistics")
    print("=" * 60)

    diag = np.array([full_frame_attn[h, i, i] for h in range(num_heads) for i in range(num_frames)])
    diag_mean = diag.mean()
    first_col = full_frame_attn[:, :, 0]
    first_col_mean = first_col[first_col != 0].mean() if (first_col != 0).any() else 0

    print(f"Diagonal mean (self-attention): {diag_mean:.4f}")
    print(f"First frame mean (sink): {first_col_mean:.4f}")

    print("\nPer-Head Statistics (last block):")
    for head_idx in range(num_heads):
        head = last_block_attn[head_idx]
        first = head[0]
        middle = head[1:-1].mean() if len(head) > 2 else head.mean()
        last = head[-1]
        sink = first - middle
        print(
            f"  H{head_idx:2d}: first={first:7.3f}, mid={middle:7.3f}, "
            f"last={last:7.3f}, sink={sink:+7.3f}"
        )


def render_attention_report(
    data_path: str | os.PathLike[str],
    save_dir: str | os.PathLike[str],
    *,
    figsize: tuple[float, float] = (5.81, 4.2),
    cmap: str = "RdBu_r",
    tick_positions: list[int] | tuple[int, ...] = (0, 36, 71),
    color_min: float = -18.0,
    color_max: float = 18.0,
    save_pdf: bool = True,
    save_png: bool = True,
    save_svg: bool = True,
    bar_color: str = "tab:blue",
    show: bool = True,
) -> dict[str, dict[str, Path]]:
    """Compatibility renderer for the historical all-in-one notebook."""
    data = load_attention_data(data_path)
    heatmap_paths = render_attention_heatmaps(
        data_path,
        save_dir,
        data=data,
        figsize=figsize,
        cmap=cmap,
        tick_positions=tick_positions,
        color_min=color_min,
        color_max=color_max,
        save_pdf=save_pdf,
        save_png=save_png,
        save_svg=save_svg,
        show=False,
    )
    lineplot_paths = render_attention_lineplots(
        data_path,
        save_dir,
        data=data,
        bar_color=bar_color,
        save_svg=save_svg,
        show=False,
    )
    row_paths = render_attention_row(
        data_path,
        save_dir,
        data=data,
        cmap=cmap,
        tick_positions=tick_positions,
        color_min=color_min,
        color_max=color_max,
        bar_color=bar_color,
        save_pdf=save_pdf,
        save_png=save_png,
        save_svg=save_svg,
        show=show,
    )
    print_attention_statistics(data_path, data=data)
    return {"heatmap": heatmap_paths, "lineplot": lineplot_paths, "row": row_paths}
