#!/usr/bin/env python3
"""Render LingBot-VA Robotwin video-to-video attention CSVs.

The CSV convention is:
  rows    = concatenated current-video Q tokens, small -> large top -> bottom
  columns = chronological video-history K tokens, small -> large left -> right
  empty fields = future K tokens unavailable to an earlier Q chunk

Examples from the jupyter-plot repository root:

  # One small figure from the first recorded step
  python3 notebooks/26Aug9-Robotwin-VV-attention/render_vv_attention.py \
      --attention-link attn-exp-vv-10-3 --attention-step step_000 \
      --layer 0 --head 0 --formats png

  # Render every discovered step, all heads, and one 4x6 grid per layer
  python3 notebooks/26Aug9-Robotwin-VV-attention/render_vv_attention.py \
      --attention-link attn-exp-vv-10-3 --grid --formats png pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

WORKSET_NAME = "26Aug9-Robotwin-VV-attention"
ATTENTION_PROBABILITY_VMAX = 1.0


def apply_plot_style() -> None:
    """Use the compact serif heatmap style of the paper reference figure."""
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


apply_plot_style()


def find_project_root(start: Path | None = None) -> Path:
    """Find a jupyter-plot root from cwd or an explicitly supplied path."""
    if start is not None:
        root = start.expanduser().resolve()
        if all((root / name).is_dir() for name in ("notebooks", "data", "figures")):
            return root
        raise FileNotFoundError(f"Not a jupyter-plot root: {root}")

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if all((candidate / name).is_dir() for name in ("notebooks", "data", "figures")):
            return candidate
    raise FileNotFoundError("Could not find jupyter-plot root")


def locate_experiment(
    project_root: Path,
    attention_link: str,
    experiment_slug: str | None = None,
) -> tuple[Path, dict]:
    """Resolve one linked output directory and its summary metadata."""
    linked_root = project_root / "data" / WORKSET_NAME / attention_link
    if not linked_root.exists():
        raise FileNotFoundError(f"Missing attention link or target: {linked_root}")

    if experiment_slug is not None:
        experiment_dir = linked_root / experiment_slug
        summary_path = experiment_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
    else:
        summary_paths = sorted(linked_root.glob("*/summary.json"))
        if len(summary_paths) != 1:
            raise RuntimeError(
                f"Expected exactly one summary.json under {linked_root}, "
                f"found {len(summary_paths)}; pass --experiment-slug explicitly."
            )
        summary_path = summary_paths[0]
        experiment_dir = summary_path.parent

    summary = json.loads(summary_path.read_text())
    return experiment_dir, summary


def discover_attention_steps(experiment_dir: Path) -> list[str]:
    """Return sorted step directories, or an empty list for legacy flat outputs."""
    step_dirs = sorted(path for path in experiment_dir.glob("step_*") if path.is_dir())
    empty = [path.name for path in step_dirs if not any(path.glob("l??h??.csv"))]
    if empty:
        raise FileNotFoundError(f"Attention steps contain no layer/head CSVs: {empty}")
    return [path.name for path in step_dirs]


def resolve_attention_step(
    experiment_dir: Path,
    attention_step: str | None,
) -> tuple[Path, str | None]:
    """Resolve one matrix directory while retaining flat-output compatibility."""
    available = discover_attention_steps(experiment_dir)
    if not available:
        if attention_step is not None:
            raise ValueError(f"Flat attention output has no step named {attention_step!r}")
        return experiment_dir, None
    if attention_step is None:
        if len(available) != 1:
            raise RuntimeError(
                f"Found {len(available)} attention steps under {experiment_dir}; "
                "select one explicitly."
            )
        attention_step = available[0]
    if attention_step not in available:
        raise ValueError(f"Unavailable attention step {attention_step!r}; available: {available}")
    return experiment_dir / attention_step, attention_step


def resolve_attention_steps(
    selection: Sequence[str] | None,
    available: Sequence[str],
) -> list[str | None]:
    """Select recorded steps, or one None sentinel for a legacy flat output."""
    available = list(available)
    if not available:
        if selection:
            raise ValueError(f"Flat attention output has no steps: {list(selection)}")
        return [None]
    if selection is None:
        return available
    selected = list(dict.fromkeys(selection))
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Unavailable attention steps: {missing}; available: {available}")
    return selected


def attention_step_summary(summary: dict, attention_step: str | None) -> dict:
    """Resolve metadata stored either at the root or under a steps entry."""
    step_summaries = summary.get("steps")
    if not step_summaries:
        return summary
    if attention_step is None:
        if len(step_summaries) != 1:
            raise RuntimeError(
                f"Summary contains {len(step_summaries)} attention steps; select one explicitly."
            )
        return step_summaries[0]
    matches = [
        step_summary
        for step_summary in step_summaries
        if step_summary.get("csv_directory") == attention_step
    ]
    if len(matches) != 1:
        available = [step_summary.get("csv_directory") for step_summary in step_summaries]
        raise ValueError(f"No unique summary for {attention_step!r}; available: {available}")
    return matches[0]


def metadata(
    experiment_dir: Path,
    summary: dict,
    attention_step: str | None = None,
) -> tuple[tuple[int, int], list[int], int, list[int], list[int]]:
    del experiment_dir  # Retained in the public signature for notebook compatibility.
    step_summary = attention_step_summary(summary, attention_step)
    shape = tuple(int(value) for value in step_summary["logical_shape_per_csv"])
    layers = [int(value) for value in summary["layer_indices"]]
    num_heads = int(step_summary["num_heads"])
    row_bounds = [0, *map(int, step_summary["row_chunk_boundaries"]), shape[0]]
    history_bounds = [0, *map(int, step_summary["history_chunk_boundaries"]), shape[1]]
    return shape, layers, num_heads, row_bounds, history_bounds


def resolve_selection(selection: Sequence[int] | None, available: Iterable[int]) -> list[int]:
    available = list(available)
    if selection is None:
        return available
    selected = [int(value) for value in selection]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Unavailable indices: {missing}")
    return selected


def load_head_matrix(
    experiment_dir: Path,
    expected_shape: tuple[int, int],
    layer: int,
    head: int,
) -> np.ndarray:
    path = experiment_dir / f"l{layer:02d}h{head:02d}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    matrix = np.genfromtxt(
        path,
        delimiter=",",
        dtype=np.float32,
        missing_values="",
        filling_values=np.nan,
        invalid_raise=True,
    )
    if matrix.shape != expected_shape:
        raise ValueError(f"{path.name}: expected {expected_shape}, got {matrix.shape}")
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError(f"{path.name}: no finite values")
    if float(finite.min()) < 0.0:
        raise ValueError(f"{path.name}: negative attention probability")
    if float(finite.max()) > ATTENTION_PROBABILITY_VMAX + 1e-5:
        raise ValueError(f"{path.name}: attention probability exceeds 1")
    return matrix


def sparse_ticks(length: int, maximum: int = 5) -> list[int]:
    count = min(maximum, length)
    return sorted(set(np.linspace(0, length - 1, count, dtype=int).tolist()))


def make_cmap(name: str = "RdBu_r"):
    cmap = mpl.colormaps[name].copy()
    cmap.set_bad("#f2f2f2")
    return cmap


def shared_probability_norm(vmax: float) -> tuple[Normalize, float]:
    """Return one fixed linear color mapping shared by every rendered head."""
    if not np.isfinite(vmax) or vmax <= 0.0:
        raise ValueError(f"vmax must be finite and positive, got {vmax}")
    vcenter = vmax / 2.0
    return Normalize(vmin=0.0, vmax=vmax, clip=True), vcenter


def output_dir(
    figures_root: Path,
    experiment_dir: Path,
    layer: int,
    attention_step: str | None = None,
) -> Path:
    path = figures_root / experiment_dir.name
    if attention_step is not None:
        path /= attention_step
    path /= f"layer{layer}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_head(
    matrix: np.ndarray,
    layer: int,
    head: int,
    vmax: float,
    row_bounds: Sequence[int],
    history_bounds: Sequence[int],
    save_dir: Path,
    formats: Sequence[str] = ("png", "pdf"),
    figsize: tuple[float, float] = (2.55, 2.45),
    dpi: int = 300,
    show: bool = False,
) -> list[Path]:
    rows, columns = matrix.shape
    cmap = make_cmap()
    norm, vcenter = shared_probability_norm(vmax)
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        norm=norm,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlim(-0.5, columns - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks(sparse_ticks(columns))
    ax.set_yticks(sparse_ticks(rows))
    ax.text(
        0.025,
        0.975,
        f"L{layer} H{head}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.set_xlabel("K history token id (small → large)", labelpad=1.5, fontsize=6)
    ax.set_ylabel("Σ Q current token id (top → bottom; small → large)", labelpad=1.5, fontsize=6)
    ax.tick_params(axis="both", pad=0.8, labelsize=5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_ticks([0.0, vcenter, vmax])
    colorbar.set_ticklabels(["0", f"{vcenter:.2g}", f"{vmax:.2g}"])
    colorbar.ax.tick_params(labelsize=5, pad=0.8, length=1.5)
    colorbar.set_label("VV attention probability", fontsize=5.5, labelpad=1.5)
    colorbar.outline.set_linewidth(0.4)
    fig.text(0.5, 0.005, "gray = future K unavailable to this Q chunk", ha="center", fontsize=4.8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    saved = []
    stem = save_dir / f"l{layer:02d}h{head:02d}_vv_attention"
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
        saved.append(path)
    if show:
        plt.show()
    plt.close(fig)
    return saved


def render_layer_grid(
    matrices: dict[int, np.ndarray],
    layer: int,
    heads: Sequence[int],
    vmax: float,
    row_bounds: Sequence[int],
    history_bounds: Sequence[int],
    save_dir: Path,
    formats: Sequence[str] = ("png", "pdf"),
    columns_per_row: int = 6,
    dpi: int = 300,
) -> list[Path]:
    if columns_per_row < 1:
        raise ValueError("columns_per_row must be positive")
    if not heads:
        raise ValueError("At least one head is required")
    rows, columns = next(iter(matrices.values())).shape
    grid_columns = min(columns_per_row, len(heads))
    grid_rows = int(np.ceil(len(heads) / grid_columns))
    cmap = make_cmap()
    norm, vcenter = shared_probability_norm(vmax)

    # Keep every heatmap square and reserve a dedicated column for the colorbar.
    fig = plt.figure(figsize=(1.34 * grid_columns + 0.55, 1.34 * grid_rows + 0.65))
    grid = fig.add_gridspec(
        grid_rows,
        grid_columns + 1,
        width_ratios=[1.0] * grid_columns + [0.055],
        left=0.065,
        right=0.965,
        bottom=0.13,
        top=0.985,
        wspace=0.14,
        hspace=0.16,
    )
    axes = np.empty((grid_rows, grid_columns), dtype=object)
    for row in range(grid_rows):
        for column in range(grid_columns):
            axes[row, column] = fig.add_subplot(grid[row, column])
    colorbar_ax = fig.add_subplot(grid[:, -1])

    image = None
    for index, ax in enumerate(axes.ravel()):
        if index >= len(heads):
            ax.axis("off")
            continue
        head = heads[index]
        image = ax.imshow(
            np.ma.masked_invalid(matrices[head]),
            cmap=cmap,
            norm=norm,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
        )
        row, column = divmod(index, grid_columns)
        ax.set_xlim(-0.5, columns - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks(sparse_ticks(columns))
        ax.set_yticks(sparse_ticks(rows))
        ax.tick_params(
            axis="both",
            pad=0.7,
            labelsize=7,
            length=2.5,
            labelbottom=row == grid_rows - 1,
            labelleft=column == 0,
        )
        ax.text(
            0.025,
            0.975,
            f"H{head}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_ticks([0.0, vcenter, vmax])
    colorbar.set_ticklabels(["0", f"{vcenter:.2g}", f"{vmax:.2g}"])
    colorbar.ax.tick_params(labelsize=7, pad=2.0, length=2.5)
    colorbar.outline.set_linewidth(0.75)
    fig.supxlabel("K history token id", fontsize=9, y=0.065)
    fig.supylabel("Q current token id", fontsize=9, x=0.012)
    fig.text(0.5, 0.018, f"Layer {layer} VV attention probability", ha="center", fontsize=11)

    saved = []
    stem = save_dir / f"layer{layer:02d}_vv_attention_4x6"
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
        saved.append(path)
    plt.close(fig)
    return saved


def render_experiment(
    project_root: Path,
    attention_link: str = "attn-exp-vv-10-3",
    experiment_slug: str | None = None,
    attention_step: str | None = None,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    layer_grid: bool = False,
    formats: Sequence[str] = ("png", "pdf"),
    color_vmax: float = ATTENTION_PROBABILITY_VMAX,
    dpi: int = 300,
) -> dict:
    experiment_dir, summary = locate_experiment(project_root, attention_link, experiment_slug)
    matrix_dir, attention_step = resolve_attention_step(experiment_dir, attention_step)
    step_summary = attention_step_summary(summary, attention_step)
    shape, available_layers, num_heads, row_bounds, history_bounds = metadata(
        experiment_dir, summary, attention_step
    )
    layers = resolve_selection(layers, available_layers)
    heads = resolve_selection(heads, range(num_heads))
    if not layers or not heads:
        raise ValueError("At least one layer and one head must be selected")
    vmax = float(color_vmax)
    shared_probability_norm(vmax)
    figures_root = project_root / "figures" / WORKSET_NAME
    small_paths = []
    grid_paths = []
    for layer in layers:
        matrices = {head: load_head_matrix(matrix_dir, shape, layer, head) for head in heads}
        layer_dir = output_dir(figures_root, experiment_dir, layer, attention_step)
        for head in heads:
            small_paths.extend(
                render_head(matrices[head], layer, head, vmax, row_bounds, history_bounds, layer_dir, formats=formats, dpi=dpi)
            )
        if layer_grid:
            grid_paths.extend(
                render_layer_grid(matrices, layer, heads, vmax, row_bounds, history_bounds, layer_dir, formats=formats, dpi=dpi)
            )
        print(f"rendered layer {layer}: {len(heads)} heads" + (" + 4x6 grid" if layer_grid else ""), flush=True)

    manifest = {
        "experiment": experiment_dir.name,
        "attention_link": attention_link,
        "attention_step": attention_step,
        "scheduler_timestep": step_summary.get("scheduler_timestep"),
        "logical_shape": list(shape),
        "layers": list(layers),
        "heads": list(heads),
        "orientation": {
            "rows": "sum_i Q_current_i; small-to-large from top to bottom",
            "columns": "K_history; small-to-large from left to right",
        },
        "future_columns": "gray / NaN",
        "cmap": "RdBu_r",
        "colorbar_scope": "identical for every selected step, layer, and head",
        "color_center": "fixed linear midpoint",
        "scale_source": "fixed attention probability range; no percentile clipping",
        "vmin": 0.0,
        "vcenter": vmax / 2.0,
        "vmax": vmax,
        "formats": list(formats),
        "small_figure_count": len(small_paths),
        "layer_grid_count": len(grid_paths),
    }
    manifest_dir = figures_root / experiment_dir.name
    if attention_step is not None:
        manifest_dir /= attention_step
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {len(small_paths)} small figures and {len(grid_paths)} layer grids")
    print(f"manifest: {manifest_path}")
    return manifest


def render_attention_steps(
    project_root: Path,
    attention_link: str = "attn-exp-vv-10-3",
    experiment_slug: str | None = None,
    attention_steps: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    layer_grid: bool = False,
    formats: Sequence[str] = ("png", "pdf"),
    color_vmax: float = ATTENTION_PROBABILITY_VMAX,
    dpi: int = 300,
) -> dict[str, dict]:
    """Render selected steps on the fixed attention-probability range [0, 1]."""
    experiment_dir, summary = locate_experiment(project_root, attention_link, experiment_slug)
    steps = resolve_attention_steps(attention_steps, discover_attention_steps(experiment_dir))
    shape, available_layers, num_heads, _, _ = metadata(experiment_dir, summary, steps[0])
    for step in steps[1:]:
        step_shape, step_layers, step_heads, _, _ = metadata(experiment_dir, summary, step)
        if (step_shape, step_layers, step_heads) != (shape, available_layers, num_heads):
            raise ValueError(f"Attention metadata differs across steps; mismatch at {step}")
    layers = resolve_selection(layers, available_layers)
    heads = resolve_selection(heads, range(num_heads))
    vmax = float(color_vmax)
    shared_probability_norm(vmax)

    manifests = {}
    for step in steps:
        label = step or "flat"
        print(f"rendering attention step: {label}", flush=True)
        manifests[label] = render_experiment(
            project_root,
            attention_link=attention_link,
            experiment_slug=experiment_slug,
            attention_step=step,
            layers=layers,
            heads=heads,
            layer_grid=layer_grid,
            formats=formats,
            color_vmax=vmax,
            dpi=dpi,
        )
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--attention-link", default="attn-exp-vv-10-3")
    parser.add_argument("--experiment-slug", default=None)
    parser.add_argument(
        "--attention-step",
        action="append",
        default=None,
        help="Repeat for selected step directories; default: every discovered step",
    )
    parser.add_argument("--layer", type=int, action="append", default=None, help="Repeat for selected layers; default: all")
    parser.add_argument("--head", type=int, action="append", default=None, help="Repeat for selected heads; default: all")
    parser.add_argument("--grid", action="store_true", help="Also render one 4x6 figure per selected layer")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=("png", "pdf"))
    parser.add_argument(
        "--vmax",
        type=float,
        default=ATTENTION_PROBABILITY_VMAX,
        help="Shared probability colorbar maximum; default: 1.0",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = find_project_root(args.project_root)
    render_attention_steps(
        project_root,
        attention_link=args.attention_link,
        experiment_slug=args.experiment_slug,
        attention_steps=args.attention_step,
        layers=args.layer,
        heads=args.head,
        layer_grid=args.grid,
        formats=args.formats,
        color_vmax=args.vmax,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
