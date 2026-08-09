#!/usr/bin/env python3
"""Render LingBot-VA Robotwin video-to-video attention CSVs.

The CSV convention is:
  rows    = concatenated current-video Q tokens, small -> large top -> bottom
  columns = visible video K tokens (past + current chunk), small -> large left -> right
  empty fields = future K tokens unavailable to an earlier Q chunk

Examples from the jupyter-plot repository root:

  # One small figure from the first selected step
  python3 notebooks/26Aug9-Robotwin-VV-attention/render_vv_attention.py \
      --attention-link attn-exp-vv-10-6 --attention-step step_000 \
      --layer 0 --head 0 --formats png

  # Render the default selected steps (0/4/9/14/19/24)
  python3 notebooks/26Aug9-Robotwin-VV-attention/render_vv_attention.py \
      --attention-link attn-exp-vv-10-6 --grid --formats png pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np

WORKSET_NAME = "26Aug9-Robotwin-VV-attention"
DEFAULT_ATTENTION_LINK = "attn-exp-vv-10-6"
DEFAULT_ATTENTION_STEPS = (
    "step_000",
    "step_004",
    "step_009",
    "step_014",
    "step_019",
    "step_024",
)
ATTENTION_PROBABILITY_VMAX = 1.0
DEFAULT_BLOCK_MAX_SIZE = 6
DEFAULT_POWER_GAMMA = 0.25


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


def block_max_pool(matrix: np.ndarray, block_size: int) -> np.ndarray:
    """Preserve the strongest raw probability in each display block."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if block_size == 1:
        return matrix
    rows, columns = matrix.shape
    pooled_rows = int(np.ceil(rows / block_size))
    pooled_columns = int(np.ceil(columns / block_size))
    padded = np.full(
        (pooled_rows * block_size, pooled_columns * block_size),
        np.nan,
        dtype=matrix.dtype,
    )
    padded[:rows, :columns] = matrix
    blocks = padded.reshape(
        pooled_rows,
        block_size,
        pooled_columns,
        block_size,
    ).transpose(0, 2, 1, 3)
    finite = np.isfinite(blocks)
    pooled = np.where(finite, blocks, -np.inf).max(axis=(2, 3))
    pooled[~finite.any(axis=(2, 3))] = np.nan
    return pooled.astype(matrix.dtype, copy=False)


def draw_chunk_boundaries(
    ax: mpl.axes.Axes,
    row_bounds: Sequence[int],
    history_bounds: Sequence[int],
) -> None:
    """Mark the original Q/K chunk boundaries on a pooled heatmap."""
    for boundary in history_bounds[1:-1]:
        ax.axvline(boundary - 0.5, color="#64748b", linewidth=0.28, alpha=0.5)
    for boundary in row_bounds[1:-1]:
        ax.axhline(boundary - 0.5, color="#64748b", linewidth=0.28, alpha=0.5)


def sparse_ticks(length: int, maximum: int = 5) -> list[int]:
    count = min(maximum, length)
    return sorted(set(np.linspace(0, length - 1, count, dtype=int).tolist()))


def make_cmap():
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "white_to_red",
        ("#ffffff", "#fee2e2", "#fca5a5", "#ef4444", "#b91c1c"),
    )
    cmap.set_bad("#f2f2f2")
    return cmap


def shared_probability_norm(
    vmax: float,
    power_gamma: float = DEFAULT_POWER_GAMMA,
) -> Normalize:
    """Return one PowerNorm mapping shared by a layer's selected heads."""
    if not np.isfinite(vmax) or vmax <= 0.0:
        raise ValueError(f"vmax must be finite and positive, got {vmax}")
    if not np.isfinite(power_gamma) or power_gamma <= 0.0:
        raise ValueError(f"power_gamma must be finite and positive, got {power_gamma}")
    return PowerNorm(gamma=power_gamma, vmin=0.0, vmax=vmax, clip=True)


def probability_colorbar_ticks(vmax: float) -> tuple[list[float], list[str]]:
    """Use actual raw-probability labels on the nonlinear colorbar."""
    candidates = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    ticks = [value for value in candidates if value < vmax]
    ticks.append(vmax)
    labels = [
        "0" if value == 0.0 else (f"{value:.0e}" if value < 1e-2 else f"{value:.2g}")
        for value in ticks
    ]
    return ticks, labels


def render_method_suffix(block_size: int, power_gamma: float) -> str:
    gamma_text = f"{power_gamma:g}".replace(".", "p")
    return f"blockmax{block_size}_power{gamma_text}"


def layer_probability_vmax(
    matrices: dict[int, np.ndarray],
    explicit_vmax: float | None = None,
    power_gamma: float = DEFAULT_POWER_GAMMA,
) -> float:
    """Use one exact colorbar maximum for every selected head in a layer."""
    if explicit_vmax is not None:
        vmax = float(explicit_vmax)
        shared_probability_norm(vmax, power_gamma)
        return vmax
    maxima = [float(np.nanmax(matrix)) for matrix in matrices.values()]
    vmax = max(maxima)
    shared_probability_norm(vmax, power_gamma)
    return vmax


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
    block_size: int = DEFAULT_BLOCK_MAX_SIZE,
    power_gamma: float = DEFAULT_POWER_GAMMA,
    show: bool = False,
) -> list[Path]:
    rows, columns = matrix.shape
    display_matrix = block_max_pool(matrix, block_size)
    cmap = make_cmap()
    norm = shared_probability_norm(vmax, power_gamma)
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        np.ma.masked_invalid(display_matrix),
        cmap=cmap,
        norm=norm,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
        extent=(-0.5, columns - 0.5, rows - 0.5, -0.5),
    )
    ax.set_xlim(-0.5, columns - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks(sparse_ticks(columns))
    ax.set_yticks(sparse_ticks(rows))
    draw_chunk_boundaries(ax, row_bounds, history_bounds)
    ax.text(
        0.025,
        0.975,
        f"L{layer} H{head}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.set_xlabel("Visible video K token id", labelpad=1.5, fontsize=6)
    ax.set_ylabel("Video Q token id", labelpad=1.5, fontsize=6)
    ax.tick_params(axis="both", pad=0.8, labelsize=5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    ticks, labels = probability_colorbar_ticks(vmax)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(labels)
    colorbar.ax.tick_params(labelsize=5, pad=0.8, length=1.5)
    colorbar.set_label("Raw VV probability (block max)", fontsize=5.5, labelpad=1.5)
    colorbar.outline.set_linewidth(0.4)
    fig.text(
        0.5,
        0.005,
        f"{block_size}×{block_size} block-max · PowerNorm γ={power_gamma:g} · gray=future K",
        ha="center",
        fontsize=4.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    saved = []
    method = render_method_suffix(block_size, power_gamma)
    stem = save_dir / f"l{layer:02d}h{head:02d}_vv_attention_{method}"
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
    block_size: int = DEFAULT_BLOCK_MAX_SIZE,
    power_gamma: float = DEFAULT_POWER_GAMMA,
) -> list[Path]:
    if columns_per_row < 1:
        raise ValueError("columns_per_row must be positive")
    if not heads:
        raise ValueError("At least one head is required")
    rows, columns = next(iter(matrices.values())).shape
    display_matrices = {
        head: block_max_pool(matrices[head], block_size)
        for head in heads
    }
    grid_columns = min(columns_per_row, len(heads))
    grid_rows = int(np.ceil(len(heads) / grid_columns))
    cmap = make_cmap()
    norm = shared_probability_norm(vmax, power_gamma)

    # At 1200 DPI this compact layout exports at roughly 10k × 7k pixels.
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
            np.ma.masked_invalid(display_matrices[head]),
            cmap=cmap,
            norm=norm,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
            extent=(-0.5, columns - 0.5, rows - 0.5, -0.5),
        )
        row, column = divmod(index, grid_columns)
        ax.set_xlim(-0.5, columns - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks(sparse_ticks(columns))
        ax.set_yticks(sparse_ticks(rows))
        draw_chunk_boundaries(ax, row_bounds, history_bounds)
        ax.tick_params(
            axis="both",
            pad=0.7,
            labelsize=7,
            length=2.5,
            labelbottom=row == grid_rows - 1 and column % 2 == 0,
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
    ticks, labels = probability_colorbar_ticks(vmax)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(labels)
    colorbar.ax.tick_params(labelsize=7, pad=2.0, length=2.5)
    colorbar.set_label("Raw VV probability (block maximum)", fontsize=8, labelpad=4)
    colorbar.outline.set_linewidth(0.75)
    fig.supxlabel("Visible video K token id (past + current chunk)", fontsize=9, y=0.065)
    fig.supylabel("Video Q token id", fontsize=9, x=0.012)
    fig.text(
        0.5,
        0.018,
        f"Layer {layer} · joint-softmax VV slice · {block_size}×{block_size} block-max · PowerNorm γ={power_gamma:g}",
        ha="center",
        fontsize=10,
    )

    saved = []
    method = render_method_suffix(block_size, power_gamma)
    stem = save_dir / f"layer{layer:02d}_vv_attention_{method}_4x6"
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
        saved.append(path)
    plt.close(fig)
    return saved


def render_experiment(
    project_root: Path,
    attention_link: str = DEFAULT_ATTENTION_LINK,
    experiment_slug: str | None = None,
    attention_step: str | None = None,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    layer_grid: bool = False,
    grid_only: bool = False,
    formats: Sequence[str] = ("png", "pdf"),
    color_vmax: float | None = None,
    dpi: int = 300,
    block_size: int = DEFAULT_BLOCK_MAX_SIZE,
    power_gamma: float = DEFAULT_POWER_GAMMA,
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
    figures_root = project_root / "figures" / WORKSET_NAME
    small_paths = []
    grid_paths = []
    layer_vmaxes = {}
    for layer in layers:
        matrices = {head: load_head_matrix(matrix_dir, shape, layer, head) for head in heads}
        layer_vmax = layer_probability_vmax(matrices, color_vmax, power_gamma)
        layer_vmaxes[layer] = layer_vmax
        layer_dir = output_dir(figures_root, experiment_dir, layer, attention_step)
        if not grid_only:
            for head in heads:
                small_paths.extend(
                    render_head(
                        matrices[head],
                        layer,
                        head,
                        layer_vmax,
                        row_bounds,
                        history_bounds,
                        layer_dir,
                        formats=formats,
                        dpi=dpi,
                        block_size=block_size,
                        power_gamma=power_gamma,
                    )
                )
        if layer_grid:
            grid_paths.extend(
                render_layer_grid(
                    matrices,
                    layer,
                    heads,
                    layer_vmax,
                    row_bounds,
                    history_bounds,
                    layer_dir,
                    formats=formats,
                    dpi=dpi,
                    block_size=block_size,
                    power_gamma=power_gamma,
                )
            )
        suffix = " + 4x6 grid" if layer_grid else ""
        print(f"rendered layer {layer}: {len(heads)} heads{suffix}; shared vmax={layer_vmax:.6g}", flush=True)

    manifest = {
        "experiment": experiment_dir.name,
        "attention_link": attention_link,
        "attention_step": attention_step,
        "scheduler_timestep": step_summary.get("scheduler_timestep"),
        "logical_shape": list(shape),
        "layers": list(layers),
        "heads": list(heads),
        "orientation": {
            "rows": "video Q tokens; small-to-large from top to bottom",
            "columns": "visible video K tokens (past + current chunk); small-to-large left to right",
        },
        "future_columns": "gray / NaN",
        "pooling": {
            "method": "block maximum of raw joint-softmax VV probabilities",
            "block_size": [block_size, block_size],
            "display_shape": [int(np.ceil(shape[0] / block_size)), int(np.ceil(shape[1] / block_size))],
        },
        "cmap": "white_to_red (0 = pure white, larger values = deeper red)",
        "normalization": {"type": "PowerNorm", "gamma": power_gamma, "vmin": 0.0},
        "colorbar_scope": "identical across all selected heads within each layer",
        "scale_source": (
            "explicit shared maximum for every layer"
            if color_vmax is not None
            else "exact finite maximum across selected heads in each layer; no percentile clipping"
        ),
        "vmin": 0.0,
        "layer_vmax": {str(layer): vmax for layer, vmax in layer_vmaxes.items()},
        "dpi": dpi,
        "grid_only": grid_only,
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
    attention_link: str = DEFAULT_ATTENTION_LINK,
    experiment_slug: str | None = None,
    attention_steps: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    layer_grid: bool = False,
    grid_only: bool = False,
    formats: Sequence[str] = ("png", "pdf"),
    color_vmax: float | None = None,
    dpi: int = 300,
    block_size: int = DEFAULT_BLOCK_MAX_SIZE,
    power_gamma: float = DEFAULT_POWER_GAMMA,
) -> dict[str, dict]:
    """Render selected steps with one shared colorbar per layer."""
    experiment_dir, summary = locate_experiment(project_root, attention_link, experiment_slug)
    selection = DEFAULT_ATTENTION_STEPS if attention_steps is None else attention_steps
    steps = resolve_attention_steps(selection, discover_attention_steps(experiment_dir))
    shape, available_layers, num_heads, _, _ = metadata(experiment_dir, summary, steps[0])
    for step in steps[1:]:
        step_shape, step_layers, step_heads, _, _ = metadata(experiment_dir, summary, step)
        if (step_shape, step_layers, step_heads) != (shape, available_layers, num_heads):
            raise ValueError(f"Attention metadata differs across steps; mismatch at {step}")
    layers = resolve_selection(layers, available_layers)
    heads = resolve_selection(heads, range(num_heads))

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
            grid_only=grid_only,
            formats=formats,
            color_vmax=color_vmax,
            dpi=dpi,
            block_size=block_size,
            power_gamma=power_gamma,
        )
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--attention-link", default=DEFAULT_ATTENTION_LINK)
    parser.add_argument("--experiment-slug", default=None)
    parser.add_argument(
        "--attention-step",
        action="append",
        default=None,
        help="Repeat for selected step directories; default: 0/4/9/14/19/24",
    )
    parser.add_argument("--layer", type=int, action="append", default=None, help="Repeat for selected layers; default: all")
    parser.add_argument("--head", type=int, action="append", default=None, help="Repeat for selected heads; default: all")
    parser.add_argument("--grid", action="store_true", help="Also render one 4x6 figure per selected layer")
    parser.add_argument("--grid-only", action="store_true", help="Render only the 4x6 layer grid, not 24 individual heads")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=("png", "pdf"))
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional fixed maximum for every layer; default: exact maximum shared by all selected heads within each layer",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--block-max", type=int, default=DEFAULT_BLOCK_MAX_SIZE, help="Block side length for max pooling; default: 6")
    parser.add_argument("--power-gamma", type=float, default=DEFAULT_POWER_GAMMA, help="PowerNorm gamma; default: 0.25")
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
        layer_grid=args.grid or args.grid_only,
        grid_only=args.grid_only,
        formats=args.formats,
        color_vmax=args.vmax,
        dpi=args.dpi,
        block_size=args.block_max,
        power_gamma=args.power_gamma,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
