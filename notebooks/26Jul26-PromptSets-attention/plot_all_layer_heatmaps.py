#!/usr/bin/env python
"""绘图阶段：把一次提取的 30 层全部渲染成 12 头三角热力图。

这是「提取 -> 绘图」两阶段流程的第二阶段。第一阶段在
`Pyramid-Forcing-Preview/experiments/extract_attn/run_extraction_streaming.py`，
产出 `<root>/run_%03d/layer<L>.pt`；本脚本只吃那个布局，不关心它是怎么来的，
所以换 prompt、换模型、换集合都不用改代码。

复用 `26Mar26-PyramidForcing-frames72/attention_plot_utils.py` 的
`render_attention_heatmaps`（就是 `26Jul26-CausalRCM-attention/layer15_2d_heatmap_all_heads.png`
那张图的渲染器），只把配色范围换成本批数据的实际范围。

    # 单个 run 的 30 层
    uv run python notebooks/26Jul26-PromptSets-attention/plot_all_layer_heatmaps.py \\
        --run-dir data/26Jul26-PromptSets-attention/fetv128/run_000 \\
        --out-dir figures/26Jul26-PromptSets-attention/fetv128_run000

    # 多个 run 共用一套自适应色标（先扫全局 min/max 再渲染）
    uv run python notebooks/26Jul26-PromptSets-attention/plot_all_layer_heatmaps.py \\
        --run-dir data/26Jul26-PromptSets-attention/fetv128/run_000 \\
        --run-dir data/26Jul26-PromptSets-attention/step128/run_000 \\
        --run-dir data/26Jul26-PromptSets-attention/chronomagic150/run_000 \\
        --out-root figures/26Jul26-PromptSets-attention
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parents[1]
UTILS_DIR = PROJECT_ROOT / "notebooks" / "26Mar26-PyramidForcing-frames72"
sys.path.insert(0, str(UTILS_DIR))

import attention_plot_utils as apu  # noqa: E402

# 参考图（26Jul26-CausalRCM-attention）用的写死范围。想和既有图并排就用它。
FIXED_RANGE = (-18.0, 18.0)


def layer_paths(run_dir, num_layers):
    """按层序返回 .pt 路径，缺任何一层就报错——半套图比没有图更坑。"""
    paths = []
    for layer in range(num_layers):
        path = run_dir / f"layer{layer}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; {run_dir} is not a complete extraction")
        paths.append(path)
    return paths


def scan_range(run_dirs, num_layers, low_pct=1.0, high_pct=99.0, robust=True):
    """扫全部 run 的全部层，给出配色范围。

    **默认走分位数，不是 min/max。** 这批数据的全局 min/max 是 [-36, 100.6]，
    但 |v|>20 的只占 1.26%，且几乎全来自 layer 28/29；用 min/max 定色标会把
    98% 的数据压进色标中段，图基本全白。p1/p99 落在约 [-15.3, 13.9]，正好贴着
    参考图写死的 ±18。

    只统计因果下三角：上三角是未来帧，恒为结构性 0，算进去会把中位数往 0 拽。

    分位数必须在**池化后的全部值**上取。取「逐层 p1 的最小值 / 逐层 p99 的最大值」
    是错的 —— 那等于又把 layer28/29 那两个极端层挑了出来，离群问题原封不动
    （实测会给出 [-28.8, 68.7] 而不是正确的 [-15.3, 13.9]）。
    全部样本 2.8M 个 float32 只有 11 MB，直接全收即可。
    """
    pooled = []
    mask = None
    for run_dir in run_dirs:
        for path in layer_paths(run_dir, num_layers):
            full = torch.load(path, map_location="cpu", weights_only=False)["full_frame_attention"]
            full = full.float().numpy()
            if mask is None or mask.shape[0] != full.shape[1]:
                mask = np.tril(np.ones((full.shape[1], full.shape[2]), dtype=bool))
            pooled.append(full[:, mask].ravel())
    pooled = np.concatenate(pooled)
    if robust:
        lo = float(np.percentile(pooled, low_pct))
        hi = float(np.percentile(pooled, high_pct))
    else:
        lo, hi = float(pooled.min()), float(pooled.max())
    print(f"  pooled {pooled.size/1e6:.1f}M causal values; "
          f"min={pooled.min():.2f} max={pooled.max():.2f} -> using [{lo:.3f}, {hi:.3f}]")
    if not (lo < 0 < hi):
        raise ValueError(
            f"range [{lo:.3f}, {hi:.3f}] does not straddle 0; TwoSlopeNorm needs vmin < 0 < vmax"
        )
    return lo, hi


def render_run(run_dir, out_dir, color_min, color_max, num_layers, tick_positions, save_svg):
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in layer_paths(run_dir, num_layers):
        data = torch.load(path, map_location="cpu", weights_only=False)
        apu.render_attention_heatmaps(
            path,
            save_dir=out_dir,  # 必须绝对路径：相对路径会被解析到 26Mar26-* 目录下
            data=data,
            tick_positions=tick_positions,
            color_min=color_min,
            color_max=color_max,
            save_pdf=True,
            save_png=True,
            save_svg=save_svg,
            show=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", action="append", required=True, type=Path,
                        help="一个提取 run 目录（含 layer0.pt … layerN.pt）。可重复传。")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="单个 run 时的输出目录。多个 run 请用 --out-root。")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="多 run 输出根目录；每个 run 落到 <out-root>/<set>_<run>/。")
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--color-range", default="auto",
                        help="auto = 扫本批数据取 p1/p99（推荐）；minmax = 朴素全局 min/max"
                             "（会被 layer28/29 的离群值吃掉对比度）；fixed18 = 参考图的 ±18；"
                             "或直接给 'MIN,MAX'")
    parser.add_argument("--pct", default="1,99",
                        help="--color-range auto 用的分位数，逗号分隔")
    parser.add_argument("--ticks", default="0,36,71", help="坐标刻度，逗号分隔")
    parser.add_argument("--svg", action="store_true", help="额外存 SVG（体积翻倍，只在要手改矢量图时开）")
    args = parser.parse_args()

    run_dirs = [d.resolve() for d in args.run_dir]
    for d in run_dirs:
        if not d.is_dir():
            sys.exit(f"not a directory: {d}")
    if args.out_dir is None and args.out_root is None:
        sys.exit("pass --out-dir (single run) or --out-root (one or more runs)")
    if len(run_dirs) > 1 and args.out_root is None:
        sys.exit("multiple --run-dir need --out-root")

    tick_positions = tuple(int(v) for v in args.ticks.split(","))

    if args.color_range in ("auto", "minmax"):
        robust = args.color_range == "auto"
        low_pct, high_pct = (float(v) for v in args.pct.split(","))
        kind = f"p{low_pct:g}/p{high_pct:g}" if robust else "min/max"
        print(f"scanning {len(run_dirs)} run(s) x {args.num_layers} layers for the {kind} range...")
        color_min, color_max = scan_range(
            run_dirs, args.num_layers, low_pct=low_pct, high_pct=high_pct, robust=robust
        )
        print(f"{kind} range: [{color_min:.3f}, {color_max:.3f}]")
        if not robust:
            print("  NOTE: min/max is dominated by layer 28/29 outliers; the plots will wash out.")
    elif args.color_range == "fixed18":
        color_min, color_max = FIXED_RANGE
        print(f"using the reference figures' fixed range: [{color_min}, {color_max}]")
    else:
        color_min, color_max = (float(v) for v in args.color_range.split(","))
        print(f"using the supplied range: [{color_min}, {color_max}]")

    for run_dir in run_dirs:
        if args.out_root is not None:
            # data/<workset>/<set>/run_000 -> <out-root>/<set>_run000
            out_dir = args.out_root / f"{run_dir.parent.name}_{run_dir.name.replace('_', '')}"
        else:
            out_dir = args.out_dir
        print(f"\n=== {run_dir} -> {out_dir} ===")
        render_run(run_dir, out_dir, color_min, color_max, args.num_layers, tick_positions, args.svg)

    print(f"\nDone. {len(run_dirs) * args.num_layers} figure(s) at range "
          f"[{color_min:.3f}, {color_max:.3f}].")


if __name__ == "__main__":
    main()
