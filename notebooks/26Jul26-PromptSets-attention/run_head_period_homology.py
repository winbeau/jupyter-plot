#!/usr/bin/env python
"""跨 prompt 集的 head period homology + 分类稳定性对照。

对 FETV-128 / Step-128 / ChronoMagic-150 各跑一遍
`spectral_analysis_utils.run_256prompts_distribution`（它对 prompt 数是参数化的，
不是写死 256），然后回答两个问题：

  1. **集合内稳定性** —— 同一个头在同一集合的不同 prompt 之间，主周期抖多少
     （`period_std`）。
  2. **跨集合一致性** —— 同一个头在三个 prompt 分布下拿到的主周期是否一致，
     以及用仓库自己的 6-cycle 判据（`CYCLE6_PERIOD_RANGE` / `CYCLE6_STD_MAX`）
     判出来的「振荡头」集合是否稳定。第 2 点直接对应论文里 Wave 头的划分。

判据一律沿用 `spectral_analysis_utils` 里既有的常量，不另立标准。

    uv run python notebooks/26Jul26-PromptSets-attention/run_head_period_homology.py
"""

import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parents[1]
UTILS_DIR = PROJECT_ROOT / "notebooks" / "26May4-PyramidForcing-multihead"
sys.path.insert(0, str(UTILS_DIR))

import spectral_analysis_utils as sau  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "26Jul26-PromptSets-attention"
OUT_DIR = PROJECT_ROOT / "figures" / "26Jul26-PromptSets-attention"

# (目录名, prompt 数, 展示名)
SETS = [
    ("fetv128", 128, "FETV-128"),
    ("step128", 128, "Step-128"),
    ("chronomagic150", 150, "ChronoMagic-150"),
]

NUM_LAYERS, NUM_HEADS = 30, 12
# 论文里的头分类矩阵。1=Anchor/stable, -1=Wave/oscillating, 2=Veil/stable_sparse
BEST_LABELS = (
    PROJECT_ROOT.parent / "Pyramid-Forcing-Preview" / "configs" / "head_configs" / "best_labels.csv"
)


def csv_path_for(name):
    # 汇总表进 data/（AGENTS.md：data/ 是可复用表格的唯一真源），
    # 图进 figures/。注意 figures/**/*.csv 是被 gitignore 的。
    return DATA_ROOT / f"head_period_homology_{name}_firstdiff_folded_top1_0_68.csv"


def run_one(name, num_prompts, label):
    print("=" * 72)
    print(f"{label}  ({num_prompts} prompts)  <- {DATA_ROOT / name}")
    print("=" * 72)
    csv_path = csv_path_for(name)
    result = sau.run_256prompts_distribution(
        batch_prompts_dir=DATA_ROOT / name,
        batch_num_prompts=num_prompts,
        batch_num_layers=NUM_LAYERS,
        batch_num_heads=NUM_HEADS,
        csv_path=csv_path,
        representative_figure_path=OUT_DIR / f"head_period_homology_{name}_representative.png",
        uniform_figure_path=OUT_DIR / f"head_period_homology_{name}_uniform.png",
        use_markers=False,
    )
    if result.get("errors"):
        print(f"  !! {len(result['errors'])} load/spectrum errors, first: {result['errors'][0]}")
    return csv_path


def load_rows(csv_path):
    """读回 CSV，返回 {(layer, head): row}。"""
    out = {}
    with Path(csv_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["layer"]), int(row["head"]))
            out[key] = {
                k: (int(v) if k in {"layer", "head", "head_index", "n_prompts", "dominant_period_count"}
                    else float(v))
                for k, v in row.items()
            }
    return out


def is_cycle6(row):
    """仓库自己的 6-cycle 判据：主周期落在 CYCLE6_PERIOD_RANGE 且跨 prompt 抖动够小。"""
    lo, hi = sau.CYCLE6_PERIOD_RANGE
    return lo <= row["period_mean"] <= hi and row["period_std"] <= sau.CYCLE6_STD_MAX


def read_best_labels(path):
    if not path.exists():
        return None
    matrix = np.loadtxt(path, delimiter=",", dtype=int)
    if matrix.shape != (NUM_LAYERS, NUM_HEADS):
        raise ValueError(f"{path}: expected ({NUM_LAYERS},{NUM_HEADS}), got {matrix.shape}")
    return matrix


def main():
    missing = [n for n, _, _ in SETS if not (DATA_ROOT / n).is_dir()]
    if missing:
        sys.exit(f"missing data directories: {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, num_prompts, label in SETS:
        path = csv_path_for(name)
        if not path.exists():
            run_one(name, num_prompts, label)
        else:
            print(f"reusing existing {path}")
        tables[name] = load_rows(path)

    keys = [(l, h) for l in range(NUM_LAYERS) for h in range(NUM_HEADS)]

    # ---------- 1. 集合内稳定性 ----------
    print()
    print("=" * 72)
    print("1. 集合内稳定性 —— 同一头在不同 prompt 之间的主周期抖动")
    print("=" * 72)
    print(f"{'set':<18}{'period_std 中位':>16}{'均值':>10}{'std<=1.5 的头':>16}{'6-cycle 头':>12}")
    for name, _, label in SETS:
        stds = np.array([tables[name][k]["period_std"] for k in keys])
        tight = int((stds <= sau.CYCLE6_STD_MAX).sum())
        c6 = sum(is_cycle6(tables[name][k]) for k in keys)
        print(f"{label:<18}{np.median(stds):>16.3f}{stds.mean():>10.3f}"
              f"{f'{tight}/360 ({tight/3.6:.0f}%)':>16}{f'{c6}/360':>12}")

    # ---------- 2. 跨集合一致性 ----------
    print()
    print("=" * 72)
    print("2. 跨集合一致性 —— 同一头在不同 prompt 分布下拿到同一个周期吗")
    print("=" * 72)
    for (na, _, la), (nb, _, lb) in combinations(SETS, 2):
        a = np.array([tables[na][k]["period_mean"] for k in keys])
        b = np.array([tables[nb][k]["period_mean"] for k in keys])
        diff = np.abs(a - b)
        corr = float(np.corrcoef(a, b)[0, 1])
        within_05 = int((diff <= 0.5).sum())
        within_10 = int((diff <= 1.0).sum())
        print(f"{la:>16} vs {lb:<18} r={corr:+.3f}  "
              f"|Δ|中位={np.median(diff):.3f}  "
              f"≤0.5: {within_05}/360 ({within_05/3.6:.0f}%)  "
              f"≤1.0: {within_10}/360 ({within_10/3.6:.0f}%)")

    # 三集合同时满足 6-cycle 的头
    c6_sets = {name: {k for k in keys if is_cycle6(tables[name][k])} for name, _, _ in SETS}
    names = [n for n, _, _ in SETS]
    inter = set.intersection(*c6_sets.values())
    union = set.union(*c6_sets.values())
    print()
    print(f"6-cycle 头（判据: period_mean∈{sau.CYCLE6_PERIOD_RANGE}, period_std≤{sau.CYCLE6_STD_MAX}）")
    for name, _, label in SETS:
        print(f"  {label:<18}{len(c6_sets[name]):>4} 个")
    print(f"  {'三集合交集':<18}{len(inter):>4} 个")
    print(f"  {'三集合并集':<18}{len(union):>4} 个")
    print(f"  {'Jaccard(交/并)':<18}{len(inter)/len(union) if union else float('nan'):>7.3f}")
    for na, nb in combinations(names, 2):
        i = len(c6_sets[na] & c6_sets[nb])
        u = len(c6_sets[na] | c6_sets[nb])
        print(f"  Jaccard({na}, {nb}) = {i/u if u else float('nan'):.3f}  ({i}/{u})")

    # ---------- 3. 与论文分类矩阵对照 ----------
    labels = read_best_labels(BEST_LABELS)
    print()
    print("=" * 72)
    print("3. 与论文的 best_labels.csv 对照")
    print("=" * 72)
    if labels is None:
        print(f"未找到 {BEST_LABELS}，跳过。")
    else:
        wave = {(l, h) for l in range(NUM_LAYERS) for h in range(NUM_HEADS) if labels[l, h] == -1}
        print(f"best_labels.csv: Anchor(1)={int((labels==1).sum())} "
              f"Wave(-1)={int((labels==-1).sum())} Veil(2)={int((labels==2).sum())}")
        print()
        print(f"{'set':<18}{'6-cycle∩Wave':>14}{'Wave 召回':>12}{'6-cycle 精度':>14}")
        for name, _, label in SETS:
            c6 = c6_sets[name]
            hit = len(c6 & wave)
            rec = hit / len(wave) if wave else float("nan")
            prec = hit / len(c6) if c6 else float("nan")
            print(f"{label:<18}{hit:>14}{rec:>11.1%}{prec:>13.1%}")
        print()
        print("注：6-cycle 只是 Wave 头的一个充分不必要的表征——Wave 的定义是注意力随时间")
        print("    振荡，不要求周期恰好落在 6 附近。上表是对照，不是准确率。")

    # ---------- 落盘 ----------
    merged = DATA_ROOT / "head_period_homology_promptsets_merged.csv"
    with merged.open("w", newline="") as handle:
        cols = ["layer", "head", "head_index"]
        for name, _, _ in SETS:
            cols += [f"{name}_period_mean", f"{name}_period_std", f"{name}_dominant_period",
                     f"{name}_is_cycle6"]
        if labels is not None:
            cols.append("best_label")
        writer = csv.writer(handle)
        writer.writerow(cols)
        for l, h in keys:
            row = [l, h, l * NUM_HEADS + h]
            for name, _, _ in SETS:
                r = tables[name][(l, h)]
                row += [f"{r['period_mean']:.6f}", f"{r['period_std']:.6f}",
                        f"{r['dominant_period']:.6f}", int(is_cycle6(r))]
            if labels is not None:
                row.append(int(labels[l, h]))
            writer.writerow(row)
    print()
    print(f"合并表 -> {merged}")


if __name__ == "__main__":
    main()
