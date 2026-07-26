#!/usr/bin/env python
"""E1 方差分解：逐头 ICC(1,1)。

设计见 `rebuttal/E1-stability-protocol.md`。三个条件：

  A  变 prompt、固定 seed        -> sigma^2_prompt
  B  固定 prompt、变 seed        -> sigma^2_seed（标尺）
  C  prompt 与 seed 同变         -> sigma^2_total

对每个头，把 N 次运行的因果下三角展平成 X[n, m]（n = 运行，m = cell），
做单因素随机效应方差分析，target = cell、replicate = 运行：

    MSB = N * sum_m (mu_m - mu)^2 / (M - 1)
    MSW = sum_m sum_n (X[n,m] - mu_m)^2 / (M * (N - 1))
    ICC(1,1) = (MSB - MSW) / (MSB + (N - 1) * MSW)

**不要用 `Var_cell(mean_n X) / (Var_cell(...) + mean_cell(Var_n X))`。**
那个式子里的 `Var_cell(mean)` 含有 `sigma^2_e / N` 的成分，会把 ICC 高估，偏的方向
恰好是「看起来更稳定」。`--selftest` 在已知方差分量的合成数据上量过：偏差随 N 减小、
随真实 ICC 降低而变大，最差 +0.0555（N=4、真值 0.50）；N=16 真值 0.50 时 +0.0242。
真实 ICC 到 0.95 以上时两式差别落到 0.001 量级，所以这不是一个会颠覆结论的量级，
但方向固定向上，报进 rebuttal 之前没有理由留着它。ANOVA 形式在同一网格上误差
均 < 0.02。

只取因果下三角：上三角是未来帧，恒为结构性 0，计入会把 cell 间方差灌水、ICC 虚高。

    uv run python notebooks/26Jul26-PromptSets-attention/icc_variance_components.py --selftest
    uv run python notebooks/26Jul26-PromptSets-attention/icc_variance_components.py \\
        --cond A=data/26Jul26-PromptSets-attention/fetv128 \\
        --cond B=<condB 目录> --cond C=<condC 目录>
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

NUM_LAYERS, NUM_HEADS, NUM_FRAMES = 30, 12, 72


def icc1_from_matrix(x_nm):
    """单因素随机效应 ICC(1,1)。

    Args:
        x_nm: [N, M]，N 次运行 × M 个 cell。

    Returns:
        (icc, ms_between, ms_within, var_target, var_error)
        icc 会被截断到 [0, 1]：MSB < MSW 时无偏估计可为负，物理上等价于 0。
    """
    x = np.asarray(x_nm, dtype=np.float64)
    n, m = x.shape
    if n < 2:
        raise ValueError(f"ICC needs at least 2 runs, got {n}")
    cell_mean = x.mean(axis=0)                       # [M]
    grand = cell_mean.mean()
    ms_between = n * ((cell_mean - grand) ** 2).sum() / (m - 1)
    ms_within = ((x - cell_mean) ** 2).sum() / (m * (n - 1))
    denom = ms_between + (n - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom > 0 else 0.0
    var_error = ms_within
    var_target = max((ms_between - ms_within) / n, 0.0)
    return float(np.clip(icc, 0.0, 1.0)), float(ms_between), float(ms_within), var_target, var_error


def icc1_naive(x_nm):
    """有偏的朴素式子，只用于 --selftest 里展示偏差方向。"""
    x = np.asarray(x_nm, dtype=np.float64)
    cell_mean = x.mean(axis=0)
    signal = cell_mean.var()
    noise = x.var(axis=0, ddof=1).mean()
    return float(signal / (signal + noise)) if signal + noise > 0 else 0.0


def causal_mask(num_frames=NUM_FRAMES):
    return np.tril(np.ones((num_frames, num_frames), dtype=bool))


def run_dirs_of(root):
    dirs = sorted(p for p in Path(root).glob("run_*") if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"no run_* directories under {root}")
    return dirs


def load_layer(run_dirs, layer, mask):
    """返回 [N, H, M] float32：N 次运行、H 个头、M 个因果 cell。"""
    stack = []
    for d in run_dirs:
        path = d / f"layer{layer}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; {d} is an incomplete extraction")
        full = torch.load(path, map_location="cpu", weights_only=False)["full_frame_attention"]
        stack.append(full.float().numpy()[:, mask])
    return np.stack(stack, axis=0)


def compute_condition(root, num_layers=NUM_LAYERS):
    mask = causal_mask()
    dirs = run_dirs_of(root)
    icc = np.full((num_layers, NUM_HEADS), np.nan)
    var_t = np.full((num_layers, NUM_HEADS), np.nan)
    var_e = np.full((num_layers, NUM_HEADS), np.nan)
    for layer in range(num_layers):
        block = load_layer(dirs, layer, mask)          # [N, H, M]
        for head in range(NUM_HEADS):
            icc[layer, head], _, _, var_t[layer, head], var_e[layer, head] = icc1_from_matrix(
                block[:, head, :]
            )
        if layer % 10 == 0:
            print(f"  layer {layer:2d}/{num_layers}  ICC median so far "
                  f"{np.nanmedian(icc[:layer + 1]):.4f}")
    return {"n_runs": len(dirs), "icc": icc, "var_target": var_t, "var_error": var_e}


def selftest():
    """在已知方差分量的合成数据上校验，并量化朴素式子的偏差。"""
    rng = np.random.default_rng(0)
    m = 2628                       # 72 帧因果下三角的 cell 数
    print(f"{'N':>5}{'true ICC':>10}{'ANOVA':>10}{'err':>9}{'naive':>10}{'bias':>9}")
    worst = 0.0
    for n in (4, 8, 16, 64, 128):
        for true_icc in (0.5, 0.8, 0.95, 0.99):
            var_e = 1.0
            var_t = true_icc * var_e / (1.0 - true_icc)
            target = rng.normal(0.0, np.sqrt(var_t), size=m)
            x = target[None, :] + rng.normal(0.0, np.sqrt(var_e), size=(n, m))
            est, _, _, _, _ = icc1_from_matrix(x)
            naive = icc1_naive(x)
            print(f"{n:>5}{true_icc:>10.3f}{est:>10.4f}{est - true_icc:>+9.4f}"
                  f"{naive:>10.4f}{naive - true_icc:>+9.4f}")
            if abs(est - true_icc) > 0.02:
                raise AssertionError(f"ANOVA ICC off by {est - true_icc:+.4f} at N={n}, true={true_icc}")
            worst = max(worst, naive - true_icc)
    print(f"\nANOVA estimator within 0.02 of truth everywhere.")
    print(f"Naive formula's worst upward bias across this grid: +{worst:.4f}")
    print("The naive bias is always positive, i.e. it makes the data look more stable than it is.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cond", action="append", default=[],
                        help="NAME=PATH，可重复。例如 A=data/26Jul26-PromptSets-attention/fetv128")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/26Jul26-PromptSets-attention"))
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--delta", type=float, default=0.05,
                        help="H1 判据：ICC_A >= ICC_B - delta。跑之前定死，不许事后改")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return 0
    if not args.cond:
        parser.error("pass --cond NAME=PATH at least once, or --selftest")

    results = {}
    for spec in args.cond:
        if "=" not in spec:
            parser.error(f"--cond needs NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        print(f"=== condition {name}: {path} ===")
        results[name] = compute_condition(path, args.num_layers)
        r = results[name]
        v = r["icc"].ravel()
        print(f"  N={r['n_runs']} runs   ICC median {np.median(v):.4f}  "
              f"mean {v.mean():.4f}  p5 {np.percentile(v, 5):.4f}  min {v.min():.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "icc_variance_components.csv"
    names = list(results)
    with out_csv.open("w", newline="") as handle:
        w = csv.writer(handle)
        cols = ["layer", "head", "head_index"]
        for n in names:
            cols += [f"{n}_icc", f"{n}_var_target", f"{n}_var_error"]
        w.writerow(cols)
        for layer in range(args.num_layers):
            for head in range(NUM_HEADS):
                row = [layer, head, layer * NUM_HEADS + head]
                for n in names:
                    row += [f"{results[n]['icc'][layer, head]:.6f}",
                            f"{results[n]['var_target'][layer, head]:.6g}",
                            f"{results[n]['var_error'][layer, head]:.6g}"]
                w.writerow(row)
    print(f"\nwrote {out_csv}")

    if "A" in results and "B" in results:
        a, b = results["A"]["icc"].ravel(), results["B"]["icc"].ravel()
        med_a, med_b = float(np.median(a)), float(np.median(b))
        passed = med_a >= med_b - args.delta
        print()
        print("H1: prompt-induced variance must not exceed seed-induced variance")
        print(f"    median ICC_A = {med_a:.4f}   median ICC_B = {med_b:.4f}   delta = {args.delta}")
        print(f"    ICC_A >= ICC_B - delta  ->  {'HOLDS' if passed else 'FAILS'}")
        per_head = int((a >= b - args.delta).sum())
        print(f"    per-head: {per_head}/{a.size} heads satisfy it")
    if "C" in results and "A" in results and "B" in results:
        # 可加性检验：sigma^2_C ~= sigma^2_A + sigma^2_B（误差分量相加）
        ea = results["A"]["var_error"].ravel()
        eb = results["B"]["var_error"].ravel()
        ec = results["C"]["var_error"].ravel()
        ratio = ec / np.clip(ea + eb, 1e-12, None)
        print()
        print("Additivity of the error components: var_C / (var_A + var_B)")
        print(f"    median {np.median(ratio):.3f}   p5 {np.percentile(ratio, 5):.3f}   "
              f"p95 {np.percentile(ratio, 95):.3f}")
        print("    near 1 => prompt and seed contribute independently; far from 1 => interaction,")
        print("    which must be reported rather than folded into a single number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
