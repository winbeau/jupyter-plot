#!/usr/bin/env python
"""阈值敏感性扫描 + 逐层分布。

两件事：

1. **检验 H2 推出的可证伪预测。** H2 说标签翻转跟的是判据边界，不是底层注意力
   不稳（实测 corr(翻转率, 到阈值距离) = -0.264，corr(翻转率, ICC) = -0.087）。
   若成立，把阈值挪到统计量分布的稀疏处，翻转率应下降。所以这里不只扫阈值，
   还把 sign-rate 与周期的**分布本身**画出来 —— 不看分布，扫阈值就只是调参。

2. **逐层分布**：评审 E1 明确要的四个量之一（逐头一致率 / per-class F1 /
   3x3 转移矩阵 / **按层分布**）。

分类规则与阈值取值见 `head_classifier.py` 的文档（论文 §4.1 / §5.1 / 附录 B.4）。
每个 (run, layer, head) 的 sign rate、周期、均值只算一次并缓存，之后扫任意
(alpha, beta) 组合都是纯 numpy。

    uv run python notebooks/26Jul26-PromptSets-attention/threshold_sweep_and_layers.py \\
        --runs data/26Jul26-PromptSets-attention/fetv128 --limit 64
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parents[1]
sys.path.insert(0, str(NOTEBOOK_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "26May4-PyramidForcing-multihead"))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "26Mar26-PyramidForcing-frames72"))

import spectral_analysis_utils as sau  # noqa: E402
import attention_plot_utils as apu  # noqa: E402
from head_classifier import BEST_LABELS, history_sequences  # noqa: E402

NUM_LAYERS, NUM_HEADS = 30, 12
ANCHOR, WAVE, VEIL = 1, -1, 2
NAMES = {ANCHOR: "Anchor", WAVE: "Wave", VEIL: "Veil"}
COLORS = {ANCHOR: "#1f4e79", WAVE: "#c0504d", VEIL: "#9bbb59"}
OUT_FIG = PROJECT_ROOT / "figures" / "26Jul26-PromptSets-attention"
OUT_DATA = PROJECT_ROOT / "data" / "26Jul26-PromptSets-attention"


def precompute(run_dirs, source, window):
    """返回 sign_rate / period / mean，形状均为 [R, L, H]。只遍历一次磁盘。"""
    r = len(run_dirs)
    sr = np.zeros((r, NUM_LAYERS, NUM_HEADS))
    pe = np.full((r, NUM_LAYERS, NUM_HEADS), np.nan)
    mu = np.zeros((r, NUM_LAYERS, NUM_HEADS))
    for ri, d in enumerate(run_dirs):
        for layer in range(NUM_LAYERS):
            payload = torch.load(d / f"layer{layer}.pt", map_location="cpu", weights_only=False)
            seqs = history_sequences(payload, source, window)
            for h in range(NUM_HEADS):
                a = seqs[h]
                sr[ri, layer, h] = float((a > 0).mean())
                mu[ri, layer, h] = float(a.mean())
                try:
                    pe[ri, layer, h] = float(
                        sau.compute_period_spectrum(a)["folded_fft_top_candidates"][0]["period"]
                    )
                except Exception:
                    pass
        if (ri + 1) % 8 == 0 or ri == 0:
            print(f"  precomputed {ri + 1}/{r} runs")
    return sr, pe, mu


def classify(sr, pe, mu, alpha, beta):
    """向量化的 §4.1 规则。输入 [R,L,H]，输出同形状的标签。"""
    lab = np.zeros_like(sr, dtype=int)
    anchor_gate = sr >= alpha
    veil_gate = (1.0 - sr) >= alpha
    wave_gate = (~anchor_gate) & (~veil_gate) & np.isfinite(pe) & (pe < beta)
    fallback = (~anchor_gate) & (~veil_gate) & (~wave_gate)
    lab[anchor_gate] = ANCHOR
    lab[veil_gate] = VEIL
    lab[wave_gate] = WAVE
    lab[fallback & (mu > 0)] = ANCHOR
    lab[fallback & (mu <= 0)] = VEIL
    return lab


def majority(lab):
    """[R,L,H] -> [L,H]，平票取标签值最小者。"""
    out = np.zeros(lab.shape[1:], dtype=int)
    for l in range(lab.shape[1]):
        for h in range(lab.shape[2]):
            c = Counter(lab[:, l, h].tolist())
            out[l, h] = max(c.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return out


def sweep(sr, pe, mu, ref, alphas, betas):
    flip = np.zeros((len(alphas), len(betas)))
    agree = np.zeros_like(flip)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            lab = classify(sr, pe, mu, a, b)
            vote = majority(lab)
            flip[i, j] = float((lab != vote[None]).mean())
            agree[i, j] = float((vote == ref).mean())
    return flip, agree


def plot_distributions(sr, pe, alpha, beta, path):
    """统计量的分布 + 阈值位置。H2 预测的检验就看阈值落在密集处还是稀疏处。"""
    apu.apply_attention_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))

    # sign rate：判据比较的是 max(r_pos, r_neg) 与 alpha
    m = np.maximum(sr, 1 - sr).ravel()
    axes[0].hist(m, bins=60, color="#1f4e79", alpha=0.85, linewidth=0)
    axes[0].axvline(alpha, color="#c0504d", linewidth=1.2)
    axes[0].text(alpha, axes[0].get_ylim()[1] * 0.92, f" α={alpha}", color="#c0504d", fontsize=6)
    axes[0].set_xlabel("max(r_pos, r_neg)")
    axes[0].set_ylabel("head-run count")
    axes[0].set_title("sign-rate statistic vs its gate", fontsize=6)

    finite = pe[np.isfinite(pe)].ravel()
    axes[1].hist(finite, bins=60, range=(0, 25), color="#1f4e79", alpha=0.85, linewidth=0)
    axes[1].axvline(beta, color="#c0504d", linewidth=1.2)
    axes[1].text(beta, axes[1].get_ylim()[1] * 0.92, f" β={beta}", color="#c0504d", fontsize=6)
    axes[1].set_xlabel("dominant period (frames)")
    axes[1].set_title("period statistic vs its gate", fontsize=6)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"  -> {path}.png/.pdf")
    return m, finite


def plot_sweep(flip, agree, alphas, betas, alpha0, beta0, path):
    apu.apply_attention_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for ax, data, title, cmap in (
        (axes[0], flip, "label flip rate (lower = more stable)", "magma_r"),
        (axes[1], agree, "agreement with best_labels.csv", "viridis"),
    ):
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap,
                       extent=[betas[0], betas[-1], alphas[0], alphas[-1]])
        ax.scatter([beta0], [alpha0], marker="+", s=40, c="white", linewidths=1.0)
        ax.set_xlabel("period threshold β")
        ax.set_title(title, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=5)
    axes[0].set_ylabel("sign-rate threshold α")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"  -> {path}.png/.pdf")


def plot_layers(vote, ref, lab, path):
    """逐层：一致率、翻转率、三类构成（我们 vs 参考）。"""
    apu.apply_attention_plot_style()
    layers = np.arange(NUM_LAYERS)
    agree_l = (vote == ref).mean(axis=1)
    flip_l = (lab != vote[None]).mean(axis=(0, 2))

    fig, axes = plt.subplots(3, 1, figsize=(6.6, 5.0), sharex=True)
    axes[0].bar(layers, agree_l, color="#1f4e79", width=0.72)
    axes[0].axhline((vote == ref).mean(), color="#c0504d", linewidth=0.9, linestyle="--")
    axes[0].set_ylabel("agreement")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("per-layer agreement with best_labels.csv (dashed = overall)", fontsize=6)

    axes[1].bar(layers, flip_l, color="#c0504d", width=0.72)
    axes[1].set_ylabel("flip rate")
    axes[1].set_title("per-layer label flip rate across prompts", fontsize=6)

    bottom_ours = np.zeros(NUM_LAYERS)
    bottom_ref = np.zeros(NUM_LAYERS)
    for cls in (ANCHOR, WAVE, VEIL):
        ours = (vote == cls).sum(axis=1)
        refs = (ref == cls).sum(axis=1)
        axes[2].bar(layers - 0.19, ours, bottom=bottom_ours, width=0.36,
                    color=COLORS[cls], label=f"{NAMES[cls]} (ours)")
        axes[2].bar(layers + 0.19, refs, bottom=bottom_ref, width=0.36,
                    color=COLORS[cls], alpha=0.45, label=f"{NAMES[cls]} (ref)")
        bottom_ours += ours
        bottom_ref += refs
    axes[2].set_ylabel("heads")
    axes[2].set_xlabel("layer")
    axes[2].set_title("per-layer class composition, left = ours / right = reference", fontsize=6)
    axes[2].legend(fontsize=4.5, ncol=3, loc="upper right", frameon=False)
    axes[2].set_xticks(range(0, NUM_LAYERS, 2))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"  -> {path}.png/.pdf")
    return agree_l, flip_l


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.80)
    p.add_argument("--beta", type=float, default=6.4)
    p.add_argument("--source", choices=["last_frame", "last_block"], default="last_frame")
    p.add_argument("--window", default="0,69")
    p.add_argument("--tag", default="fetv128")
    args = p.parse_args()

    lo, hi = args.window.split(",")
    window = (int(lo), None if hi.strip().lower() == "none" else int(hi))
    dirs = sorted(d for d in args.runs.glob("run_*") if d.is_dir())[:args.limit]
    if not dirs:
        sys.exit(f"no run_* under {args.runs}")
    ref = np.loadtxt(BEST_LABELS, delimiter=",", dtype=int)

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    print(f"precomputing statistics for {len(dirs)} runs (source={args.source}, window={window})")
    sr, pe, mu = precompute(dirs, args.source, window)

    print("\ndistributions:")
    signstat, periods = plot_distributions(sr, pe, args.alpha, args.beta,
                                           OUT_FIG / f"threshold_distributions_{args.tag}")

    alphas = np.round(np.arange(0.60, 0.961, 0.025), 4)
    betas = np.round(np.arange(5.0, 9.01, 0.2), 4)
    print(f"\nsweeping {len(alphas)} alpha x {len(betas)} beta ...")
    flip, agree = sweep(sr, pe, mu, ref, alphas, betas)
    plot_sweep(flip, agree, alphas, betas, args.alpha, args.beta,
               OUT_FIG / f"threshold_sweep_{args.tag}")

    ai = int(np.argmin(np.abs(alphas - args.alpha)))
    bi = int(np.argmin(np.abs(betas - args.beta)))
    fi, fj = np.unravel_index(np.argmin(flip), flip.shape)
    gi, gj = np.unravel_index(np.argmax(agree), agree.shape)
    print(f"\npaper setting  alpha={alphas[ai]} beta={betas[bi]}: "
          f"flip {flip[ai, bi]:.4f}  agreement {agree[ai, bi]:.4f}")
    print(f"min flip       alpha={alphas[fi]} beta={betas[fj]}: "
          f"flip {flip[fi, fj]:.4f}  agreement {agree[fi, fj]:.4f}")
    print(f"max agreement  alpha={alphas[gi]} beta={betas[gj]}: "
          f"flip {flip[gi, gj]:.4f}  agreement {agree[gi, gj]:.4f}")

    # H2 的检验：阈值处的统计量密度，与翻转率是否同向
    dens_alpha = float(((signstat > args.alpha - 0.02) & (signstat < args.alpha + 0.02)).mean())
    best_alpha = alphas[fi]
    dens_best = float(((signstat > best_alpha - 0.02) & (signstat < best_alpha + 0.02)).mean())
    print(f"\nH2 check: fraction of head-runs within +-0.02 of the sign gate")
    print(f"  at paper alpha={args.alpha}: {dens_alpha:.4f}")
    print(f"  at min-flip alpha={best_alpha}: {dens_best:.4f}")

    print("\nper-layer distribution:")
    lab = classify(sr, pe, mu, args.alpha, args.beta)
    vote = majority(lab)
    agree_l, flip_l = plot_layers(vote, ref, lab, OUT_FIG / f"per_layer_distribution_{args.tag}")

    out_csv = OUT_DATA / f"threshold_sweep_{args.tag}.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["alpha", "beta", "flip_rate", "agreement_with_reference"])
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                w.writerow([a, b, f"{flip[i, j]:.6f}", f"{agree[i, j]:.6f}"])
    layer_csv = OUT_DATA / f"per_layer_labels_{args.tag}.csv"
    with layer_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "agreement", "flip_rate",
                    "ours_anchor", "ours_wave", "ours_veil",
                    "ref_anchor", "ref_wave", "ref_veil"])
        for l in range(NUM_LAYERS):
            w.writerow([l, f"{agree_l[l]:.6f}", f"{flip_l[l]:.6f}"]
                       + [int((vote[l] == c).sum()) for c in (ANCHOR, WAVE, VEIL)]
                       + [int((ref[l] == c).sum()) for c in (ANCHOR, WAVE, VEIL)])
    print(f"\nwrote {out_csv}\nwrote {layer_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
