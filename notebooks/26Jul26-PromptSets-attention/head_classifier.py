#!/usr/bin/env python
"""按论文 §4.1 复现 Tri-Pattern 头分类，产出 E1 要的标签层统计量。

规则逐条来自投稿版正文，不是猜的：

  a(t) = A_T[T, t],  t = 1..T-1        最后一帧对历史帧的 pre-softmax 分数
  r_pos = mean(1{a(t) > 0}),  r_neg = 1 - r_pos
    r_pos >= alpha            -> Anchor (label  1)
    r_neg >= alpha            -> Veil   (label  2)
  其余头做频域周期估计:
    P = Period(FFT(preprocess(a)));  P < beta  -> Wave (label -1)
  仍未分类的 mean-score fallback:
    mean(a) > 0 -> Anchor, 否则 -> Veil

alpha = 0.80、beta = 6.4 见正文 5.1（"The sign-rate and period thresholds are set
to 80% and 6.4"）与附录 B.4 的双阈值消融。周期估计复用
`26May4-PyramidForcing-multihead/spectral_analysis_utils.compute_period_spectrum`
（一阶差分 -> 去 DC -> 加窗 -> rFFT -> 谐波折叠 -> top-1），即产出
`best_labels.csv` 的那条流水线。

**a(t) 取哪一行是一个已知的口径分歧。** 论文写的是最后一帧那一行
（`full_frame_attention[:, T-1, :]`）；`jupyter-plot` 的 loader 优先读
`last_frame_attention_per_head`，缺失时才回退到 `last_block_frame_attention`
（最后一个 block 三个 query 帧的均值）。本仓库当前产出的 .pt 只有后者，因此一直
走回退分支。`--source` 让两者都能跑，差异是可测量的，不必假设。

    uv run python notebooks/26Jul26-PromptSets-attention/head_classifier.py \\
        --runs data/26Jul26-PromptSets-attention/fetv128 --limit 64
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "26May4-PyramidForcing-multihead"))

import spectral_analysis_utils as sau  # noqa: E402

NUM_LAYERS, NUM_HEADS = 30, 12
ANCHOR, WAVE, VEIL = 1, -1, 2
NAMES = {ANCHOR: "Anchor", WAVE: "Wave", VEIL: "Veil"}
BEST_LABELS = PROJECT_ROOT.parent / "Pyramid-Forcing-Preview" / "configs" / "head_configs" / "best_labels.csv"


def history_sequences(payload, source, window):
    """返回 [num_heads, L]：每个头的历史注意力序列 a(t)。"""
    if source == "last_frame":
        full = payload["full_frame_attention"].float().numpy()      # [H, T, T]
        seq = full[:, full.shape[1] - 1, :]                          # 最后一帧那一行
    else:
        seq = payload["last_block_frame_attention"].float().numpy()  # [H, T]
    lo, hi = window
    hi = seq.shape[1] if hi is None else hi
    return seq[:, lo:hi]


def classify_head(a, alpha, beta):
    """返回 (label, reason)。reason 记录是哪条规则定的类，用于诊断边界头。"""
    r_pos = float((a > 0).mean())
    if r_pos >= alpha:
        return ANCHOR, "sign_pos"
    if (1.0 - r_pos) >= alpha:
        return VEIL, "sign_neg"
    try:
        spectrum = sau.compute_period_spectrum(a)
        period = float(spectrum["folded_fft_top_candidates"][0]["period"])
    except Exception:
        period = float("nan")
    if np.isfinite(period) and period < beta:
        return WAVE, "period"
    return (ANCHOR, "fallback_pos") if float(a.mean()) > 0 else (VEIL, "fallback_neg")


def classify_run(run_dir, alpha, beta, source, window):
    labels = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=int)
    reasons = np.empty((NUM_LAYERS, NUM_HEADS), dtype=object)
    for layer in range(NUM_LAYERS):
        payload = torch.load(run_dir / f"layer{layer}.pt", map_location="cpu", weights_only=False)
        seqs = history_sequences(payload, source, window)
        for head in range(NUM_HEADS):
            labels[layer, head], reasons[layer, head] = classify_head(seqs[head], alpha, beta)
    return labels, reasons


def majority_vote(stack):
    """[R, L, H] -> [L, H]。平票时取标签值最小者，保证确定性。"""
    out = np.zeros(stack.shape[1:], dtype=int)
    for l in range(stack.shape[1]):
        for h in range(stack.shape[2]):
            counts = Counter(stack[:, l, h].tolist())
            out[l, h] = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return out


def per_class_f1(ref, pred):
    rows = {}
    for lab in (ANCHOR, WAVE, VEIL):
        tp = int(((pred == lab) & (ref == lab)).sum())
        fp = int(((pred == lab) & (ref != lab)).sum())
        fn = int(((pred != lab) & (ref == lab)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec and np.isfinite(prec + rec) else float("nan")
        rows[lab] = {"support": int((ref == lab).sum()), "tp": tp, "precision": prec,
                     "recall": rec, "f1": f1}
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, required=True, help="含 run_* 的目录")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.80, help="sign-rate 阈值（论文 80%%）")
    p.add_argument("--beta", type=float, default=6.4, help="period 阈值（论文 6.4）")
    p.add_argument("--source", choices=["last_frame", "last_block"], default="last_frame",
                   help="last_frame = 论文定义的 A_T[T, :]；last_block = 现有 .pt 里的块均值")
    p.add_argument("--window", default="0,69",
                   help="a(t) 的下标窗口 lo,hi（hi 可为 none）。论文写 t=1..T-1；"
                        "jupyter-plot 的周期分析用 0,69")
    args = p.parse_args()

    lo, hi = args.window.split(",")
    window = (int(lo), None if hi.strip().lower() == "none" else int(hi))

    dirs = sorted(d for d in args.runs.glob("run_*") if d.is_dir())
    if args.limit:
        dirs = dirs[:args.limit]
    if not dirs:
        sys.exit(f"no run_* under {args.runs}")

    print(f"runs   : {len(dirs)} from {args.runs}")
    print(f"rule   : alpha={args.alpha}  beta={args.beta}  source={args.source}  window={window}")

    stack = np.stack([classify_run(d, args.alpha, args.beta, args.source, window)[0] for d in dirs])
    voted = majority_vote(stack)

    counts = Counter(voted.ravel().tolist())
    print(f"\nmajority-vote composition: " +
          "  ".join(f"{NAMES[k]}={counts.get(k, 0)}" for k in (ANCHOR, WAVE, VEIL)))

    # 逐 run 与多数投票的一致率
    agree = np.array([(stack[i] == voted).mean() for i in range(len(dirs))])
    print(f"per-run agreement vs our own majority vote: "
          f"mean {agree.mean():.4f}  median {np.median(agree):.4f}  "
          f"p5 {np.percentile(agree, 5):.4f}  min {agree.min():.4f}")

    if not BEST_LABELS.exists():
        print(f"\n{BEST_LABELS} not found; skipping the comparison against the published labels.")
        return 0

    ref = np.loadtxt(BEST_LABELS, delimiter=",", dtype=int)
    print(f"\nreference best_labels.csv composition: " +
          "  ".join(f"{NAMES[k]}={int((ref == k).sum())}" for k in (ANCHOR, WAVE, VEIL)))
    print(f"majority vote vs reference: per-head agreement {float((voted == ref).mean()):.4f} "
          f"({int((voted == ref).sum())}/360)")

    per_run_ref = np.array([(stack[i] == ref).mean() for i in range(len(dirs))])
    print(f"per-run agreement vs reference: mean {per_run_ref.mean():.4f}  "
          f"median {np.median(per_run_ref):.4f}  min {per_run_ref.min():.4f}")

    print("\nper-class F1 (reference as ground truth, our majority vote as prediction):")
    print(f"{'class':>8}{'support':>9}{'precision':>11}{'recall':>9}{'f1':>8}")
    for lab, r in per_class_f1(ref, voted).items():
        print(f"{NAMES[lab]:>8}{r['support']:>9}{r['precision']:>11.3f}{r['recall']:>9.3f}{r['f1']:>8.3f}")

    print("\n3x3 transition matrix (rows = reference, cols = ours), row-normalised:")
    order = [ANCHOR, WAVE, VEIL]
    header = "ref/ours"
    print(f"{header:>10}" + "".join(f"{NAMES[c]:>9}" for c in order))
    for r in order:
        tot = int((ref == r).sum())
        cells = [int(((ref == r) & (voted == c)).sum()) for c in order]
        print(f"{NAMES[r]:>10}" + "".join(f"{c / tot:>9.3f}" if tot else f"{'-':>9}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
