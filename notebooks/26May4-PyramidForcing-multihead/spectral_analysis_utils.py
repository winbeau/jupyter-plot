# Shared implementation for the spectral analysis notebooks.
# Keep notebook-facing parameters in the notebooks; keep analysis logic here.

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

mpl.rcParams.update({
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
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0,
})

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]

ATTN_DIR = PROJECT_ROOT / "data/26Mar26-PyramidForcing-frames72/eagle"
LAYER = 12
HEAD = 0

# End is exclusive: 0:69 means frames 0-68; None means through the last frame.
FFT_RANGES = [
    {"label": "0-68 first diff", "start": 0, "end": 69},
    {"label": "0-71 first diff", "start": 0, "end": None},
]

SOFTMAX_WINDOW_START = 51
SOFTMAX_WINDOW_END = 72
SOFTMAX_WINDOW_LABEL = "51-71"
SOFTMAX_FFT_RANGES = [
    {"label": f"{SOFTMAX_WINDOW_LABEL} local softmax first diff", "start": 0, "end": None},
]

# Period display range in frames/cycle. Zero frequency is excluded from period plots.
PERIOD_MIN = 2.0
PERIOD_MAX = None

# Response-period selection excludes low-frequency envelope peaks by default.
RESPONSE_PERIOD_MIN = 4.0
RESPONSE_PERIOD_MAX = 18.0
MIN_RESPONSE_CYCLES = 3.0

# ACF verifies FFT candidates; it does not replace FFT period selection.
FFT_TOP_K = 5
ACF_TOL = 2
ACF_MIN = 0.2
NEAR_TIE_RATIO = 1.05
LOW_CYCLE_THRESHOLD = 3.0

# Harmonic folding affects FFT ranking only; raw FFT amplitude remains the plotted/physical value.
HARMONIC_FOLDING_ENABLED = True
HARMONIC_FOLD_MIN_PERIOD = 3.0
HARMONIC_FOLD_MAX_PERIOD_DIVISOR = 6.0
HARMONIC_FOLD_MAX_MULTIPLE = 4
HARMONIC_FOLD_TOL = 0.2
HARMONIC_FOLD_DECAY = 0.5
FFT_DEBUG_TOP_K = 3
SAVE_MAIN_FIGURE = True
MAIN_FIGURE_DPI = 300
MAIN_FIGURE_SIZE = (5.50, 4.20)
OUTPUT_DIR = PROJECT_ROOT / "figures/26May4-PyramidForcing-multihead"
MAIN_FIGURE_PATH = OUTPUT_DIR / f"spectral_analysis_L{LAYER}H{HEAD}_firstdiff_main.png"

REMOVE_DC = True
APPLY_WINDOW = True
SOFTMAX_TEMPERATURE = 1.0

def load_attention(attn_dir, layer, head):
    attn_path = attn_dir / f"layer{layer}.pt"
    payload = torch.load(attn_path, map_location="cpu", weights_only=False)

    if "last_frame_attention_per_head" in payload:
        per_head = payload["last_frame_attention_per_head"]
    elif "last_block_frame_attention" in payload:
        per_head = payload["last_block_frame_attention"]
    else:
        available = ", ".join(sorted(payload.keys()))
        raise KeyError(
            "Expected last_frame_attention_per_head or last_block_frame_attention "
            f"in {attn_path}; available keys: {available}"
        )

    if per_head.ndim != 2:
        raise ValueError(f"Expected per-head attention to be 2D, got shape {tuple(per_head.shape)}")
    if not 0 <= head < per_head.shape[0]:
        raise IndexError(f"HEAD={head} is out of range for {per_head.shape[0]} heads")

    attention = per_head[head].detach().cpu().float().numpy()
    if attention.ndim != 1:
        raise ValueError(f"Expected selected head attention to be 1D, got shape {attention.shape}")

    frame_idx = np.arange(attention.shape[0])
    return attention, frame_idx, attn_path

def normalize_range(start, end, size):
    start = 0 if start is None else start
    end = size if end is None else end
    start_norm = start if start >= 0 else size + start
    end_norm = end if end >= 0 else size + end
    if start_norm < 0 or end_norm > size or start_norm >= end_norm:
        raise ValueError(f"Invalid FFT range {start}:{end} for {size} frames")
    return start, end, start_norm, end_norm

def softmax_1d(values, temperature=1.0):
    if temperature <= 0:
        raise ValueError("SOFTMAX_TEMPERATURE must be positive")
    scaled = values.astype(np.float64) / temperature
    scaled = scaled - np.max(scaled)
    exp_values = np.exp(scaled)
    return exp_values / exp_values.sum()


def local_softmax_1d(values, start, end, temperature=1.0):
    _, _, start_norm, end_norm = normalize_range(start, end, values.size)
    softmax_window = softmax_1d(values[start_norm:end_norm], temperature=temperature)
    softmax_full = np.full(values.shape, np.nan, dtype=np.float64)
    softmax_full[start_norm:end_norm] = softmax_window
    return softmax_full, softmax_window

def preprocess_signal(sequence, remove_dc=True, apply_window=True):
    if sequence.ndim != 1 or sequence.size < 3:
        raise ValueError(f"First-difference FFT input must be 1D with at least 3 frames, got shape {sequence.shape}")

    signal = np.diff(sequence.astype(np.float64, copy=True))
    if remove_dc:
        signal = signal - signal.mean()
    if apply_window:
        signal = signal * np.hanning(signal.size)
    return signal

def compute_harmonic_folded_spectrum(period, amplitude, signal_size):
    if HARMONIC_FOLD_MAX_PERIOD_DIVISOR is None:
        max_period = np.inf
    elif HARMONIC_FOLD_MAX_PERIOD_DIVISOR <= 0:
        raise ValueError("HARMONIC_FOLD_MAX_PERIOD_DIVISOR must be positive")
    else:
        max_period = signal_size / HARMONIC_FOLD_MAX_PERIOD_DIVISOR

    folded = amplitude.astype(np.float64, copy=True)
    folding_mask = (period >= HARMONIC_FOLD_MIN_PERIOD) & (period <= max_period)
    for idx in np.flatnonzero(folding_mask):
        base = period[idx]
        for multiple in range(2, HARMONIC_FOLD_MAX_MULTIPLE + 1):
            target = multiple * base
            harmonic_idx = int(np.argmin(np.abs(period - target)))
            if abs(period[harmonic_idx] - target) / target < HARMONIC_FOLD_TOL:
                folded[idx] += amplitude[harmonic_idx] * (HARMONIC_FOLD_DECAY ** (multiple - 1))
    return folded


def compute_period_spectrum(sequence):
    signal = preprocess_signal(sequence, remove_dc=REMOVE_DC, apply_window=APPLY_WINDOW)
    freq = np.fft.rfftfreq(signal.size, d=1.0)
    amplitude = np.abs(np.fft.rfft(signal))
    nonzero = freq > 0
    if not np.any(nonzero):
        raise ValueError("FFT produced no nonzero frequency bins")

    freq_bin = np.flatnonzero(nonzero)
    period = 1.0 / freq[nonzero]
    period_amp = amplitude[nonzero]
    order = np.argsort(period)
    period, period_amp, freq_bin = period[order], period_amp[order], freq_bin[order]

    display_mask = np.ones_like(period, dtype=bool)
    if PERIOD_MIN is not None:
        display_mask &= period >= PERIOD_MIN
    if PERIOD_MAX is not None:
        display_mask &= period <= PERIOD_MAX
    if not np.any(display_mask):
        raise ValueError(f"Period display range [{PERIOD_MIN}, {PERIOD_MAX}] contains no FFT bins")

    folded_amp = (
        compute_harmonic_folded_spectrum(period, period_amp, signal.size)
        if HARMONIC_FOLDING_ENABLED
        else period_amp.astype(np.float64, copy=True)
    )

    cycles = signal.size / period
    response_mask = np.ones_like(period, dtype=bool)
    if RESPONSE_PERIOD_MIN is not None:
        response_mask &= period >= RESPONSE_PERIOD_MIN
    if RESPONSE_PERIOD_MAX is not None:
        response_mask &= period <= RESPONSE_PERIOD_MAX
    if MIN_RESPONSE_CYCLES is not None:
        response_mask &= cycles >= MIN_RESPONSE_CYCLES

    global_idx = int(np.argmax(folded_amp))
    if np.any(response_mask):
        candidates = np.flatnonzero(response_mask)
        response_idx = int(candidates[np.argmax(folded_amp[response_mask])])
    else:
        response_idx = None

    def make_candidates(order_indices):
        return [
            {
                "rank": int(rank),
                "period": float(period[idx]),
                "k": int(freq_bin[idx]),
                "amp": float(period_amp[idx]),
                "folded_score": float(folded_amp[idx]),
                "cycles": float(cycles[idx]),
            }
            for rank, idx in enumerate(order_indices, start=1)
        ]

    pool = np.flatnonzero(display_mask)
    raw_top = pool[np.argsort(period_amp[display_mask])[::-1]][:FFT_TOP_K]
    folded_top = pool[np.argsort(folded_amp[display_mask])[::-1]][:FFT_TOP_K]
    folded_candidates = make_candidates(folded_top)

    return {
        "period": period,
        "period_amplitude": period_amp,
        "folded_period_amplitude": folded_amp,
        "freq_bin": freq_bin,
        "cycles": cycles,
        "display_mask": display_mask,
        "raw_fft_top_candidates": make_candidates(raw_top),
        "folded_fft_top_candidates": folded_candidates,
        "fft_top_candidates": folded_candidates,
        "global_period": float(period[global_idx]),
        "global_k": int(freq_bin[global_idx]),
        "global_amp": float(period_amp[global_idx]),
        "global_folded_score": float(folded_amp[global_idx]),
        "response_period": np.nan if response_idx is None else float(period[response_idx]),
        "response_k": None if response_idx is None else int(freq_bin[response_idx]),
        "response_amp": np.nan if response_idx is None else float(period_amp[response_idx]),
        "response_folded_score": np.nan if response_idx is None else float(folded_amp[response_idx]),
        "signal_size": signal.size,
    }

def compute_acf(sequence, max_lag):
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"ACF input must be 1D with at least 2 frames, got shape {values.shape}")

    max_lag = int(min(max_lag, values.size - 1))
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")

    centered = values - values.mean()
    denom = float(np.dot(centered, centered))
    lags = np.arange(max_lag + 1)
    acf_values = np.zeros(max_lag + 1, dtype=np.float64)
    if denom <= np.finfo(np.float64).eps:
        acf_values[0] = 1.0
        return lags, acf_values

    for lag in lags:
        acf_values[lag] = np.dot(centered[: values.size - lag], centered[lag:]) / denom
    return lags, acf_values


def find_acf_peaks(acf_values, min_value=ACF_MIN):
    values = np.asarray(acf_values, dtype=np.float64)
    peaks = []
    for lag in range(1, values.size - 1):
        v = values[lag]
        if v >= min_value and v >= values[lag - 1] and v >= values[lag + 1]:
            peaks.append({"lag": int(lag), "acf": float(v)})
    return peaks


def validate_fft_candidates_with_acf(sequence, spectrum):
    lags, acf_values = compute_acf(sequence, max_lag=len(sequence) - 1)
    acf_peaks = find_acf_peaks(acf_values, min_value=ACF_MIN)
    acf_first_peak_lag = acf_peaks[0]["lag"] if acf_peaks else None

    top_candidates = spectrum["fft_top_candidates"]
    top1_top2_ratio = (
        top_candidates[0]["folded_score"] / top_candidates[1]["folded_score"]
        if len(top_candidates) >= 2 and top_candidates[1]["folded_score"] > 0
        else np.nan
    )
    near_tie = bool(np.isfinite(top1_top2_ratio) and top1_top2_ratio < NEAR_TIE_RATIO)

    def entry_for_k(k):
        if k is None:
            return None
        matches = np.flatnonzero(spectrum["freq_bin"] == k)
        if matches.size == 0:
            return None
        idx = int(matches[0])
        return {
            "period": float(spectrum["period"][idx]),
            "k": int(k),
            "amp": float(spectrum["period_amplitude"][idx]),
            "folded_score": float(spectrum["folded_period_amplitude"][idx]),
            "cycles": float(spectrum["cycles"][idx]),
        }

    candidate_map = {}
    def add(source, entry, rank=None):
        if entry is None or not np.isfinite(entry["period"]):
            return
        row = candidate_map.setdefault(entry["k"], {**entry, "source": [], "rank": rank})
        row["source"].append(source)
        if rank is not None:
            row["rank"] = rank

    add("response", entry_for_k(spectrum.get("response_k")))
    add("global", entry_for_k(spectrum.get("global_k")))
    for c in top_candidates:
        add(f"top{c['rank']}", c, rank=c["rank"])

    validations = []
    for row in candidate_map.values():
        lag = int(round(row["period"]))
        lo = max(1, lag - ACF_TOL)
        hi = min(len(acf_values) - 1, lag + ACF_TOL)
        local_peaks = [p for p in acf_peaks if lo <= p["lag"] <= hi]
        if local_peaks:
            best = max(local_peaks, key=lambda p: p["acf"])
            best_lag, best_acf = best["lag"], best["acf"]
            has_confirming = best_acf >= ACF_MIN
        elif lo <= hi:
            window = acf_values[lo : hi + 1]
            offset = int(np.argmax(window))
            best_lag, best_acf = int(lo + offset), float(window[offset])
            has_confirming = False
        else:
            best_lag, best_acf, has_confirming = None, np.nan, False

        if row["cycles"] < LOW_CYCLE_THRESHOLD:
            status = "low_cycle_envelope"
        elif has_confirming:
            status = "acf_confirmed"
        elif near_tie and row.get("rank") in (1, 2):
            status = "near_tie"
        else:
            status = "weak_or_unconfirmed"

        validations.append({
            **row,
            "source": ",".join(row["source"]),
            "lag": lag,
            "acf": float(acf_values[lag]) if 0 <= lag < len(acf_values) else np.nan,
            "best_lag": best_lag,
            "best_acf": float(best_acf) if np.isfinite(best_acf) else np.nan,
            "status": status,
        })

    validations.sort(key=lambda r: (r["rank"] is None, r["rank"] or 999, -r["folded_score"]))
    sources = lambda row: row["source"].split(",")
    response_status = next((r["status"] for r in validations if "response" in sources(r)), None)
    global_status = next((r["status"] for r in validations if "global" in sources(r)), None)

    return {
        "acf_lag": lags,
        "acf_value": acf_values,
        "acf_first_peak_lag": acf_first_peak_lag,
        "acf_validations": validations,
        "period_status": {
            "top1_top2_ratio": float(top1_top2_ratio) if np.isfinite(top1_top2_ratio) else np.nan,
            "near_tie": near_tie,
            "response_status": response_status,
            "global_status": global_status,
            "low_cycle_count": int(sum(r["status"] == "low_cycle_envelope" for r in validations)),
        },
    }

def analyze_fft_ranges(values, fft_ranges):
    results = []
    for cfg in fft_ranges:
        start, end, start_norm, end_norm = normalize_range(cfg.get("start"), cfg.get("end"), values.size)
        window_values = values[start_norm:end_norm]
        spectrum = compute_period_spectrum(window_values)
        acf_validation = validate_fft_candidates_with_acf(window_values, spectrum)
        results.append({**cfg, **spectrum, **acf_validation, "start_norm": start_norm, "end_norm": end_norm})
    return results

def _format_float(value, digits=3):
    return "nan" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _format_optional_int(value):
    return "none" if value is None else str(int(value))


def _print_fft_candidate_table(title, candidates):
    print(title)
    print("      rank   k   period      amp folded_score  cycles")
    for candidate in candidates[:FFT_DEBUG_TOP_K]:
        print(
            f"      {candidate['rank']:>4d} {candidate['k']:>3d} "
            f"{candidate['period']:>8.3f} {candidate['amp']:>8.4g} "
            f"{candidate['folded_score']:>12.4g} {candidate['cycles']:>7.2f}"
        )


def print_results(title, results):
    print(title)
    for result in results:
        inclusive_end = result["end_norm"] - 1
        print(
            f"  {result['label']}: frames {result['start_norm']}-{inclusive_end} "
            f"({result['signal_size']} frames), "
            f"global={result['global_period']:.3f} frames "
            f"(amp={result['global_amp']:.6g}, folded_score={result['global_folded_score']:.6g}), "
            f"response={result['response_period']:.3f} frames "
            f"(amp={result['response_amp']:.6g}, folded_score={result['response_folded_score']:.6g})"
        )
        _print_fft_candidate_table("    Raw FFT top candidates:", result["raw_fft_top_candidates"])
        _print_fft_candidate_table("    Folded FFT top candidates:", result["folded_fft_top_candidates"])
        status = result["period_status"]
        ratio = status["top1_top2_ratio"]
        ratio_text = "nan" if not np.isfinite(ratio) else f"{ratio:.3f}"
        print(
            f"    ACF first peak lag={_format_optional_int(result['acf_first_peak_lag'])}; "
            f"top1/top2_ratio={ratio_text}; "
            f"near_tie={status['near_tie']}; "
            f"response_status={status['response_status']}; "
            f"global_status={status['global_status']}; "
            f"low_cycle_count={status['low_cycle_count']}"
        )
        print("    FFT candidates verified by ACF:")
        print("      source           k   period      amp folded_score  cycles  lag     acf  best_lag best_acf  status")
        for row in result["acf_validations"]:
            best_lag = _format_optional_int(row["best_lag"])
            print(
                f"      {row['source']:<14} {row['k']:>3d} "
                f"{row['period']:>8.3f} {row['amp']:>8.4g} {row['folded_score']:>12.4g} {row['cycles']:>7.2f} "
                f"{row['lag']:>4d} {_format_float(row['acf'], 3):>7} "
                f"{best_lag:>8} {_format_float(row['best_acf'], 3):>8}  {row['status']}"
            )

def print_attention_diagnostics(
    raw_attention,
    softmax_full,
    softmax_window,
    softmax_window_start,
    softmax_window_label,
    tail_start=60,
    top_k=10,
):
    raw_top = np.argsort(raw_attention)[-top_k:][::-1]
    softmax_top = np.argsort(softmax_window)[-top_k:][::-1]
    softmax_window_end = softmax_window_start + softmax_window.size

    print("Attention diagnostics")
    print(
        f"  raw: sum={raw_attention.sum():.6g}, min={raw_attention.min():.6g}, "
        f"max={raw_attention.max():.6g} at frame {int(np.argmax(raw_attention))}"
    )
    print(
        f"  local softmax window: frames {softmax_window_label}; "
        f"window sum={softmax_window.sum():.6g} (normalization is local, not over all {raw_attention.size} frames), "
        f"min={softmax_window.min():.6g}, max={softmax_window.max():.6g} "
        f"at frame {int(softmax_window_start + np.argmax(softmax_window))}"
    )
    print(f"  raw frames {tail_start}-{raw_attention.size - 1}:")
    print("   ", np.array2string(raw_attention[tail_start:], precision=6, suppress_small=False))
    print(f"  local softmax frames {softmax_window_start}-{softmax_window_end - 1}:")
    print("   ", np.array2string(softmax_window, precision=6, suppress_small=False))
    print(f"  full softmax plot vector NaN outside frames {softmax_window_label}; nansum={np.nansum(softmax_full):.6g}")
    print("  top raw frames:")
    for idx in raw_top:
        print(f"    frame {int(idx):2d}: {raw_attention[idx]:.6g}")
    print("  top local softmax frames:")
    for idx in softmax_top:
        frame = int(softmax_window_start + idx)
        print(f"    frame {frame:2d}: {softmax_window[idx]:.6g}")

def plot_attention_and_spectra(
    frame_idx,
    raw_attention,
    softmax_attention,
    raw_results,
    softmax_results,
    softmax_window_start,
    softmax_window_end,
    softmax_window_label,
    save_path=None,
    save_dpi=300,
):
    fig, axes = plt.subplots(3, 2, figsize=MAIN_FIGURE_SIZE, constrained_layout=True)
    ax_raw_time, ax_softmax_time = axes[0]
    ax_raw_period, ax_softmax_period = axes[1]
    ax_raw_acf, ax_softmax_acf = axes[2]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax_raw_time.plot(frame_idx, raw_attention, marker="o", linewidth=0.8, markersize=1.8, color="black", label="Raw attention")
    ax_softmax_time.plot(frame_idx, softmax_attention, marker="o", linewidth=0.8, markersize=1.8, color="black", label=f"Local softmax {softmax_window_label}")

    for idx, result in enumerate(raw_results):
        color = colors[idx % len(colors)]
        ax_raw_time.axvspan(
            result["start_norm"] - 0.5,
            result["end_norm"] - 0.5,
            color=color,
            alpha=0.12,
            label=f"FFT {result['label']}",
        )

    ax_softmax_time.axvspan(
        softmax_window_start - 0.5,
        softmax_window_end - 0.5,
        color=colors[0],
        alpha=0.12,
        label=f"Local window {softmax_window_label}",
    )

    for ax, ylabel, title in [
        (ax_raw_time, "Attention", "Raw selected-head attention"),
        (ax_softmax_time, "Softmax(attention)", "Local softmax selected-head attention"),
    ]:
        ax.set_xlabel("Frame", fontsize=6)
        ax.set_ylabel(ylabel, fontsize=6)
        ax.set_title(title, fontsize=6)
        ax.tick_params(axis="both", labelsize=5, pad=1)
        ax.grid(True, alpha=0.25, linewidth=0.35)
        ax.legend(loc="best", fontsize=3.8, frameon=False)

    for ax, results, title in [
        (ax_raw_period, raw_results, "Raw period spectrum + folded score"),
        (ax_softmax_period, softmax_results, "Local softmax period spectrum + folded score"),
    ]:
        for idx, result in enumerate(results):
            color = colors[idx % len(colors)]
            mask = result["display_mask"]
            ax.plot(
                result["period"][mask],
                result["period_amplitude"][mask],
                marker="o",
                linewidth=0.8,
                markersize=1.8,
                color=color,
                label=f"{result['label']} raw amp | response {result['response_period']:.2f} | global {result['global_period']:.2f}",
            )
            ax.plot(
                result["period"][mask],
                result["folded_period_amplitude"][mask],
                marker="x",
                linestyle="-.",
                linewidth=0.7,
                markersize=1.8,
                color=color,
                alpha=0.75,
                label=f"{result['label']} folded score",
            )
            if np.isfinite(result["global_period"]):
                ax.axvline(result["global_period"], color=color, linestyle=":", linewidth=0.6, alpha=0.6)
            if np.isfinite(result["response_period"]):
                ax.axvline(result["response_period"], color=color, linestyle="--", linewidth=0.7, alpha=0.9)

        ax.set_xlabel("Period (frames/cycle)", fontsize=6)
        ax.set_ylabel("Raw FFT amp / folded score", fontsize=6)
        ax.set_title(title, fontsize=6)
        ax.tick_params(axis="both", labelsize=5, pad=1)
        ax.grid(True, alpha=0.25, linewidth=0.35)
        ax.legend(loc="best", fontsize=3.8, frameon=False)

    status_styles = {
        "acf_confirmed": {"color": "tab:green", "marker": "o", "label": "ACF confirmed"},
        "low_cycle_envelope": {"color": "tab:orange", "marker": "s", "label": "Low-cycle envelope"},
        "near_tie": {"color": "tab:purple", "marker": "D", "label": "Near FFT tie"},
        "weak_or_unconfirmed": {"color": "0.45", "marker": "x", "label": "Weak/unconfirmed"},
    }

    for ax, results, title in [
        (ax_raw_acf, raw_results, "Raw ACF validation"),
        (ax_softmax_acf, softmax_results, "Local softmax ACF validation"),
    ]:
        used_labels = set()
        for idx, result in enumerate(results):
            color = colors[idx % len(colors)]
            lags = result["acf_lag"]
            acf_values = result["acf_value"]
            ax.plot(lags[1:], acf_values[1:], linewidth=1.5, color=color, label=f"{result['label']} ACF")
            first_peak = result["acf_first_peak_lag"]
            if first_peak is not None:
                ax.axvline(first_peak, color=color, linestyle="--", linewidth=0.6, alpha=0.7)
                ax.scatter(
                    [first_peak],
                    [acf_values[first_peak]],
                    color=color,
                    marker="^",
                    s=45,
                    zorder=4,
                    label=f"{result['label']} first peak",
                )

            for row in result["acf_validations"]:
                style = status_styles[row["status"]]
                label = style["label"] if style["label"] not in used_labels else None
                if label is not None:
                    used_labels.add(style["label"])
                ax.axvline(row["lag"], color=style["color"], linestyle=":", linewidth=0.6, alpha=0.45)
                if row["best_lag"] is not None and np.isfinite(row["best_acf"]):
                    ax.scatter(
                        [row["best_lag"]],
                        [row["best_acf"]],
                        color=style["color"],
                        marker=style["marker"],
                        s=14,
                        zorder=5,
                        label=label,
                    )

        ax.axhline(ACF_MIN, color="0.25", linestyle="--", linewidth=0.6, alpha=0.55, label=f"ACF min {ACF_MIN:g}")
        ax.set_xlabel("Lag (frames)", fontsize=6)
        ax.set_ylabel("ACF", fontsize=6)
        ax.set_title(title, fontsize=6)
        ax.tick_params(axis="both", labelsize=5, pad=1)
        ax.grid(True, alpha=0.25, linewidth=0.35)
        ax.legend(loc="best", fontsize=3.8, frameon=False)

    fig.suptitle(
        f"Layer {LAYER}, Head {HEAD} | FFT solid=raw, dash-dot=folded score; vertical dashed=response, dotted=global",
        fontsize=6,
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(save_path.with_suffix(".pdf"), dpi=save_dpi, bbox_inches="tight", pad_inches=0.01)
        print(f"Saved main figure to {save_path}")
    plt.show()

import csv
from collections import Counter


BATCH_PROMPTS_DIR = PROJECT_ROOT / "data/26May4-PyramidForcing-multihead/prompts256"
BATCH_NUM_PROMPTS = 256
BATCH_NUM_LAYERS = 30
BATCH_NUM_HEADS = 12
BATCH_RANGE_LABEL = "0-68"
BATCH_RANGE_START = 0
BATCH_RANGE_END = 69
HEAD_PERIOD_HOMOLOGY_CSV_PATH = OUTPUT_DIR / "head_period_homology_256_firstdiff_folded_top1_0_68.csv"
HEAD_PERIOD_HOMOLOGY_FIGURE_PATH = OUTPUT_DIR / "head_period_homology_256_firstdiff_folded_top1_0_68.png"
HEAD_PERIOD_HOMOLOGY_SCATTER_FIGURE_PATH = OUTPUT_DIR / "head_period_homology_256_firstdiff_folded_top1_0_68_scatter.png"
COMPACT_FIGURE_SIZE = (3.04, 1.65)
BALANCED_HEAD_COUNT = 120
CYCLE6_HEAD_FRACTION = 0.5
CYCLE6_PERIOD_RANGE = (5.8, 6.5)
CYCLE6_PERIOD_CENTER = 6.18
CYCLE6_STD_MAX = 1.5
HIGH_PERIOD_MIN = 6.6
HIGH_PERIOD_STD_MAX = 3.5
CYCLE6_PER_LAYER_TARGET = 2
UNIFORM_HEAD_COUNT = 24
HEAD_PERIOD_HOMOLOGY_REPRESENTATIVE_FIGURE_PATH = (
    OUTPUT_DIR / "head_period_homology_uniform_layer4_120_mean_std.png"
)
HEAD_PERIOD_HOMOLOGY_UNIFORM_FIGURE_PATH = (
    OUTPUT_DIR / "head_period_homology_uniform_24_mean_std.png"
)


def compute_head_period_homology_256():
    prompt_dirs = [BATCH_PROMPTS_DIR / f"run_{i:03d}" for i in range(BATCH_NUM_PROMPTS)]
    missing = [p for p in prompt_dirs if not p.is_dir()]
    if missing:
        preview = ", ".join(str(p) for p in missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} prompt directories under {BATCH_PROMPTS_DIR}: {preview}")

    periods = np.full((BATCH_NUM_LAYERS, BATCH_NUM_HEADS, BATCH_NUM_PROMPTS), np.nan, dtype=np.float64)
    errors = []

    for prompt_idx, prompt_dir in enumerate(prompt_dirs):
        if prompt_idx == 0 or (prompt_idx + 1) % 32 == 0 or prompt_idx + 1 == BATCH_NUM_PROMPTS:
            print(f"Processing prompt {prompt_idx + 1}/{BATCH_NUM_PROMPTS}: {prompt_dir.name}")
        for layer in range(BATCH_NUM_LAYERS):
            attn_path = prompt_dir / f"layer{layer}.pt"
            try:
                payload = torch.load(attn_path, map_location="cpu", weights_only=False)
                if "last_frame_attention_per_head" in payload:
                    per_head = payload["last_frame_attention_per_head"]
                elif "last_block_frame_attention" in payload:
                    per_head = payload["last_block_frame_attention"]
                else:
                    raise KeyError("Missing last_frame_attention_per_head or last_block_frame_attention")
                if per_head.ndim != 2 or per_head.shape[0] < BATCH_NUM_HEADS:
                    raise ValueError(f"Unexpected per-head shape {tuple(per_head.shape)}")
                per_head_np = per_head[:BATCH_NUM_HEADS].detach().cpu().float().numpy()
                for head in range(BATCH_NUM_HEADS):
                    sequence = per_head_np[head, BATCH_RANGE_START:BATCH_RANGE_END]
                    spectrum = compute_period_spectrum(sequence)
                    periods[layer, head, prompt_idx] = float(spectrum["folded_fft_top_candidates"][0]["period"])
            except Exception as exc:
                errors.append({"prompt_idx": prompt_idx, "layer": layer, "path": str(attn_path), "error": repr(exc)})

    return summarize_head_periods(periods), errors, periods


def summarize_head_periods(periods):
    rows = []
    num_layers, num_heads, _ = periods.shape
    for layer in range(num_layers):
        for head in range(num_heads):
            valid = periods[layer, head][np.isfinite(periods[layer, head])]
            if valid.size:
                counts = Counter(float(v) for v in valid)
                dominant_period, dominant_count = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
                p_min, p_max = float(np.min(valid)), float(np.max(valid))
                p_q25, p_q75 = float(np.quantile(valid, 0.25)), float(np.quantile(valid, 0.75))
                stats = {
                    "n_prompts": int(valid.size),
                    "period_mean": float(np.mean(valid)),
                    "period_min": p_min,
                    "period_max": p_max,
                    "period_std": float(np.std(valid)),
                    "period_q25": p_q25,
                    "period_q75": p_q75,
                    "period_iqr": p_q75 - p_q25,
                    "period_range": p_max - p_min,
                    "dominant_period": float(dominant_period),
                    "dominant_period_count": int(dominant_count),
                }
            else:
                stats = {k: (0 if k in ("n_prompts", "dominant_period_count") else np.nan)
                         for k in ("n_prompts", "period_mean", "period_min", "period_max", "period_std",
                                   "period_q25", "period_q75", "period_iqr", "period_range",
                                   "dominant_period", "dominant_period_count")}
            rows.append({"layer": layer, "head": head, "head_index": layer * num_heads + head, **stats})
    return rows


def write_head_period_homology_csv(rows, csv_path=HEAD_PERIOD_HOMOLOGY_CSV_PATH):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")


def read_head_period_homology_csv(csv_path=HEAD_PERIOD_HOMOLOGY_CSV_PATH):
    csv_path = Path(csv_path)
    int_fields = {"layer", "head", "head_index", "n_prompts", "dominant_period_count"}
    rows = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            typed = {}
            for key, value in row.items():
                if key in int_fields:
                    typed[key] = int(value)
                else:
                    typed[key] = float(value) if value != "" else np.nan
            rows.append(typed)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def _add_layer_grid(ax):
    for layer in range(BATCH_NUM_LAYERS + 1):
        ax.axvline(layer * BATCH_NUM_HEADS - 0.5, color="0.72", linewidth=0.7, alpha=0.75, zorder=0)
    layer_centers = [layer * BATCH_NUM_HEADS + (BATCH_NUM_HEADS - 1) / 2 for layer in range(BATCH_NUM_LAYERS)]
    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(layer_centers)
    top_ax.set_xticklabels([f"L{layer}" for layer in range(BATCH_NUM_LAYERS)], fontsize=8)
    top_ax.tick_params(axis="x", length=0, pad=3)
    top_ax.set_xlabel("Layer")
    ax.set_xlim(-0.5, BATCH_NUM_LAYERS * BATCH_NUM_HEADS - 0.5)


def plot_head_period_homology(rows, figure_path=HEAD_PERIOD_HOMOLOGY_FIGURE_PATH, dpi=300):
    x = np.array([r["head_index"] for r in rows], dtype=np.float64)
    mean = np.array([r["period_mean"] for r in rows], dtype=np.float64)
    std = np.array([r["period_std"] for r in rows], dtype=np.float64)
    lower = mean - std
    upper = mean + std

    fig, ax = plt.subplots(figsize=(22, 7), constrained_layout=True)
    color = "#1f4e79"
    ax.fill_between(x, lower, upper, color=color, alpha=0.16, label="mean +/- std")
    ax.plot(x, mean, color=color, marker="o", markersize=3.2, linewidth=1.6, label="mean")

    _add_layer_grid(ax)
    ax.set_xlabel("Head index (layer * 12 + head)")
    ax.set_ylabel("Period (frames/cycle)")
    ax.set_title("All-head period homology across 256 prompts | first diff | folded FFT top1 | frames 0-68")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper right")

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {figure_path}")
    plt.show()


def plot_head_period_homology_scatter(rows, periods, figure_path=HEAD_PERIOD_HOMOLOGY_SCATTER_FIGURE_PATH, dpi=300):
    expected_shape = (BATCH_NUM_LAYERS, BATCH_NUM_HEADS, BATCH_NUM_PROMPTS)
    if periods.shape != expected_shape:
        raise AssertionError(f"Expected periods shape {expected_shape}, got {periods.shape}")

    x = np.array([r["head_index"] for r in rows], dtype=np.float64)
    flat = periods.reshape(BATCH_NUM_LAYERS * BATCH_NUM_HEADS, BATCH_NUM_PROMPTS)
    prompt_offsets = np.linspace(-0.35, 0.35, BATCH_NUM_PROMPTS)
    scatter_x = (x[:, None] + prompt_offsets[None, :]).reshape(-1)
    scatter_y = flat.reshape(-1)
    finite = np.isfinite(scatter_y)

    fig, ax = plt.subplots(figsize=(22, 7), constrained_layout=True)
    ax.scatter(scatter_x[finite], scatter_y[finite], s=9, color="#2f6fae", alpha=0.38,
               linewidths=0, label="prompt periods", zorder=1)

    _add_layer_grid(ax)
    ax.set_ylim(0, (BATCH_RANGE_END - BATCH_RANGE_START - 1) / 2)
    ax.set_xlabel("Head index (layer * 12 + head; prompts spread within each head)")
    ax.set_ylabel("Folded FFT top1 period (frames/cycle)")
    ax.set_title("All-head period distribution across 256 prompts | first diff | folded FFT top1 | frames 0-68 | ylim 0-Ndiff/2")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper right")

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved scatter figure to {figure_path}")
    plt.show()


def _finite_rows(rows):
    return [
        r for r in rows
        if np.isfinite(r["period_mean"])
        and np.isfinite(r["period_std"])
        and np.isfinite(r["dominant_period"])
    ]


def _select_unique(sorted_rows, count, used_head_indices=None):
    used = set() if used_head_indices is None else set(used_head_indices)
    selected = []
    for row in sorted_rows:
        head_index = row["head_index"]
        if head_index in used:
            continue
        selected.append(row)
        used.add(head_index)
        if len(selected) >= count:
            break
    return selected


def select_representative_head_period_homology_rows(
    rows,
    *,
    head_count=BALANCED_HEAD_COUNT,
    cycle6_head_fraction=CYCLE6_HEAD_FRACTION,
    cycle6_period_range=CYCLE6_PERIOD_RANGE,
    cycle6_period_center=CYCLE6_PERIOD_CENTER,
    cycle6_std_max=CYCLE6_STD_MAX,
    high_period_min=HIGH_PERIOD_MIN,
    high_period_std_max=HIGH_PERIOD_STD_MAX,
    cycle6_per_layer_target=CYCLE6_PER_LAYER_TARGET,
):
    rows = _finite_rows(rows)
    if head_count <= 0:
        raise ValueError("head_count must be positive")
    if not 0 <= cycle6_head_fraction <= 1:
        raise ValueError("cycle6_head_fraction must be between 0 and 1")
    if len(cycle6_period_range) != 2:
        raise ValueError("cycle6_period_range must contain exactly two values")
    cycle6_min, cycle6_max = cycle6_period_range
    if cycle6_min > cycle6_max:
        raise ValueError("cycle6_period_range minimum must be <= maximum")
    if cycle6_per_layer_target < 0:
        raise ValueError("cycle6_per_layer_target must be non-negative")

    selected = []
    used = set()
    bucket_counts = Counter()

    cycle6_quota = int(round(head_count * cycle6_head_fraction))
    cycle6_quota = max(0, min(head_count, cycle6_quota))
    high_quota = head_count - cycle6_quota

    def cycle6_key(row):
        return (
            row["period_std"],
            abs(row["period_mean"] - cycle6_period_center),
            -row["dominant_period_count"],
            row["layer"],
            row["head"],
        )

    def high_key(row, layer_counts):
        return (
            layer_counts[row["layer"]],
            row["period_std"],
            -row["period_mean"],
            row["layer"],
            row["head"],
        )

    def is_cycle6(row):
        return cycle6_min <= row["period_mean"] <= cycle6_max

    def is_high(row):
        return row["period_mean"] >= high_period_min

    def std_ok(row, std_max):
        return std_max is None or row["period_std"] <= std_max

    def add_row(row, bucket, source):
        if row["head_index"] in used:
            return False
        tagged = dict(row)
        tagged["selection_bucket"] = bucket
        tagged["selection_source"] = source
        selected.append(tagged)
        used.add(row["head_index"])
        bucket_counts[bucket] += 1
        return True

    cycle6_strict = [r for r in rows if is_cycle6(r) and std_ok(r, cycle6_std_max)]
    cycle6_by_layer = {}
    for row in sorted(cycle6_strict, key=cycle6_key):
        cycle6_by_layer.setdefault(row["layer"], []).append(row)

    for layer in sorted(cycle6_by_layer):
        for row in cycle6_by_layer[layer][:cycle6_per_layer_target]:
            if bucket_counts["cycle6"] >= cycle6_quota:
                break
            add_row(row, "cycle6", "strict_layer_target")

    cycle6_remaining = sorted(
        [r for r in cycle6_strict if r["head_index"] not in used],
        key=cycle6_key,
    )
    for row in cycle6_remaining:
        if bucket_counts["cycle6"] >= cycle6_quota:
            break
        add_row(row, "cycle6", "strict_global_fill")

    cycle6_selected_count = bucket_counts["cycle6"]
    if cycle6_selected_count < cycle6_quota:
        print(
            f"WARNING: only found {cycle6_selected_count}/{cycle6_quota} cycle6 heads with "
            f"period_mean in [{cycle6_min}, {cycle6_max}] and period_std <= {cycle6_std_max}; "
            "relaxing cycle6 std limit for remaining slots."
        )
        cycle6_relaxed = sorted(
            [r for r in rows if is_cycle6(r) and r["head_index"] not in used],
            key=cycle6_key,
        )
        for row in cycle6_relaxed:
            if bucket_counts["cycle6"] >= cycle6_quota:
                break
            add_row(row, "cycle6", "relaxed_std_fill")

    layer_counts = Counter(r["layer"] for r in selected)

    def add_high_candidates(candidates, source):
        remaining = list(candidates)
        while bucket_counts["high"] < high_quota and remaining:
            remaining = [r for r in remaining if r["head_index"] not in used]
            if not remaining:
                break
            remaining.sort(key=lambda r: high_key(r, layer_counts))
            row = remaining.pop(0)
            if add_row(row, "high", source):
                layer_counts[row["layer"]] += 1

    high_strict = [
        r for r in rows
        if is_high(r)
        and std_ok(r, high_period_std_max)
        and r["head_index"] not in used
    ]
    add_high_candidates(high_strict, "strict")

    high_selected_count = bucket_counts["high"]
    if high_selected_count < high_quota:
        print(
            f"WARNING: only found {high_selected_count}/{high_quota} high-period heads with "
            f"period_mean >= {high_period_min} and period_std <= {high_period_std_max}; "
            "relaxing high-period std limit for remaining slots."
        )
        high_relaxed = [
            r for r in rows
            if is_high(r)
            and r["head_index"] not in used
        ]
        add_high_candidates(high_relaxed, "relaxed_std_fill")

    if len(selected) < head_count:
        print(f"WARNING: selected only {len(selected)}/{head_count} representative heads.")

    return sorted(selected, key=lambda r: (r["layer"], r["head"], r["head_index"]))


def select_uniform_head_period_homology_rows(rows, *, head_count=UNIFORM_HEAD_COUNT):
    rows = _finite_rows(rows)
    if head_count <= 0:
        raise ValueError("head_count must be positive")
    sorted_rows = sorted(rows, key=lambda r: r["head_index"])
    head_indices = np.array([r["head_index"] for r in sorted_rows], dtype=np.float64)
    targets = np.linspace(head_indices.min(), head_indices.max(), min(head_count, len(sorted_rows)))

    selected = []
    used = set()
    for target in targets:
        order = np.argsort(np.abs(head_indices - target), kind="stable")
        for row_idx in order:
            row = sorted_rows[int(row_idx)]
            if row["head_index"] not in used:
                selected.append(row)
                used.add(row["head_index"])
                break

    return sorted(selected, key=lambda r: r["head_index"])


def _save_compact_head_period_homology(rows, figure_path, title, figure_size, dpi):
    if not rows:
        raise ValueError("No rows selected for compact head period homology plot")

    x = np.arange(len(rows), dtype=np.float64)
    mean = np.array([r["period_mean"] for r in rows], dtype=np.float64)
    std = np.array([r["period_std"] for r in rows], dtype=np.float64)
    lower = mean - std
    upper = mean + std

    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    color = "#1f4e79"
    ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0, label="standard")
    ax.plot(x, mean, color=color, marker=None, linewidth=1.0, label="mean")

    ax.set_xlim(-0.4, len(rows) - 0.6)
    layer_ticks_all = []
    layer_labels_all = []
    for layer in sorted({r["layer"] for r in rows}):
        layer_positions = [idx for idx, row in enumerate(rows) if row["layer"] == layer]
        layer_ticks_all.append((layer_positions[0] + layer_positions[-1]) / 2)
        layer_labels_all.append(f"L{layer}")
    label_stride = max(1, int(np.ceil(len(layer_ticks_all) / 10)))
    layer_ticks = layer_ticks_all[::label_stride]
    layer_labels = layer_labels_all[::label_stride]
    ax.set_xticks(layer_ticks)
    ax.set_xticklabels(layer_labels, rotation=90, ha="center", fontsize=6, fontstyle="italic")
    ax.tick_params(axis="x", pad=1, length=1.5)
    ax.tick_params(axis="y", pad=1)
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=6, frameon=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_clip_on(False)
    border_inset = 0.0015
    ax.add_patch(
        mpl.patches.Rectangle(
            (border_inset, border_inset),
            1 - 2 * border_inset,
            1 - 2 * border_inset,
            transform=ax.transAxes,
            fill=False,
            edgecolor="black",
            linewidth=0.6,
            clip_on=False,
            zorder=10,
        )
    )

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_path.with_suffix(".pdf")
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(figure_path, dpi=dpi, bbox_inches=None, pad_inches=0)
        fig.savefig(pdf_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    print(f"Saved compact figure to {figure_path}")
    print(f"Saved compact figure to {pdf_path}")
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()
    return rows


def plot_representative_head_period_homology(
    rows,
    figure_path=HEAD_PERIOD_HOMOLOGY_REPRESENTATIVE_FIGURE_PATH,
    *,
    figure_size=COMPACT_FIGURE_SIZE,
    head_count=BALANCED_HEAD_COUNT,
    cycle6_head_fraction=CYCLE6_HEAD_FRACTION,
    cycle6_period_range=CYCLE6_PERIOD_RANGE,
    cycle6_period_center=CYCLE6_PERIOD_CENTER,
    cycle6_std_max=CYCLE6_STD_MAX,
    high_period_min=HIGH_PERIOD_MIN,
    high_period_std_max=HIGH_PERIOD_STD_MAX,
    cycle6_per_layer_target=CYCLE6_PER_LAYER_TARGET,
    dpi=300,
):
    selected = select_representative_head_period_homology_rows(
        rows,
        head_count=head_count,
        cycle6_head_fraction=cycle6_head_fraction,
        cycle6_period_range=cycle6_period_range,
        cycle6_period_center=cycle6_period_center,
        cycle6_std_max=cycle6_std_max,
        high_period_min=high_period_min,
        high_period_std_max=high_period_std_max,
        cycle6_per_layer_target=cycle6_per_layer_target,
    )
    cycle6_quota = int(round(head_count * cycle6_head_fraction))
    high_quota = head_count - cycle6_quota
    return _save_compact_head_period_homology(
        selected,
        figure_path,
        f"Balanced heads | {cycle6_quota} cycle6 + {high_quota} high | mean +/- std",
        figure_size,
        dpi,
    )


def plot_uniform_head_period_homology(
    rows,
    figure_path=HEAD_PERIOD_HOMOLOGY_UNIFORM_FIGURE_PATH,
    *,
    figure_size=COMPACT_FIGURE_SIZE,
    head_count=UNIFORM_HEAD_COUNT,
    dpi=300,
):
    selected = select_uniform_head_period_homology_rows(rows, head_count=head_count)
    return _save_compact_head_period_homology(
        selected,
        figure_path,
        "Uniform heads | 256 prompts | mean +/- std",
        figure_size,
        dpi,
    )


def plot_compact_head_period_homology_from_csv(
    csv_path=HEAD_PERIOD_HOMOLOGY_CSV_PATH,
    *,
    representative_figure_path=HEAD_PERIOD_HOMOLOGY_REPRESENTATIVE_FIGURE_PATH,
    uniform_figure_path=HEAD_PERIOD_HOMOLOGY_UNIFORM_FIGURE_PATH,
    compact_figure_size=COMPACT_FIGURE_SIZE,
    balanced_head_count=BALANCED_HEAD_COUNT,
    cycle6_head_fraction=CYCLE6_HEAD_FRACTION,
    cycle6_period_range=CYCLE6_PERIOD_RANGE,
    cycle6_period_center=CYCLE6_PERIOD_CENTER,
    cycle6_std_max=CYCLE6_STD_MAX,
    high_period_min=HIGH_PERIOD_MIN,
    high_period_std_max=HIGH_PERIOD_STD_MAX,
    cycle6_per_layer_target=CYCLE6_PER_LAYER_TARGET,
    uniform_head_count=UNIFORM_HEAD_COUNT,
    dpi=300,
):
    rows = read_head_period_homology_csv(csv_path)
    representative_rows = plot_representative_head_period_homology(
        rows,
        representative_figure_path,
        figure_size=compact_figure_size,
        head_count=balanced_head_count,
        cycle6_head_fraction=cycle6_head_fraction,
        cycle6_period_range=cycle6_period_range,
        cycle6_period_center=cycle6_period_center,
        cycle6_std_max=cycle6_std_max,
        high_period_min=high_period_min,
        high_period_std_max=high_period_std_max,
        cycle6_per_layer_target=cycle6_per_layer_target,
        dpi=dpi,
    )
    uniform_rows = plot_uniform_head_period_homology(
        rows,
        uniform_figure_path,
        figure_size=compact_figure_size,
        head_count=uniform_head_count,
        dpi=dpi,
    )
    return {"rows": rows, "representative_rows": representative_rows, "uniform_rows": uniform_rows}


def _head_period_homology_marker_paths(csv_path):
    csv_path = Path(csv_path)
    return (
        csv_path.with_name(csv_path.name + ".running"),
        csv_path.with_name(csv_path.name + ".done"),
    )


def _write_head_period_homology_marker(marker_path, text):
    marker_path = Path(marker_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(text)
    print(f"Wrote marker {marker_path}")


def validate_head_period_homology(rows, errors, periods):
    if errors:
        raise RuntimeError(f"Batch period estimation had {len(errors)} errors; first errors: {errors[:5]}")
    expected_rows = BATCH_NUM_LAYERS * BATCH_NUM_HEADS
    expected_estimates = expected_rows * BATCH_NUM_PROMPTS
    observed = int(np.isfinite(periods).sum())
    assert len(rows) == expected_rows, f"Expected {expected_rows} rows, got {len(rows)}"
    assert observed == expected_estimates, f"Expected {expected_estimates} estimates, got {observed}"

    l12h0_top = Counter(float(v) for v in periods[12, 0]).most_common(8)
    print(f"Validated {observed} period estimates across {len(rows)} heads")
    print("L12H0 top periods:", [(round(p, 3), c) for p, c in l12h0_top])


def _update_runtime_config(**kwargs):
    globals().update(kwargs)


def run_fft_six_panel(
    *,
    attn_dir,
    layer,
    head,
    fft_ranges,
    softmax_window_start,
    softmax_window_end,
    softmax_window_label,
    softmax_fft_ranges,
    period_min=2.0,
    period_max=None,
    response_period_min=4.0,
    response_period_max=18.0,
    min_response_cycles=3.0,
    fft_top_k=5,
    acf_tol=2,
    acf_min=0.2,
    near_tie_ratio=1.05,
    low_cycle_threshold=3.0,
    harmonic_folding_enabled=True,
    harmonic_fold_min_period=3.0,
    harmonic_fold_max_period_divisor=6.0,
    harmonic_fold_max_multiple=4,
    harmonic_fold_tol=0.2,
    harmonic_fold_decay=0.5,
    fft_debug_top_k=3,
    remove_dc=True,
    apply_window=True,
    softmax_temperature=1.0,
    save_main_figure=True,
    main_figure_dpi=300,
    main_figure_size=(5.50, 4.20),
    output_dir=Path("../../figures/26May4-PyramidForcing-multihead"),
    main_figure_path=None,
):
    attn_dir = Path(attn_dir)
    output_dir = Path(output_dir)
    if main_figure_path is None:
        main_figure_path = output_dir / f"spectral_analysis_L{layer}H{head}_firstdiff_main.png"
    else:
        main_figure_path = Path(main_figure_path)

    _update_runtime_config(
        LAYER=layer,
        HEAD=head,
        FFT_RANGES=fft_ranges,
        SOFTMAX_WINDOW_START=softmax_window_start,
        SOFTMAX_WINDOW_END=softmax_window_end,
        SOFTMAX_WINDOW_LABEL=softmax_window_label,
        SOFTMAX_FFT_RANGES=softmax_fft_ranges,
        PERIOD_MIN=period_min,
        PERIOD_MAX=period_max,
        RESPONSE_PERIOD_MIN=response_period_min,
        RESPONSE_PERIOD_MAX=response_period_max,
        MIN_RESPONSE_CYCLES=min_response_cycles,
        FFT_TOP_K=fft_top_k,
        ACF_TOL=acf_tol,
        ACF_MIN=acf_min,
        NEAR_TIE_RATIO=near_tie_ratio,
        LOW_CYCLE_THRESHOLD=low_cycle_threshold,
        HARMONIC_FOLDING_ENABLED=harmonic_folding_enabled,
        HARMONIC_FOLD_MIN_PERIOD=harmonic_fold_min_period,
        HARMONIC_FOLD_MAX_PERIOD_DIVISOR=harmonic_fold_max_period_divisor,
        HARMONIC_FOLD_MAX_MULTIPLE=harmonic_fold_max_multiple,
        HARMONIC_FOLD_TOL=harmonic_fold_tol,
        HARMONIC_FOLD_DECAY=harmonic_fold_decay,
        FFT_DEBUG_TOP_K=fft_debug_top_k,
        REMOVE_DC=remove_dc,
        APPLY_WINDOW=apply_window,
        SOFTMAX_TEMPERATURE=softmax_temperature,
        MAIN_FIGURE_SIZE=main_figure_size,
    )

    attention, frame_idx, attn_path = load_attention(attn_dir, layer, head)
    softmax_attention, softmax_window = local_softmax_1d(
        attention,
        softmax_window_start,
        softmax_window_end,
        temperature=softmax_temperature,
    )

    raw_results = analyze_fft_ranges(attention, fft_ranges)
    softmax_results = analyze_fft_ranges(softmax_window, softmax_fft_ranges)

    print(f"Loaded {attn_path}: layer={layer}, head={head}, frames={attention.size}, preprocess=first_difference")
    print_attention_diagnostics(
        attention,
        softmax_attention,
        softmax_window,
        softmax_window_start,
        softmax_window_label,
    )
    print_results("Raw attention", raw_results)
    print_results("Local softmax attention", softmax_results)

    plot_attention_and_spectra(
        frame_idx,
        attention,
        softmax_attention,
        raw_results,
        softmax_results,
        softmax_window_start,
        softmax_window_end,
        softmax_window_label,
        save_path=main_figure_path if save_main_figure else None,
        save_dpi=main_figure_dpi,
    )

    return {
        "attention": attention,
        "frame_idx": frame_idx,
        "attn_path": attn_path,
        "softmax_attention": softmax_attention,
        "softmax_window": softmax_window,
        "raw_results": raw_results,
        "softmax_results": softmax_results,
        "main_figure_path": main_figure_path if save_main_figure else None,
    }


def run_256prompts_distribution(
    *,
    batch_prompts_dir,
    batch_num_prompts=256,
    batch_num_layers=30,
    batch_num_heads=12,
    batch_range_label="0-68",
    batch_range_start=0,
    batch_range_end=69,
    csv_path=HEAD_PERIOD_HOMOLOGY_CSV_PATH,
    representative_figure_path=HEAD_PERIOD_HOMOLOGY_REPRESENTATIVE_FIGURE_PATH,
    uniform_figure_path=HEAD_PERIOD_HOMOLOGY_UNIFORM_FIGURE_PATH,
    compact_figure_size=COMPACT_FIGURE_SIZE,
    balanced_head_count=BALANCED_HEAD_COUNT,
    cycle6_head_fraction=CYCLE6_HEAD_FRACTION,
    cycle6_period_range=CYCLE6_PERIOD_RANGE,
    cycle6_period_center=CYCLE6_PERIOD_CENTER,
    cycle6_std_max=CYCLE6_STD_MAX,
    high_period_min=HIGH_PERIOD_MIN,
    high_period_std_max=HIGH_PERIOD_STD_MAX,
    cycle6_per_layer_target=CYCLE6_PER_LAYER_TARGET,
    uniform_head_count=UNIFORM_HEAD_COUNT,
    figure_path=None,
    scatter_figure_path=None,
    dpi=300,
    period_min=2.0,
    period_max=None,
    response_period_min=4.0,
    response_period_max=18.0,
    min_response_cycles=3.0,
    fft_top_k=5,
    acf_tol=2,
    acf_min=0.2,
    near_tie_ratio=1.05,
    low_cycle_threshold=3.0,
    harmonic_folding_enabled=True,
    harmonic_fold_min_period=3.0,
    harmonic_fold_max_period_divisor=6.0,
    harmonic_fold_max_multiple=4,
    harmonic_fold_tol=0.2,
    harmonic_fold_decay=0.5,
    remove_dc=True,
    apply_window=True,
    use_markers=True,
    force_rerun=False,
):
    csv_path = Path(csv_path)
    representative_figure_path = Path(representative_figure_path)
    uniform_figure_path = Path(uniform_figure_path)
    running_marker_path, done_marker_path = _head_period_homology_marker_paths(csv_path)

    if use_markers and not force_rerun:
        if done_marker_path.exists():
            print(f"Found done marker {done_marker_path}; skipping full 256-prompts computation.")
            if csv_path.exists():
                compact_result = plot_compact_head_period_homology_from_csv(
                    csv_path=csv_path,
                    representative_figure_path=representative_figure_path,
                    uniform_figure_path=uniform_figure_path,
                    compact_figure_size=compact_figure_size,
                    balanced_head_count=balanced_head_count,
                    cycle6_head_fraction=cycle6_head_fraction,
                    cycle6_period_range=cycle6_period_range,
                    cycle6_period_center=cycle6_period_center,
                    cycle6_std_max=cycle6_std_max,
                    high_period_min=high_period_min,
                    high_period_std_max=high_period_std_max,
                    cycle6_per_layer_target=cycle6_per_layer_target,
                    uniform_head_count=uniform_head_count,
                    dpi=dpi,
                )
                return {
                    **compact_result,
                    "errors": [],
                    "periods": None,
                    "skipped": True,
                    "skip_reason": "done_marker",
                    "running_marker_path": running_marker_path,
                    "done_marker_path": done_marker_path,
                }
            print(f"CSV {csv_path} is missing; delete {done_marker_path} or set force_rerun=True to recompute.")
            return {
                "rows": None,
                "errors": [],
                "periods": None,
                "representative_rows": None,
                "uniform_rows": None,
                "skipped": True,
                "skip_reason": "done_marker_without_csv",
                "running_marker_path": running_marker_path,
                "done_marker_path": done_marker_path,
            }
        if running_marker_path.exists():
            print(f"Found running marker {running_marker_path}; skipping because another run may be active.")
            return {
                "rows": None,
                "errors": [],
                "periods": None,
                "representative_rows": None,
                "uniform_rows": None,
                "skipped": True,
                "skip_reason": "running_marker",
                "running_marker_path": running_marker_path,
                "done_marker_path": done_marker_path,
            }

    if use_markers and force_rerun:
        if running_marker_path.exists():
            running_marker_path.unlink()
        if done_marker_path.exists():
            done_marker_path.unlink()

    _update_runtime_config(
        BATCH_PROMPTS_DIR=Path(batch_prompts_dir),
        BATCH_NUM_PROMPTS=batch_num_prompts,
        BATCH_NUM_LAYERS=batch_num_layers,
        BATCH_NUM_HEADS=batch_num_heads,
        BATCH_RANGE_LABEL=batch_range_label,
        BATCH_RANGE_START=batch_range_start,
        BATCH_RANGE_END=batch_range_end,
        HEAD_PERIOD_HOMOLOGY_CSV_PATH=csv_path,
        HEAD_PERIOD_HOMOLOGY_REPRESENTATIVE_FIGURE_PATH=representative_figure_path,
        HEAD_PERIOD_HOMOLOGY_UNIFORM_FIGURE_PATH=uniform_figure_path,
        COMPACT_FIGURE_SIZE=compact_figure_size,
        BALANCED_HEAD_COUNT=balanced_head_count,
        CYCLE6_HEAD_FRACTION=cycle6_head_fraction,
        CYCLE6_PERIOD_RANGE=cycle6_period_range,
        CYCLE6_PERIOD_CENTER=cycle6_period_center,
        CYCLE6_STD_MAX=cycle6_std_max,
        HIGH_PERIOD_MIN=high_period_min,
        HIGH_PERIOD_STD_MAX=high_period_std_max,
        CYCLE6_PER_LAYER_TARGET=cycle6_per_layer_target,
        UNIFORM_HEAD_COUNT=uniform_head_count,
        PERIOD_MIN=period_min,
        PERIOD_MAX=period_max,
        RESPONSE_PERIOD_MIN=response_period_min,
        RESPONSE_PERIOD_MAX=response_period_max,
        MIN_RESPONSE_CYCLES=min_response_cycles,
        FFT_TOP_K=fft_top_k,
        ACF_TOL=acf_tol,
        ACF_MIN=acf_min,
        NEAR_TIE_RATIO=near_tie_ratio,
        LOW_CYCLE_THRESHOLD=low_cycle_threshold,
        HARMONIC_FOLDING_ENABLED=harmonic_folding_enabled,
        HARMONIC_FOLD_MIN_PERIOD=harmonic_fold_min_period,
        HARMONIC_FOLD_MAX_PERIOD_DIVISOR=harmonic_fold_max_period_divisor,
        HARMONIC_FOLD_MAX_MULTIPLE=harmonic_fold_max_multiple,
        HARMONIC_FOLD_TOL=harmonic_fold_tol,
        HARMONIC_FOLD_DECAY=harmonic_fold_decay,
        REMOVE_DC=remove_dc,
        APPLY_WINDOW=apply_window,
    )

    if use_markers:
        _write_head_period_homology_marker(running_marker_path, "running\n")

    try:
        rows, errors, periods = compute_head_period_homology_256()
        validate_head_period_homology(rows, errors, periods)
        write_head_period_homology_csv(rows, csv_path=HEAD_PERIOD_HOMOLOGY_CSV_PATH)
        representative_rows = plot_representative_head_period_homology(
            rows,
            figure_path=HEAD_PERIOD_HOMOLOGY_REPRESENTATIVE_FIGURE_PATH,
            figure_size=COMPACT_FIGURE_SIZE,
            head_count=BALANCED_HEAD_COUNT,
            cycle6_head_fraction=CYCLE6_HEAD_FRACTION,
            cycle6_period_range=CYCLE6_PERIOD_RANGE,
            cycle6_period_center=CYCLE6_PERIOD_CENTER,
            cycle6_std_max=CYCLE6_STD_MAX,
            high_period_min=HIGH_PERIOD_MIN,
            high_period_std_max=HIGH_PERIOD_STD_MAX,
            cycle6_per_layer_target=CYCLE6_PER_LAYER_TARGET,
            dpi=dpi,
        )
        uniform_rows = plot_uniform_head_period_homology(
            rows,
            figure_path=HEAD_PERIOD_HOMOLOGY_UNIFORM_FIGURE_PATH,
            figure_size=COMPACT_FIGURE_SIZE,
            head_count=UNIFORM_HEAD_COUNT,
            dpi=dpi,
        )
        if use_markers:
            _write_head_period_homology_marker(done_marker_path, "done\n")
    finally:
        if use_markers and running_marker_path.exists():
            running_marker_path.unlink()

    return {
        "rows": rows,
        "errors": errors,
        "periods": periods,
        "representative_rows": representative_rows,
        "uniform_rows": uniform_rows,
        "skipped": False,
        "running_marker_path": running_marker_path,
        "done_marker_path": done_marker_path,
    }


def maybe_run_256prompts_distribution(run_batch=False, **kwargs):
    if not run_batch:
        print("Set RUN_BATCH = True to execute the full 256-prompts distribution analysis.")
        return None
    return run_256prompts_distribution(**kwargs)
