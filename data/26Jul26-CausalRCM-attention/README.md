# Causal-rCM frame-level attention artifacts

The `.pt` files here are gitignored (`data/**/*.pt`). Regenerate them from the
`rcm` submodule of the NeurIPS2026 super-repo — `winbeau/rcm`, branch `main`:

```bash
# 72 latent frames (frames72/) — 285 pixel frames
PYTHONPATH=. uv run --no-sync python rcm/inference/wan2pt1_t2v_causal_infer.py \
    --distilled \
    --dit_path assets/checkpoints/Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt \
    --num_steps 4 --mid_t 15/16 5/6 5/8 \
    --first_chunk_t 1 --chunk_t 1 \
    --num_frames 285 --seed 0 \
    --prompt "A majestic eagle soaring through a cloudy sky, cinematic lighting" \
    --save_path output/attn_probe_72f.mp4 \
    --extract_attn_layers all \
    --attn_output_dir cache/attn/causal-rcm-1.3B-72f \
    --attn_tag c1-1_step4

# 21 latent frames (frames21/) — same command with --num_frames 81
```

Then copy `cache/attn/causal-rcm-1.3B-72f/*.pt` into `frames72/`.

Layout: one `.pt` per DiT layer, `layer<N>_c1-1_step4.pt`, N in 0..29.

## Schema

Identical to the Self-Forcing artifacts in `26Mar13-*` / `26Mar26-*`, so
`notebooks/26Mar26-PyramidForcing-frames72/attention_plot_utils.py` reads them
unmodified:

| key | value |
| --- | --- |
| `full_frame_attention` | `[num_heads, F, F]` fp16 — mean pre-softmax logits per (query frame, key frame) |
| `last_block_frame_attention` | `[num_heads, F]` fp16 |
| `is_logits` | `True` |
| `num_frames` / `num_heads` | 72 (or 21) / 12 |
| `frame_seq_length` | 1560 (480p 16:9 → latent 60×104, `H*W//4`) |
| `block_sizes` | all ones — `--chunk_t 1` puts one latent frame per chunk |
| `extraction_method` | `rcm-observer-pooled` |

Plus rcm-specific provenance: `model`, `dit_path`, `first_chunk_t`, `chunk_t`,
`num_steps`, `seed`, `resolution`, `capture_pass`.

## What the numbers are

Captured on the once-per-chunk `KVCacheMode.APPEND` forward, which runs at t=0
on the already-denoised chunk — clean context, exactly one observation per
chunk. The per-denoising-step dimension is **not** retained.

Values are logits, not post-softmax weights. The notebook softmaxes each causal
row itself when it needs attention *mass*.
