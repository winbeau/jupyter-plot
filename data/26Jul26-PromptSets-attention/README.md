# 跨 prompt 集的 Self-Forcing 帧级注意力

用来回答「头分类在不同提示词下稳不稳」的原始数据。三个 prompt 集，各自 30 层 ×
12 头 × 72 帧：

| 子目录 | prompt 集 | 条数 |
| --- | --- | --- |
| `fetv128/` | FETV-128 | 128 |
| `step128/` | Step-Video-T2V-Eval | 128 |
| `chronomagic150/` | ChronoMagic-Bench-150 | 150 |

`.pt` 文件被 gitignore（`data/**/*.pt`）。重新生成它们要用 NeurIPS2026 超级仓库的
`Pyramid-Forcing-Preview` submodule —— `IF-LAB-PKU/Pyramid-Forcing-Preview`，
分支 `main`：

```bash
cd Pyramid-Forcing-Preview
python experiments/extract_attn/fetch_prompt_sets.py     # 写 prompts/*.txt

./experiments/extract_attn/run_extraction_batch.sh \
    prompts/FETV_num128.txt             cache/attn/fetv128         0,1,2,3,4 72
./experiments/extract_attn/run_extraction_batch.sh \
    prompts/StepVideoT2VEval_num128.txt cache/attn/step128         0,1,2,3,4 72
./experiments/extract_attn/run_extraction_batch.sh \
    prompts/ChronoMagicBench_num150.txt cache/attn/chronomagic150  0,1,2,3,4 72
```

然后把 `cache/attn/<set>/` 整个拷进对应子目录。每个子目录里另有 `prompts_shard*.csv`
（`index,run,prompt`）和 `manifest_shard*.json` 记录当次的配置。

## 布局

```
<set>/run_000/layer0.pt … layer29.pt
      run_001/…
      …
```

`run_%03d` 的编号是 prompt 在 `prompts/*.txt` 里的行号（0 起）。提取按 prompt 序号
轮转分片到多卡，分片不改变编号，所以合起来是连续的。
`notebooks/26May4-PyramidForcing-multihead/spectral_analysis_utils.py` 的
`compute_head_period_homology_256()` 直接吃这个布局，把 `BATCH_PROMPTS_DIR` 指过来、
`BATCH_NUM_PROMPTS` 改成 128 / 128 / 150 即可。

## Schema

与 `26Mar26-*` / `26Jul26-CausalRCM-attention` 一致，`attention_plot_utils.py` 和
`spectral_analysis_utils.py` 都不用改：

| key | value |
| --- | --- |
| `full_frame_attention` | `[12, 72, 72]` fp16，(query frame, key frame) 的 pre-softmax logit 均值 |
| `last_block_frame_attention` | `[12, 72]` fp16，在最后一个 block 的 3 个 query 帧上取平均 |
| `is_logits` | `True` |
| `num_frames` / `num_heads` | 72 / 12 |
| `frame_seq_length` | 1560（480p 16:9 → latent 60×104，`H*W//4`） |
| `block_sizes` | `[3] * 24`（`num_frame_per_block=3`，`independent_first_frame=false`） |
| `extraction_method` | `streaming-pooled-first` |

外加 provenance：`prompt` / `prompt_index` / `prompt_source` / `layer_index` /
`config_path` / `checkpoint_path` / `seed` / `capture_mode` / `capture_method` /
`capture_pass`。

## 数值是什么

Self-Forcing 基线（`configs/self-forcing.yaml`，**没有** Pyramid Forcing 的剪枝缓存），
Wan2.1-T2V-1.3B + `self_forcing_dmd.pt` 的 EMA 权重，`seed=42`，噪声在每个 prompt 前
重置，所以 prompt 之间只有文本不同。

值是 **logits，不是 softmax 后的权重** —— notebook 需要注意力*质量*时自己对因果行
做 softmax。

每个 block 跑 4 次加噪 forward + 1 次 clean KV 写入 forward，这里记的是**第一次**
（噪声最高那一步），与 `26Mar26-*` 的既有产物一致。rCM 那批记的是 clean pass，
两者的观测点不同，跨集合比较时注意。

## 三个 prompt 集的坑

- **FETV-128 是我们自己抽的**，上游 `llyx97/FETV` 只有 619 条、没有 128 子集；
  这里用固定步长 `round(i·619/128)`。它的 prompt 平均约 10 词，比
  MovieGenVideoBench 的约 107 词短一个数量级 —— 与 `prompts256` 那批对比时，
  prompt 长度不是配平的。
- **Step-128 全是中文**，上游没有英文版。Wan2.1 的 UMT5 text encoder 支持中文，
  能直接跑，但相对另外两个集合存在语种偏移。
- **ChronoMagic-150 全是 "Time-lapse of …"** 的延时摄影描述，题材本身很窄。
