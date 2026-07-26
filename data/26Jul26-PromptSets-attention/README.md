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

---

# Head period homology 结果

```bash
uv run python notebooks/26Jul26-PromptSets-attention/run_head_period_homology.py
```

对三个集合各跑一遍 `spectral_analysis_utils.run_256prompts_distribution`（它对
prompt 数是参数化的），判据全部沿用该模块既有常量，没有另立标准。产物：

| 文件 | 内容 |
| --- | --- |
| `head_period_homology_<set>_firstdiff_folded_top1_0_68.csv` | 每集合 360 行（30 层 × 12 头）的 `period_mean/std/min/max/q25/q75/dominant` |
| `head_period_homology_promptsets_merged.csv` | 三集合并排 + `is_cycle6` + `best_label` 对照 |
| `../../figures/26Jul26-PromptSets-attention/*.png/pdf` | representative / uniform 两张紧凑图，每集合一组 |

## 先记住一件事：周期是离散的

FFT 跑在 68 点的一阶差分序列上，所以周期取值只能是 `68/k`。周期 6 附近的相邻格点
是 **5.67 / 6.18 / 6.80** —— 分辨率约 0.6 帧。下面所有「Δ 小于 0.5 帧」「std=0」的
说法都要放在这个网格上理解：`std=0` 意思是**所有 prompt 都投给了同一个 bin**，不是
连续值恰好相等。

## 1. 跨 prompt 分布高度一致

`period_mean` 的相关性：

| 对比 | r | \|Δ\| 中位 | ≤0.5 帧 | ≤1.0 帧 |
| --- | --- | --- | --- | --- |
| FETV-128 vs Step-128 | +0.989 | 0.087 | 307/360 (85%) | 346/360 (96%) |
| FETV-128 vs ChronoMagic-150 | +0.977 | 0.127 | 276/360 (77%) | 321/360 (89%) |
| Step-128 vs ChronoMagic-150 | +0.975 | 0.124 | 273/360 (76%) | 325/360 (90%) |

三个集合的分布差异是很大的（英文短句 / 中文 / 英文延时摄影），|Δ| 中位数却只有
0.09–0.13 帧，**远小于一个 FFT 格点**。也就是说周期结构不是 MovieGen prompt 分布的
产物。

## 2. 6-cycle 头集合：150 个铁核 + 22 个边界

判据 `period_mean ∈ (5.8, 6.5)` 且 `period_std ≤ 1.5`（`CYCLE6_PERIOD_RANGE` /
`CYCLE6_STD_MAX`）：

| | 头数 |
| --- | --- |
| FETV-128 | 163 |
| Step-128 | 163 |
| ChronoMagic-150 | 159 |
| **三集合交集** | **150** |
| 三集合并集 | 172 |
| Jaccard | 0.872 |

两两 Jaccard 0.894–0.928。

**最强的那个数字**：那 150 个交集头，跨三个集合取 `period_std` 最大值，中位数是
**0.000** —— 超过一半的核心头在全部 406 个 prompt 上投出的都是同一个 FFT bin。
对比之下 22 个边界头的同一指标中位是 **1.753**。分布是双峰的，不是连续退化。

## 3. 阈值不脆

扫 `std_max`，三集合 Jaccard：

| std_max | 0.5 | 0.8 | 1.0 | 1.2 | **1.5** | 2.0 | 2.5 | ∞ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Jaccard | 0.832 | 0.835 | 0.867 | 0.874 | **0.872** | 0.876 | 0.872 | 0.788 |

0.5→2.5 之间只在 0.83–0.88 波动，现用的 1.5 不在悬崖边上。但**完全丢掉 std 条件会掉到
0.788** —— 说明 `period_std` 这一项确实在挡掉一批不稳的头，值得保留。

## 4. 与 `best_labels.csv` 的对照

| set | 6-cycle ∩ Wave | Wave 召回 | 6-cycle 精度 |
| --- | --- | --- | --- |
| FETV-128 | 118 | 75.6% | 72.4% |
| Step-128 | 121 | 77.6% | 74.2% |
| ChronoMagic-150 | 115 | 73.7% | 72.3% |

（`best_labels.csv`：Anchor 172 / Wave 156 / Veil 32。）

**这不是准确率。** 6-cycle 只是 Wave 的一个充分不必要表征 —— Wave 的定义是注意力随
时间振荡，并不要求周期恰好落在 6 附近。三个集合的召回一致在 74–78%，说明约四分之一
的 Wave 头用别的周期振荡，或者根本不靠周期性被划进去。要把这张表写进论文，得先确认
`best_labels.csv` 当初是怎么定的标签。

---

# 全层热力图（两阶段流程的第二阶段）

提取（`Pyramid-Forcing-Preview/experiments/extract_attn/run_extraction_streaming.py`）
和绘图是分开的，绘图脚本只认 `<root>/run_%03d/layer<L>.pt` 这个布局，换 prompt /
换模型 / 换集合都不用改代码：

```bash
uv run python notebooks/26Jul26-PromptSets-attention/plot_all_layer_heatmaps.py \
    --run-dir data/26Jul26-PromptSets-attention/fetv128/run_000 \
    --run-dir data/26Jul26-PromptSets-attention/step128/run_000 \
    --run-dir data/26Jul26-PromptSets-attention/chronomagic150/run_000 \
    --out-root figures/26Jul26-PromptSets-attention
```

渲染器直接复用 `26Mar26-PyramidForcing-frames72/attention_plot_utils.py` 的
`render_attention_heatmaps` —— 就是 `26Jul26-CausalRCM-attention/layer15_2d_heatmap_all_heads.png`
那张图的渲染器，只换配色范围。产物：三个集合 × 30 层 = **90 张**，每张是一层的
12 个头（3×4 三角热力图），PNG + PDF，共约 32 MB，落在
`figures/26Jul26-PromptSets-attention/<set>_run000/layer<L>_2d_heatmap_all_heads.{png,pdf}`。

当前用的是各集合的 `run_000`（单 prompt，与参考图同性质）。跨 prompt 的均值 /
方差分布是后续工作，脚本已经按可复用的方式写好，届时只换 `--run-dir` 即可。

## 配色范围：两个都踩过的坑

`--color-range auto` 取的是**池化后全部因果下三角值**的 p1/p99，本批数据是
**[-15.33, 13.95]**。这个数不是随便定的，两个更朴素的做法都会翻车：

1. **朴素全局 min/max** 给出 `[-36.0, 100.6]`。但 `|v| > 20` 只占 **1.26%**，而且
   几乎全来自 layer 28/29 —— 用它定色标会把 98% 的数据压进色标中段，图基本全白。
   （保留为 `--color-range minmax`，仅供复现这个失败模式。）
2. **逐层取 p1 的最小值 / p99 的最大值** 给出 `[-28.8, 68.7]`，看着像分位数其实
   等于又把 layer28/29 那两个极端层挑了出来，离群问题原封不动。分位数**必须在
   池化后的样本上取**。

顺带纠正一个容易犯的错：本批 logits 的范围不是 `[-6.5, 11.1]` —— 那只是 **layer 0**
的范围，深层大得多。p1/p99 落在 ±15 附近，说明参考图那个写死的 ±18 其实是贴合数据的，
所以这批图与既有的 rCM / frames72 图仍可并排比较。要严格对齐既有图就传
`--color-range fixed18`。

只统计因果下三角：上三角是未来帧，恒为结构性 0，算进分位数会把中位数往 0 拽。

---

# 提取性能

全部实测于空闲的 H200 NVL（GPU 4），同 prompt 同 seed，72 潜帧：

| 配置 | 提取耗时 |
| --- | --- |
| pooled，1 层 | 33.6 s |
| **pooled，30 层** | **33.5 s** |
| chunked，1 层 | 36.6 s |
| chunked，30 层 | 114.0 s |

另有 wall 179.2 s − 提取 114.0 s ⇒ **单进程启动约 65 s**（uv + import + 加载
5.6 GB checkpoint + 建模型）。

生产跑：406 个 prompt × 30 层，5 张卡，**51 分钟**（02:48:32 → 03:39:50）。

## 快在哪，按贡献排序

**① 一次推理抓 30 层，而不是每层重跑一遍。** 这是主因。前身
`AdaHead/experiments/extract_attn/script_extract_frames72_l0_l29.sh` 是**每层一个
独立进程**，同一个 prompt 的 24 block 自回归 rollout 要完整跑 30 遍、checkpoint
重新加载 30 次。按实测 65 s 启动 + 33.6 s 推理，单卡 30 层 ≈ **49 分钟/prompt**。

**② pooled 取代 chunked：实测 3.4×。** 30 层 114.0 s → 33.6 s。逐层看 chunked
每层加 3.0 s、pooled 加约 0；捕获成本 114.0 − 33.6 = 80.4 s 被压到接近 0，剩下
全是 DiT+VAE —— 这就是为什么抓 30 层和抓 1 层一样快。

**③ 丢掉 block 内 4/5 的重复捕获。** 这条是**从代码路径推的，不是实测**：
Self-Forcing 每个 block 跑 4 次加噪 + 1 次 clean forward，`(q_frames, k_frames)`
完全相同，旧代码 5 次全算完再由聚合器扔掉 4 次。据此旧 chunked 的捕获成本应是
80.4 × 5 ≈ 402 s，30 层合计约 436 s/prompt。

顺带：AdaHead 脚本里那一大段显存估算和 `--force` 开关（注释写「60 帧运行时
30 GB、保存峰值 36 GB」）是被 chunked 每次物化 `[12, 1560, 4680]` fp32 分数块
（单块 350 MB）逼出来的。pooled 的中间量只有 KB 级，那套估算和警告不再需要。

## 没做的对比

MovieGen-256 那批的 homology CSV **不在本地** —— `figures/**/*.csv` 被 gitignore，
当初只提交了 PDF/PNG，`data/26May4-PyramidForcing-multihead/prompts256` 的 `.pt` 也
没有。所以本文档**没有**给出与 256-prompt 基线的数值对比。要补的话需要先重跑那批提取。
