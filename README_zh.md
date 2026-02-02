# Jupyter 绘图工作站

这是一个以 Jupyter 为核心的绘图工作站，也是个人的绘图学习路线记录。仓库以清晰的目录结构保存笔记本、图片与数据，便于复现实验和横向对比。

## 目录结构

- `notebooks/`：按日期/任务组织的实验笔记本（主要工作区）。
- `figures/`：需要长期保留的最终图片输出。
- `data/`：笔记本用到的数据集。
- `main.py`：可选的临时脚本或辅助入口。

## 快速开始

环境要求：Python 3.12。

使用 `uv`（若已安装）：

```bash
uv sync
uv run jupyter lab
```

使用传统虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
jupyter lab
```

说明：本仓库通过 `pyproject.toml` 管理依赖。

## 输出与图片

- 过程产物建议放在 `outputs/`（默认不纳入版本管理）。
- 需要保留的图片请放入 `figures/`（会纳入版本管理）。
- `.gitignore` 会保留 `outputs/` 下的图片文件，方便记录过程渲染结果。

## 学习路线记录

笔记本按日期与任务拆分，强调单一主题、单一目标。建议在笔记本中记录关键假设与参数设置，并将阶段性“最终图”导出到 `figures/` 以便对比与回溯。

## 小建议

- 文件名尽量包含日期与主题。
- 最终图统一存放到 `figures/`。
- 在笔记本里写清楚数据来源与参数选择，保证可复现。
