# Jupyter Plot Workstation

A small Jupyter-based plotting workstation and a personal learning path for data visualization. This repo keeps notebooks, figures, and notes in a predictable structure so experiments are easy to reproduce and compare.

## Structure

- `notebooks/`: dated tasks and experiments (the main working area).
- `figures/`: curated exports that are worth keeping.
- `data/`: datasets used by notebooks.
- `main.py`: optional scratch runner / helpers.

## Quickstart

Requirements: Python 3.12.

Using `uv` (recommended if you already have it):

```bash
uv sync
uv run jupyter lab
```

Or with a standard venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
jupyter lab
```

Note: this repo uses `pyproject.toml` for dependencies.

## Outputs and figures

- Generated artifacts should go to `outputs/` (ignored by git).
- Final or curated images should go to `figures/` (tracked by git).
- The gitignore keeps image files under `outputs/` so you can still preserve renders when needed.

## Learning path notes

Notebooks are organized by date/task so the progression is easy to follow. The intention is to keep each notebook focused on a single technique or visualization goal, with figures saved for later comparison.

## Tips

- Keep notebook names descriptive and date-stamped.
- Export any “final” plots to `figures/` for versioning.
- Record assumptions and parameter choices in the notebook so results are reproducible.
