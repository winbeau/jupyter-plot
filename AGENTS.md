# AGENTS.md

## Project Layout
- This repo uses three canonical content roots only:
  - `data/`: input datasets, summary tables, and reusable CSV artifacts.
  - `notebooks/`: the canonical plotting logic and exploratory workflow.
  - `figures/`: final exported plots that are worth tracking in git.
- `figure_sources/` is no longer part of the project structure and should not be recreated.
- `outputs/` is for disposable/generated artifacts that do not need to be tracked.

## Current Worksets
- `26Feb2-mcm`
  - `notebooks/26Feb2-mcm/` contains the main historical plotting notebooks such as `fig01-...` through `fig13-...`, plus `task1`, `task2`, and `task3` notebooks.
  - `data/26Feb2-mcm/` contains the CSV sources those notebooks read from.
  - `figures/26Feb2-mcm/` contains the tracked exports, mostly PDF plus selected PNG companions.
- `26Mar13-PyramidForcingSweep`
  - `data/26Mar13-PyramidForcingSweep/` contains:
    - `vbench_pyramid-forcing-sweep.csv`
    - `total_lite_mean_matrix.csv`
    - `total_lite_std_matrix.csv`
  - `notebooks/26Mar13-PyramidForcingSweep/vbench_total_lite_sr_3d.ipynb` is the canonical plotting notebook.
  - `figures/26Mar13-PyramidForcingSweep/` contains the final exports:
    - `vbench_total_lite_sr_3d.png`
    - `vbench_total_lite_sr_3d.pdf`

## Data Source Rules
- Treat `data/` as the single source of truth for reusable input tables.
- If a notebook needs a CSV or matrix table, put it under the matching dated subdirectory in `data/`.
- Do not keep duplicate CSV copies alongside notebooks or figures.
- Prefer descriptive dated folders such as `data/<date-or-workset>/...` over flat top-level CSV files.

## Plotting Script Rules
- Canonical plotting logic lives in notebooks, not standalone `plot.py` files.
- If a plot needs configurable parameters, keep them near the top of the notebook in explicit config cells.
- Separate plotting configuration from data/path configuration when the notebook is non-trivial.
- Notebooks should read from `data/...` and write final exports to `figures/...`.
- If helper logic becomes large, keep it inside the notebook unless there is a clear reusable module need.

## Figure Output Rules
- Final deliverables go to `figures/<workset>/...`.
- Prefer PDF for archival/publication-quality output and add PNG when screen preview is useful.
- Keep figure filenames stable and descriptive so notebooks and git history stay easy to follow.
- If a historical figure has both PDF and PNG, both may be tracked in `figures/` when they serve different usage needs.

## Maintenance Notes
- Before adding a new plot, first decide its workset folder under `data/`, `notebooks/`, and `figures/`.
- When reorganizing content, migrate files into the canonical three-root structure rather than creating a fourth location.
- Avoid reintroducing duplicated artifacts that make `data/` and plotting source directories overlap.
