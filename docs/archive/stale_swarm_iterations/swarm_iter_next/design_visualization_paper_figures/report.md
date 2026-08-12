# Design Report — Paper Figures & Tables (task_16)

## 1. What was implemented

A new CPU-only orchestrator script that regenerates the paper figures and tables from the evaluation JSON files already stored under `outputs/`.

The script handles the heterogeneous JSON schemas produced by different evaluation runs (e.g. `mpjpe_mm` vs `mpjpe`, `pck_50mm` vs `pck@50mm`) and normalizes them before generating figures and tables.

Generated artifacts:

- **Main MPJPE bar chart** (`docs/figures/icra2027/main_mpjpe_bar.png`)
- **Per-joint MPJPE bar charts** for every included model (`docs/figures/icra2027/per_joint_mpjpe_<model>.png`)
- **Robustness grid** (`docs/figures/icra2027/robustness_grid.png`)
- **Main results table** in Markdown and LaTeX (`docs/tables/icra2027/main_results.md`, `.tex`)
- **Robustness table** in Markdown (`docs/tables/icra2027/robustness.md`)

## 2. File paths

- `experiments/generate_paper_figures.py` — new orchestrator script
- `experiments/lib/paper_figures.py` — reused figure helpers (read-only)
- `experiments/lib/paper_tables.py` — reused table helpers (read-only)
- `docs/figures/icra2027/` — generated PNG figures
- `docs/tables/icra2027/` — generated Markdown/LaTeX tables
- `docs/swarm_iter_next/design_visualization_paper_figures/report.md` — this report

## 3. How to test / validate

Run the orchestrator from the project root:

```bash
python experiments/generate_paper_figures.py
```

Optional arguments:

```bash
python experiments/generate_paper_figures.py \
    --eval_dir outputs \
    --robustness outputs/robustness_residual_final5/robustness_report.json \
    --out_dir docs/figures/icra2027 \
    --table_dir docs/tables/icra2027
```

Smoke test passed on the current repository (Python 3.13, matplotlib 3.10.6, numpy 2.3.5). It produced 10 figure files and 3 table files in ~1 s on CPU.

Models included in the smoke run:

| Model | MPJPE (mm) | PA-MPJPE (mm) | AUC |
|---|---|---:|---:|
| metrics_residual_mpiinf3dhp | 10.46 | 8.93 | 0.9303 |
| eval_residual_final5 | 11.17 | 8.24 | 0.9256 |
| eval_crossview_residual_d64_h128 | 15.29 | 13.49 | 0.8981 |
| eval_ray_attention_temporal_residual_mpiinf3dhp_eval | 19.49 | 19.16 | 0.8701 |
| metrics_campe_mpiinf3dhp | 11.25 | 9.14 | 0.9250 |
| metrics_adaptive_mpiinf3dhp | 12.73 | 9.14 | 0.9151 |
| eval_residual_h36m_h128 | 5.74 | 3.99 | 0.9618 |
| metrics_campegraph_h36m_s5a02 | 0.62 | 0.70 | 0.9936 |

## 4. Expected impact

- Removes manual copy/paste of MPJPE values into the paper draft.
- Provides a single command to regenerate all quantitative figures/tables whenever a new checkpoint is evaluated.
- Makes the camera-ready figures consistent (fonts, colors, DPI) by centralizing the plotting code in the existing `experiments/lib/` helpers.
- Serves as a stepping stone toward the fully automated figure pipeline described in `docs/swarm_iter11_paper_figures_and_tables_report.md`.

## 5. Blockers / next steps

1. **Model size (params) is missing** from most evaluation JSONs, so the Params column in the generated tables is currently `0`. Future eval scripts should write `n_params`.
2. **Dataset mixing** — the current table groups MPI-INF-3DHP and Human3.6M results together. A `dataset` field or filename-based grouping should be added to produce separate MPI and H36M tables.
3. **PCK-vs-threshold curves** cannot be drawn yet because the per-frame/per-clip error arrays are not stored in the JSONs. The eval scripts should emit `pck_thresholds` and `pck_values` arrays.
4. **Qualitative 3D skeleton overlays** and **uncertainty/weight heatmaps** require per-sample inference data and are out of scope for this CPU-only figure generator. They can be added as a second stage that loads model weights and sample clips.
5. **LaTeX table caption is hard-coded** to "MPI-INF-3DHP cross-subject results"; it should be generalized once the table is split by dataset.
