# CVPR 2027 Paper Figures & Tables Plan

> **Scope:** Inventory of all figures and tables needed for the CVPR 2027 submission, plus the exact scripts, data, and deadlines to produce them.  
> **Status:** draft plan (2026-08-11).  
> **Related docs:** `docs/cvpr2027_submission_checklist.md`, `docs/roadmap_cvpr2027.md`, `docs/paper_results_table.md`, `docs/results_true_gt_h36m.md`, `docs/results_true_gt_shelf_campus.md`.

---

## 1. Design conventions

| Item | Convention |
|------|------------|
| Resolution | **≥300 dpi** PNG for final assets; keep SVG/PDF source where possible. |
| Fonts | Use `matplotlib.rc('font', family='serif', size=11)` to match the CVPR template. |
| Figure width | Single column ≈ 3.3 in; double column ≈ 6.8 in; full page ≈ 7.0 in. |
| Color palette | DLT = `#d62728`, Iskakov = `#2ca02c`, v25 = `#1f77b4`, v80 = `#ff7f0e`, v57 = `#9467bd`. Re-use these colors across every figure. |
| Data policy | **All numbers must come from true-GT / non-circular protocols.**  Never use the old circular `data/h36m_hf/*.npz` or `data/webbridge/h36m*.npz` numbers in any figure or table. |
| Output home | `docs/figures/icra2027/` for PNGs; `docs/tables/icra2027/` for generated `.md`/`.tex`. |

---

## 2. Main-paper figures

| ID | Figure / Panel | Purpose | Data / artifacts needed | Generation command / script | Output path | Status | Deadline |
|----|------------------|---------|------------------------|-----------------------------|-------------|--------|----------|
| **Fig. 1** | Architecture / pipeline diagram | Show the geometry-first fusion flow: 2D keypoints → ray tokenisation → geometry-aware attention → weighted triangulation → residual refinement. | None (schematic). Update `experiments/draw_architecture_figure.py` to the v25/v46/v57 stack and export at 300 dpi. | `python experiments/draw_architecture_figure.py` | `docs/figures/architecture.png` | ⚠️ exists but outdated | 2026-10-31 |
| **Fig. 2** | True-GT main-results bar chart | Compare DLT, Iskakov, v25, v80, v57 on the four true-GT benchmarks (H36M, MPI-INF-3DHP, Shelf/Campus, AIST++). | Final eval JSONs with `mpjpe_mm`. | Extend `experiments/draw_results_bar_chart.py` or call `experiments/generate_paper_figures.py` with the updated result list. | `docs/figures/true_gt_main_results_bar.png` | 🔴 blocked by missing MPI detected-2D & final medium runs | 2026-09-01 |
| **Fig. 3** | Sparse-view `MPJPE@k` curves | Demonstrate robustness as view count drops (k = 2..14 on MPI, 2..4 on H36M). | Trained checkpoint + manifest + `.npz`. | `python experiments/eval_variable_views.py --checkpoint <ckpt> --dataset_manifest <txt> --output_csv outputs/variable_views.csv` then `python experiments/plot_variable_views.py --input outputs/variable_views.csv --output docs/figures/variable_views_<model>_<dataset>.png` | `docs/figures/variable_views_*.png` | 🟡 scripts ready; need final checkpoints | 2026-10-01 |
| **Fig. 4** | Robustness grid (noise / occlusion / outliers) | Show degradation under 2D noise, random joint occlusion, and 2D outliers. | Robustness report JSON from a trained model. | `python experiments/eval_residual_robustness_mpiinf3dhp_v1.py --checkpoint <ckpt> --val <npz> --out_dir outputs/robustness_<name>` then `python experiments/draw_robustness_chart.py` or `python experiments/generate_paper_figures.py --robustness outputs/robustness_<name>/robustness_report.json` | `docs/figures/robustness_final5.png` / `docs/figures/icra2027/robustness_grid.png` | 🟡 scripts ready; need true-GT robustness run | 2026-10-01 |
| **Fig. 5** | Calibration-perturbation heatmap | MPJPE under rotation / translation / focal-length / principal-point perturbations. | Eval JSON or CSV per perturbation level. | Extend `experiments/eval_omniview_fusion_v5_robustness.py` or write a new `experiments/plot_calibration_perturbation_heatmap.py`. | `docs/figures/calibration_perturbation_heatmap.png` | 🔴 not started | 2026-10-01 |
| **Fig. 6** | Cross-domain transfer matrix | Heatmap of source→target MPJPE for H36M, MPI, AIST++, Shelf/Campus. | Cross-trained checkpoints and per-target eval JSONs. | New script: `experiments/plot_cross_domain_transfer.py` reading `outputs/cross_domain_eval.json`. | `docs/figures/cross_domain_transfer_matrix.png` | 🔴 not started | 2026-10-15 |
| **Fig. 7** | Per-joint MPJPE bar chart | Identify which joints hurt most on the best model. | Eval JSON with `per_joint_mpjpe`. | `python experiments/generate_paper_figures.py --eval_dir outputs/eval_jsons` (uses `experiments/lib/paper_figures.py::draw_per_joint_mpjpe`). | `docs/figures/icra2027/per_joint_mpjpe_<model>.png` | 🟡 lib ready; need final eval JSONs | 2026-10-31 |
| **Fig. 8** | Qualitative success/failure panels | Side-by-side 2D projections, predicted 3D pose, and error map. | Selected frames + predictions + GT. | `python scripts/visualize_multiview_pose.py` or `python experiments/visualize_fusion.py`; curate 3–4 representative frames. | `docs/figures/qualitative_<dataset>.png` | 🔴 not started | 2026-10-31 |

---

## 3. Main-paper tables

| ID | Table | Purpose | Data / artifacts needed | Generation command / script | Output path | Status | Deadline |
|----|-------|---------|------------------------|-----------------------------|-------------|--------|----------|
| **Tab. 1** | True-GT main results | MPJPE/PA-MPJPE across H36M, MPI-INF-3DHP, Shelf/Campus, AIST++ for DLT/Iskakov/v25/v80/v57. | Final numbers from `docs/paper_results_table.md` and eval JSONs. | `python experiments/generate_paper_figures.py` writes `docs/tables/icra2027/main_results.{md,tex}`; hand-finish in the LaTeX source. | `docs/tables/icra2027/main_results.tex` | 🟡 draft in `paper_results_table.md`; needs final true-GT MPI & AIST medium | 2026-09-01 |
| **Tab. 2** | Dataset protocol summary | Non-circular protocol, views, joints, train/val split for each benchmark. | Manifests in `configs/splits/`. | Hand-written; update from `docs/roadmap_cvpr2027.md` §5. | Inline in paper / `docs/tables/dataset_protocol.tex` |  exists in prose; needs LaTeX | 2026-09-01 |
| **Tab. 3** | Ablation study | v25 full vs. no geometry attention / no depth-proposal head / no GeoBA / regularization variants. | Ablation eval JSONs. | Extend `experiments/lib/paper_tables.py` + `experiments/generate_paper_figures.py`. | `docs/tables/icra2027/ablation.tex` | 🔴 blocked by ablation runs | 2026-10-15 |
| **Tab. 4** | Robustness table | MPJPE under noise, occlusion, outliers (numerical companion to Fig. 4). | `robustness_report.json`. | `experiments/generate_paper_figures.py` writes `docs/tables/icra2027/robustness.md`. | `docs/tables/icra2027/robustness.md` |  lib ready; need report | 2026-10-01 |
| **Tab. 5** | Cross-domain transfer table | Source train → target val MPJPE. | Cross-dataset eval JSONs. | New function in `experiments/lib/paper_tables.py`. | `docs/tables/icra2027/cross_domain.tex` | 🔴 not started | 2026-10-15 |
| **Tab. 6** | SOTA comparison | Iskakov, VoxelPose, MVPose, DLT on identical splits. | Published numbers or reproduced baselines. | Hand-written; update from `docs/results_iskakov_baseline.md`. | `docs/tables/icra2027/sota_comparison.tex` | 🟡 Iskakov done; VoxelPose/MVPose open | 2026-10-15 |
| **Tab. 7** | Model complexity & latency | Params, FLOPs, RTX-4090 latency for each variant. | Model summaries + timing script. | Add to `experiments/lib/paper_tables.py`; run `experiments/profile_model_latency.py` if available. | `docs/tables/icra2027/complexity.tex` | 🔴 not started | 2026-10-31 |

---

## 4. Supplementary figures & tables

| ID | Asset | Purpose | Generation path | Status |
|----|-------|---------|-----------------|--------|
| **Supp. Fig. 1** | Training / convergence curves | Show divergence of v25/v80/v57 on true-GT H36M to justify early stopping. | `python scripts/plot_training_curves.py outputs/<run>.log --out docs/figures/supplementary/training_<model>.png` | 🟡 script ready |
| **Supp. Fig. 2** | Per-joint error maps for every method | Same as Fig. 7 but for all baselines. | `experiments/generate_paper_figures.py` | 🟡 lib ready |
| **Supp. Fig. 3** | PCK@threshold curves | Per-model PCK curves. | `experiments/generate_paper_figures.py` | 🟡 lib ready |
| **Supp. Fig. 4** | Additional qualitative examples | More failure modes and cross-dataset examples. | `scripts/visualize_multiview_pose.py` |  not started |
| **Supp. Tab. 1** | Full per-epoch H36M true-GT table | Epoch-by-epoch MPJPE for v25/v80/v57. | Copy from `docs/results_true_gt_h36m.md`. | 🟡 exists |
| **Supp. Tab. 2** | Hyperparameter settings | Exact configs for main models. | Export from `outputs/*.config.json`. | 🟡 configs exist |

---

## 5. Generation workflow

Run the pipeline in this order.  **GPU training is currently blocked by the running `v25_true_gt_baseline_fix` ablation; only CPU-only plotting/data steps should run in parallel.**

1. **Finalize true-GT evaluations** (GPU, when free)
   - Run H36M true-GT medium for v25/v80/v57 and Iskakov.
   - Complete MPI-INF-3DHP real detected-2D generation (`scripts/generate_mpi_detected_2d.py`) and run baselines.
   - Run AIST++ medium and Shelf/Campus long runs.
   - Write per-model eval JSONs to `outputs/eval_jsons/`.

2. **Generate canonical figures & tables** (CPU)
   ```bash
   python experiments/generate_paper_figures.py \
       --eval_dir outputs/eval_jsons \
       --robustness outputs/robustness_residual_final5/robustness_report.json \
       --out_dir docs/figures/icra2027 \
       --table_dir docs/tables/icra2027
   ```
   This produces: main bar chart, per-joint charts, PCK curves, robustness grid, and the main/robustness tables.

3. **Generate sparse-view curves** (GPU + CPU)
   ```bash
   # H36M
   bash scripts/eval_variable_views_h36m_true_gt/run_v25.sh
   bash scripts/eval_variable_views_h36m_true_gt/run_v80.sh
   bash scripts/eval_variable_views_h36m_true_gt/run_v57.sh

   # MPI
   python experiments/eval_variable_views.py \
       --model_class omniview_v5 \
       --checkpoint outputs/<best>.pth \
       --config outputs/<best>.config.json \
       --dataset_manifest <manifest> \
       --output_csv outputs/variable_views_<model>.csv

   # Plot
   python experiments/plot_variable_views.py \
       --input outputs/variable_views_<model>.csv \
       --output docs/figures/variable_views_<model>_<dataset>.png
   ```

4. **Generate robustness data** (GPU + CPU)
   ```bash
   python experiments/eval_residual_robustness_mpiinf3dhp_v1.py \
       --checkpoint outputs/<best>.pth \
       --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
       --clip_len 13 --batch_size 8 \
       --out_dir outputs/robustness_<name>

   python experiments/draw_robustness_chart.py
   # or
   python experiments/generate_paper_figures.py --robustness outputs/robustness_<name>/robustness_report.json
   ```

5. **Generate cross-domain transfer matrix** (GPU + CPU)
   - Train / evaluate source→target pairs.
   - Run a new `experiments/plot_cross_domain_transfer.py`.

6. **Architecture & qualitative figures** (CPU)
   - Update `experiments/draw_architecture_figure.py` and run it.
   - Curate frames and run `scripts/visualize_multiview_pose.py`.

7. **Camera-ready polish**
   - Re-render all PNGs at 300 dpi.
   - Run `python scripts/check_figure_resolution.py` (to be created) to enforce ≥300 dpi.
   - Compile LaTeX tables with the CVPR `.bst` and template.

---

## 6. Open blockers that affect figures/tables

| Blocker | Impact on this plan | Owner / next action |
|---------|---------------------|---------------------|
| MPI-INF-3DHP detected-2D alignment | Real detected-2D `.npz` ready, but DLT baseline ~326–400 mm; blocks MPI main results (Tab. 1, Fig. 2) and SOTA table | Diagnose/fix camera/label coordinate-frame mismatch; re-run DLT until ~20–30 mm |
| v25/v80/v57 H36M true-GT divergence / overfitting | Main H36M numbers are provisional (best epoch only) | Wait for ablation queue; add EMA/SWA/regularization runs |
| AIST++ & Shelf/Campus medium/long runs not done | Cross-domain transfer (Fig. 6, Tab. 5) lacks data | Run when GPU free |
| VoxelPose / MVPose baselines not implemented | SOTA table (Tab. 6) is incomplete | Add configs/scripts; run when GPU free |
| No calibration-perturbation heatmap script | Fig. 5 cannot be generated yet | Write `experiments/plot_calibration_perturbation_heatmap.py` |

---

## 7. Quick command reference

```bash
# 1. Regenerate all canonical figures/tables (CPU only)
python experiments/generate_paper_figures.py \
    --eval_dir outputs/eval_jsons \
    --robustness outputs/robustness_residual_final5/robustness_report.json \
    --out_dir docs/figures/icra2027 \
    --table_dir docs/tables/icra2027

# 2. Architecture diagram
python experiments/draw_architecture_figure.py

# 3. Sparse-view curve
python experiments/plot_variable_views.py \
    --input outputs/variable_views.csv \
    --output docs/figures/variable_views.png

# 4. Robustness figure from a report
python experiments/draw_robustness_chart.py

# 5. Training-curve supplementary figure
python scripts/plot_training_curves.py \
    outputs/omniview_fusion_v25_h36m_true_gt_medium.log \
    --out docs/figures/supplementary/training_v25_h36m.png
```

---

## 8. Checklist before submission

- [ ] Every figure is ≥300 dpi and uses the agreed color palette.
- [ ] Every plotted number is from a non-circular / true-GT source.
- [ ] `docs/paper_results_table.md` and the generated LaTeX tables match exactly.
- [ ] All eval JSONs referenced in the generation scripts exist under `outputs/eval_jsons/`.
- [ ] Cross-domain transfer table has at least one completed source→target pair.
- [ ] Supplementary video storyboard is aligned with the main figures (optional).
