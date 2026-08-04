# Iter11 Paper Figures & Tables Roadmap

## 1. Current state

The repo already produces a handful of paper-ready visuals under `docs/figures/`:

- `architecture.png` — hand-drawn block diagram from `experiments/draw_architecture_figure.py`
- `mpi_mpjpe_bar.png` — static bar chart from `experiments/draw_results_bar_chart.py`
- `robustness_final5.png` — 1×3 robustness curves from `experiments/draw_robustness_chart.py`

These are one-off scripts whose data are hard-coded (MPJPE values in the Python file, JSON paths fixed in the robustness script). The paper draft (`docs/paper_draft_icra_cvpr_2027.md`) also contains hand-edited Markdown/LaTeX tables for main results, ablations, robustness, and runtime.

**What is missing for ICRA/CVPR 2027 camera-readiness:**

1. No single entry point that regenerates *all* figures and tables from raw evaluation JSON/`.npz` outputs.
2. No per-joint or per-view diagnostic plots for the new `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` model.
3. No PCK-vs-threshold curves, error distributions, or qualitative 3D pose overlays.
4. No automated table generation (LaTeX/Markdown) from evaluation metrics.
5. No visualization of the new components: predicted uncertainty (`log_var`), per-view DLT weights, and Gauss-Newton refinement convergence.

## 2. Concrete, implementable improvements

### 2.1 Create a unified figure/table orchestrator

Add `experiments/generate_paper_figures_and_tables.py`. It should:

- Read a YAML config listing which checkpoints/eval-JSON files correspond to which model variants.
- Generate every figure below into `docs/figures/icra2027/`.
- Emit `docs/tables/icra2027/main_results.md` and `docs/tables/icra2027/main_results.tex`.
- Print a checklist so the author knows every figure is up-to-date.

The helpers are split into `experiments/lib/paper_figures.py` and `experiments/lib/paper_tables.py`, and a working orchestrator is already in `experiments/generate_paper_figures_and_tables.py`.

### 2.2 New figures needed

| Figure | Source data | Purpose |
|--------|-------------|---------|
| **Fig. 1 — architecture** (updated) | Static matplotlib | Include uncertainty, GN triangulation, and residual head. |
| **Fig. 2 — main results bar chart** | Eval JSONs | Replaces hard-coded `draw_results_bar_chart.py`. |
| **Fig. 3 — per-joint MPJPE** | `per_joint_mpjpe` | Shows failure modes (wrists, ankles, spine). |
| **Fig. 4 — PCK vs threshold** | `pck_auc()` | Standard curve; yields AUC visually. |
| **Fig. 5 — error distribution** | Per-frame MPJPE | Demonstrates variance and outliers. |
| **Fig. 6 — robustness grid** | Robustness JSON | Cleaner grid with combined perturbations. |
| **Fig. 7 — uncertainty/weight heatmap** | `weights`, `log_var` | Validate low weight for occluded views. |
| **Fig. 8 — Gauss-Newton convergence** | Triangulation error | Justify the learned GN step. |
| **Fig. 9 — runtime scaling** | Benchmark JSON | Batch vs latency/throughput. |
| **Fig. 10 — qualitative 3D poses** | MPI/H36M clips | Side-by-side GT, DLT, full model. |

### 2.3 Automated table generation

Add functions inside the orchestrator that read evaluation JSONs and emit:

- **Table 1 — Main MPI-INF-3DHP / H36M results** (model, params, MPJPE, PA-MPJPE, PCK@50/100/150, AUC).
- **Table 2 — Ablation study** (residual on/off, temporal on/off, cross-view on/off, uncertainty on/off, GN vs DLT).
- **Table 3 — Robustness perturbations** (noise std, occlusion rate, outlier rate, combined).
- **Table 4 — Runtime** (batch size, latency, throughput, memory).
- **Table 5 — Cross-dataset generalization** (MPI → Shelf/Campus).

Tables should be emitted as both Markdown and `booktabs` LaTeX so the draft stays in sync.

### 2.4 Standardize evaluation JSON schema

Every `eval_*.py` script should write a JSON file of this form:

```json
{
  "model": "RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1",
  "dataset": "mpi_inf_3dhp_s2_seq1",
  "mpjpe_mm": 11.17,
  "pa_mpjpe_mm": 8.24,
  "pck_50": 1.000,
  "pck_100": 1.000,
  "pck_150": 1.000,
  "auc": 0.926,
  "per_joint_mpjpe": [...],
  "per_view_mpjpe": [...],
  "pck_thresholds": [...],
  "pck_values": [...],
  "n_params": 243000,
  "date": "2026-08-04"
}
```

## 3. Recommended experiments to run

1. **Evaluate the new advanced model** on MPI-INF-3DHP S2 Seq1 using `train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py`.
   - Metrics: MPJPE, PA-MPJPE, PCK@50/100/150, AUC, per-joint MPJPE.
   - Save to: `outputs/eval_advanced_v1_mpi.json`.

2. **Generate the full figure set** by running the new orchestrator after all evals complete.

3. **Robustness sweep** on the advanced model (noise, occlusion, outliers, combined) using the existing `eval_residual_robustness_mpiinf3dhp_v1.py` harness but adapted to the new model.

4. **Runtime benchmark** with `experiments/benchmark_inference_v3.py` on the advanced model.

5. **Cross-dataset zero-shot** on Shelf/Campus to populate the generalization table.

## 4. Metrics to track

- **MPJPE / PA-MPJPE** on MPI-INF-3DHP and H36M for every variant.
- **PCK@50/100/150 mm** and **AUC** for completeness.
- **Per-joint MPJPE** to guide visualizations.
- **Mean predicted uncertainty per view/joint** to sanity-check the uncertainty head.
- **Gauss-Newton reprojection RMSE** before/after refinement.
- **Runtime**: latency (ms/clip) and throughput (clips/s) at batch sizes 1, 4, 8, 16.
- **Model size** (parameters and FLOPs if possible).

## 5. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| New advanced model still converging and may not yet beat 11.17 mm. | Run the figure pipeline on existing best checkpoint first; add advanced-model figures as soon as training finishes. |
| Hard-coded figure scripts drift out of sync with final numbers. | The orchestrator reads only JSON eval outputs; never hard-code metrics. |
| Shelf/Campus pseudogt not ready. | Leave placeholder rows in tables and skip if data are missing. |
| Matplotlib 3D pose renders look unprofessional. | Use consistent skeleton color scheme and fixed camera angle; generate PDF vector figures. |
| LaTeX table formatting breaks. | Generate Markdown first; LaTeX is a second output derived from the same data. |

## 6. Suggested file structure

```text
experiments/generate_paper_figures_and_tables.py   # new orchestrator
experiments/lib/paper_figures.py                  # figure helpers
experiments/lib/paper_tables.py                   # LaTeX/Markdown table helpers
docs/figures/icra2027/                             # output figures
docs/tables/icra2027/                              # output tables
```

## 7. Code sketch

```python
# experiments/generate_paper_figures_and_tables.py
import json
import matplotlib.pyplot as plt
from pathlib import Path
from motionflow_mv.eval.metrics import mpjpe, pa_mpjpe, pck, pck_auc, per_joint_mpjpe

FIG_DIR = Path("docs/figures/icra2027")
TABLE_DIR = Path("docs/tables/icra2027")


def load_evals(eval_dir: Path):
    return {p.stem: json.loads(p.read_text()) for p in eval_dir.glob("*.json")}


def draw_main_bar_chart(results: dict, out_path: Path):
    names = list(results.keys())
    scores = [results[k]["mpjpe_mm"] for k in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, scores)
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("MPI-INF-3DHP cross-subject MPJPE")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")


def main():
    results = load_evals(Path("outputs/eval_jsons"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    draw_main_bar_chart(results, FIG_DIR / "main_mpjpe_bar.png")
    # ... additional figure and table generators ...
    print("Figures and tables generated.")


if __name__ == "__main__":
    main()
```

## 8. Expected outcome

A single command regenerates all paper figures and tables from evaluation JSONs. The new advanced model’s uncertainty, learned triangulation, and Gauss-Newton refinement are explicitly visualized, and the paper draft’s tables stay synchronized with the latest results. This removes manual copy/paste and produces camera-ready, vector-format figures for the ICRA/CVPR 2027 submission.
