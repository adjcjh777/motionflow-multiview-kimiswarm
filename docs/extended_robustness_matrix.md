# Extended Robustness Evaluation Matrix

This prototype extends the existing `experiments/run_robustness_matrix.py` to cover three input-degradation axes:

* **Gaussian 2-D keypoint noise**
* **Per-joint occlusion (joint dropout)**
* **Whole-view dropout**

and their pairwise / three-way combinations.

## Files

* `experiments/prototypes/run_extended_robustness_matrix.py` — evaluation script.
* `scripts/run_extended_robustness_matrix_wsl.sh` — convenience runner for WSL.
* `tests/test_extended_robustness_matrix.py` — CPU-only pytest smoke test.

## Usage

### CPU smoke test (no checkpoint required)

```bash
python experiments/prototypes/run_extended_robustness_matrix.py
```

### Real evaluation

```bash
scripts/run_extended_robustness_matrix_wsl.sh \
    bayesian_tri_v2_pp \
    outputs/bayesian_tri_v2_large_scale_mpiinf3dhp.pth \
    data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz
```

This writes three files to `outputs/extended_robustness_matrix_<model>/`:

* `robustness_matrix.json`
* `robustness_matrix.md`
* `robustness_matrix.csv`

## Condition matrix

| Category | Conditions |
|---|---|
| Baseline | `clean` |
| Noise | `noise_0.5px`, `noise_1.0px`, `noise_2.0px` |
| Joint occlusion | `joint_occlusion_10`, `joint_occlusion_20`, `joint_occlusion_30` |
| View dropout | `view_dropout_10`, `view_dropout_30`, `view_dropout_50` |
| Two-axis combos | `noise_1.0px_joint_occlusion_20`, `noise_1.0px_view_dropout_30`, `joint_occlusion_20_view_dropout_30` |
| Three-axis combo | `noise_1.0px_joint_occlusion_20_view_dropout_30` |

## Implementation notes

* Perturbations are applied in the order: noise → joint occlusion → view dropout.
* Each condition uses a deterministic seed derived from the global seed and the condition name, so repeated runs are reproducible while different conditions do not share random masks.
* The script reuses `experiments/eval_full_metrics.py` for model registry, dataset loading, and builder logic, so it supports every model class that `run_robustness_matrix.py` supports.
