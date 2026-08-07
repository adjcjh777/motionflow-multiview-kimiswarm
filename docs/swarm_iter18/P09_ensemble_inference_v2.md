# P09 Ensemble Inference v2

**Branch:** `feat/swarm-iter18-omniview`  
**Author:** Kimi Code subagent  
**Date:** 2026-08-07  
**Status:** Prototype implemented + CPU smoke test passed

## 1. Goal

Improve the existing multi-checkpoint ensemble inference from a simple (weighted) average to an **uncertainty-aware, robust ensemble** that can exploit the per-view anisotropic covariance predicted by `RayAttentionFusionModelBayesianTriV2` and its variants.

## 2. What changed

New file: `experiments/prototypes/swarm_iter18/ensemble_inference_v2.py`

It introduces `BayesianTriV2Ensemble`, a drop-in replacement for the existing `MultiCheckpointEnsemble` used by `experiments/prototypes/eval_ensemble_checkpoints.py`, with the following v2 features:

| Feature | v1 (`MultiCheckpointEnsemble`) | v2 (`BayesianTriV2Ensemble`) |
|---|---|---|
| Aggregation | Uniform or global weighted mean | `uniform`, `inverse_variance`, `robust_median`, `trimmed_mean` |
| Per-joint weighting | No | Yes (inverse_variance) |
| Uses predicted covariance | No | Yes |
| Epistemic uncertainty output | No | Yes (variance across members) |
| Robust to outlier checkpoints | No | `robust_median`, `trimmed_mean` |

### 2.1 Aggregation strategies

- **`uniform`**: classic mean across checkpoints.
- **`inverse_variance`**: per-joint weights derived from the predicted image-space Cholesky covariance `L`.  Precision is computed as `1 / det(Σ)` and normalised across the ensemble.
- **`robust_median`**: median prediction across the ensemble.
- **`trimmed_mean`**: sort predictions along the ensemble dimension, drop the lowest/highest `trim_alpha` fraction, then average the rest.

### 2.2 Outputs

The script reports the usual MPI-INF-3DHP metrics (MPJPE, PA-MPJPE, PCK, AUC) plus:

- **Mean epistemic std** across joints/frames.
- Optional **JSON** metrics file.
- Optional **NPZ** file containing `pred_3d`, `gt_3d`, and `epistemic_var`.

## 3. Usage

### Full evaluation

```bash
python experiments/prototypes/swarm_iter18/ensemble_inference_v2.py \
    --model bayesian_tri_v2_pp \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/bayesian_tri_v2_large_scale_mpiinf3dhp_seed0.pth \
    --checkpoint outputs/bayesian_tri_v2_large_scale_mpiinf3dhp_seed1.pth \
    --checkpoint outputs/bayesian_tri_v2_large_scale_mpiinf3dhp_seed2.pth \
    --clip_len 13 --d 128 --residual_hidden 256 --n_st_layers 3 \
    --val_stride 50 \
    --strategy inverse_variance \
    --output_json outputs/ensemble_v2_mpiinf3dhp.json
```

### CPU smoke test

```bash
python experiments/prototypes/swarm_iter18/ensemble_inference_v2.py --smoke_test
```

The smoke test builds two tiny `RayAttentionFusionModelBayesianTriV2` instances, saves them to temporary checkpoints, and runs the ensemble through all four strategies on CPU.

## 4. Implementation notes

- The module is **isolated** under `experiments/prototypes/swarm_iter18/`.  No existing shared modules were modified.
- It reuses `experiments/eval_full_metrics.py` for `TemporalClipDataset`, `build_model`, and `collate_fn`.
- Member models are forced into covariance-returning mode by setting `model.return_covariance = True` after construction when the strategy is `inverse_variance`.
- `BayesianTriV2Ensemble` subclasses `nn.Module` and stores members in a `ModuleList`, so it can be moved to/from devices with `.to(device)`.

## 5. Next steps

1. Run the script on the existing `bayesian_tri_v2` checkpoint ensemble on MPI-INF-3DHP S2/Seq1 and compare MPJPE against the 8.35 mm anchor.
2. If `inverse_variance` improves over `uniform`, consider training a small meta-learner that predicts per-member weights from validation performance.
3. Extend to multi-scale / multi-seed ensembles across different architectures (e.g. `bayesian_tri_v2_pp` + `bayesian_tri_v2_attention_entropy_pp`).
