# Design: Variable View-Count Inference and Benchmark

## 1. Motivation

The current best model, `RayAttentionFusionModelTemporalResidual`, is trained
and evaluated with a fixed number of cameras (`n_views`).  In practice,
deployment rigs may have different view counts, or cameras may drop out due to
occlusion, synchronization failures, or calibration loss.  We need a way to run
inference with **any number of active views up to the trained maximum**, without
retraining the model each time a view is added or removed.

## 2. Design decisions

### 2.1 Zero-confidence masking (inference-time adaptation)

Rather than changing the architecture, we keep the fixed-view model and **mask
out dropped views by setting their confidence to zero** before the forward
pass.

Why this works for our models:

- The weight head predicts per-view logits for every slot.
- The predicted weights are multiplied by the input confidence before DLT
  triangulation:
  ```
  weights = sigmoid(logits) * confidences
  ```
- Therefore a zero-confidence view contributes zero weight to the triangulation,
  effectively removing it from the geometric solve.

This is a pragmatic, training-free solution.  The attention layers still see
the dropped views, but because their observations are zeroed the model tends to
ignore them.  The approach is compatible with every `RayAttentionFusionModel*`
variant that follows the same `weights * confidences` convention.

### 2.2 Wrapper API

`VariableViewInferenceWrapper` exposes a `__call__` that accepts an
`active_views` argument:

```python
wrapper = VariableViewInferenceWrapper(model)
pred, weights = wrapper(x, K, R, t, active_views=[0, 2])
```

`active_views` may be:
- an integer `k` (use the first `k` views),
- a list of view indices, or
- a boolean mask.

`prepare_variable_view_input` pads the input to the model's expected
`n_views_max` and applies the mask.  If the input already has `n_views_max`
views, only the mask is applied.

### 2.3 Benchmark protocol

`experiments/eval_variable_views.py` benchmarks a trained (or randomly
initialized) model across view counts `k = min_views .. n_views_max`:

1. For each `k`, enumerate all `C(V, k)` subsets.
2. For each subset, zero-mask the complement.
3. Run the model on short temporal clips.
4. Report mean and standard deviation of MPJPE across all subsets for that
   `k`.

The script supports both real `.npz` datasets (with optional checkpoint) and a
built-in synthetic smoke mode, so it can run without external data.

## 3. Files added

- `motionflow_mv/fusion/variable_view_inference.py`
  - `prepare_variable_view_input`
  - `apply_view_mask`
  - `generate_view_subsets`
  - `VariableViewInferenceWrapper`

- `experiments/eval_variable_views.py`
  - Synthetic data generation and projection.
  - Variable view-count MPJPE benchmark.
  - Optional checkpoint/dataset loading for real evaluation.

- `docs/swarm_iter_next/design_variable_view_inference/report.md`
  - This document.

## 4. Expected impact

- **Deployment flexibility:** A single checkpoint trained with, e.g., 14 views
can be deployed on rigs with 2-14 views without retraining or architecture
modification.
- **Robustness analysis:** We can now measure how accuracy degrades as views are
lost, informing operational requirements (minimum view count, camera placement).
- **Preparation for true variable-view models:** The benchmark establishes the
protocol and metrics; a future model with geometry-based camera positional
encoding (instead of learned view embeddings) can drop the fixed-slot limitation
entirely.

## 5. Limitations and next steps

- Attention layers still process masked views, which may add a small amount of
  noise.  A future architecture should use a learned or geometry-based camera
  token that is independent of view order and count.
- The current wrapper does not handle **more views than `n_views_max`**;
  handling that requires view selection or a fully variable architecture.
- No retraining on variable view subsets has been done; fine-tuning with random
  view dropout may improve masked inference accuracy.
