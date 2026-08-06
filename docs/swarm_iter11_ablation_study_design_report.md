# Iter11+ Ablation Study Design for MotionFlow-MultiView

## 1. Current State

The codebase now contains a full-stack multi-view fusion model,
`RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1`
(`motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`).
It fuses five previously separate ideas: ray-aware per-view attention,
spatio-temporal (time, view) attention, uncertainty-weighted DLT, differentiable
Gauss-Newton triangulation, and residual MLP refinement.

The current best MPI-INF-3DHP validation MPJPE is **~11.17 mm**, produced by the
`RayAttentionFusionModelTemporalCrossviewResidual` model. The new combined model
has only been lightly trained; a fast temporal-residual + reprojection run
reached **47.54 mm** because of limited data/epochs. H36M WebBridge conversion
is ongoing at `data/webbridge/h36m`.

Existing ablation tooling is fragmented:

- `experiments/run_ablations.py` targets the outdated `RayAttentionFusionModelV3`
  and a synthetic/H36M stub.
- `experiments/ablate_residual_hidden_mpiinf3dhp.py` only varies
  `residual_hidden` for `RayAttentionFusionModelTemporalResidual`.
- `experiments/ablation_v1_v2_h36m.py` compares view-only vs. view+joint
  attention on a small H36M subset.
- There is **no unified ablation harness** for the combined model and **no
  component-wise isolation** of cross-view attention, uncertainty weighting,
  Gauss-Newton triangulation, or residual refinement on the same split.

## 2. Concrete, Implementable Improvements

### 2.1 Unified component ablation harness

Create `experiments/run_ablations_iter11.py` that trains every variant on the
same MPI-INF-3DHP split (train S1 Seq1+2, val S2 Seq1) with identical
hyperparameters and seeds. Build each variant from a shared base class with
Boolean toggles so every component's contribution is isolated.

Proposed variants:

| Variant | What is removed / changed | Scientific question |
|---|---|---|
| A | Raw DLT baseline | Geometric lower bound |
| B | Cross-view temporal attention, no residual | Is the residual head needed? |
| C | Full combined model | Best attainable with all components |
| D | B with `clip_len = 1` | Value of temporal attention |
| E | B with linear residual (1-layer) | Is the non-linear MLP useful? |
| F | B without cross-view attention | Value of the (time, view) grid transformer |
| G | B with sigmoid weights instead of uncertainty | Value of learned log-variance |
| H | B without learned Gauss-Newton triangulation | Does GN refinement help? |
| I | B without camera embedding | Does camera conditioning help or hurt? |
| J | B without joint-level attention | Value of cross-joint features |

### 2.2 Hyperparameter ablations

Systematically vary: `d ∈ {32, 64, 128}`, `n_st_layers ∈ {1, 2, 3}`,
`residual_hidden ∈ {32, 64, 128, 256}`, `gn_iters ∈ {0, 1, 3, 5}` and
`gn_damping ∈ {1e-7, 1e-6, 1e-5}`, `uncertainty_loss_weight ∈ {0.0, 0.05,
0.1, 0.2}`, `reproj_weight ∈ {0.0, 0.01, 0.05, 0.1}`, and `clip_len ∈ {1, 7,
13, 21, 31}`. Run a fast smoke grid (5 epochs, 1k clips) to narrow the range,
then a full grid (30 epochs, 4k clips) around promising regions.

### 2.3 Robustness and data ablations

Robustness matrix on the validation split: Gaussian 2D noise
`σ ∈ {0, 1, 2, 5, 10, 20}` px, view dropout `p ∈ {0, 0.1, 0.2, 0.3, 0.4}`,
sparse 2D outliers `o ∈ {0, 0.05, 0.1, 0.2}` (100 px scale), and per-joint
occlusion `∈ {0, 0.2, 0.5}`. Also vary training-scale (1k–16k clips),
cross-subject transfer, cross-dataset transfer (MPI-INF-3DHP ↔ H36M WebBridge),
and camera-embedding on/off.

## 3. Metrics to Track

Use `motionflow_mv/eval/metrics.py::compute_all_metrics` and record MPJPE,
PA-MPJPE, PCK@50/100/150 mm, AUC, per-joint/per-view MPJPE, mean
reprojection error (px), parameter count, and runtime on the RTX 4090. Each
run should write a JSON row with seed, checkpoint path, and wall time.

## 4. Implementation Snippet

```python
# experiments/run_ablations_iter11.py
from motionflow_mv.fusion.ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model import (
    RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1 as FullModel,
)

COMPONENTS = {
    "full":            {"use_uncertainty": True, "use_gn": True,
                       "use_residual": True, "use_crossview": True,
                       "use_camera_emb": True},
    "no_residual":     {"use_residual": False},
    "no_gn":           {"use_gn": False},
    "sigmoid_weights": {"use_uncertainty": False},
    "no_crossview":    {"use_crossview": False},
    "no_camera_emb":   {"use_camera_emb": False},
    "linear_residual": {"residual_hidden": 0},
    "single_frame":    {"clip_len": 1},
}

def build_model(name: str, j: int, n_views: int, base_kwargs: dict):
    flags = COMPONENTS[name]
    # Requires refactoring the combined model so each flag is a constructor
    # argument; currently each variant needs a small subclass.
    ...
    return model
```

The next engineering step is to refactor the combined model so these toggles are
constructor arguments rather than hard-coded subclasses, because the current
model inherits from `RayAttentionFusionModelTemporalCrossview` and wires every
component inside `forward()`.

## 5. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Combined model is larger/slower; full grid is expensive | High | Run smoke grid first; prioritize the top 4–5 variants on the RTX 4090 |
| Gauss-Newton / uncertainty destabilize training | Medium | Clamp `log_var`, clip gradients, start with `uncertainty_loss_weight=0` |
| Camera embeddings hurt generalization as in v3/v4 | Medium | Explicit camera on/off ablation; freeze embeddings early |
| H36M WebBridge incomplete | Medium | Fall back to MPI-INF-3DHP-only ablations, queue H36M runs |
| Component interactions mislead single-factor ablations | Medium | Add a leave-one-out and add-one-in ladder starting from DLT |

## 6. Suggested Deliverables

1. `experiments/run_ablations_iter11.py` — unified ablation harness.
2. `docs/swarm_iter11/ablation_component_table.md` — component results.
3. `docs/swarm_iter11/ablation_hyperparam_table.md` — hyperparameter sweep.
4. `docs/swarm_iter11/ablation_robustness_table.md` — robustness matrix.
5. Refactored combined model with Boolean component flags.

## 7. Summary

The immediate next step is a single, reproducible ablation harness for the
combined model on MPI-INF-3DHP. The highest-value ablations are (1) residual
head on/off, (2) uncertainty vs. sigmoid weights, (3) learned Gauss-Newton vs.
DLT, (4) cross-view attention vs. temporal-only, and (5) camera embedding on/off.
Track the full `compute_all_metrics` suite plus parameter count and runtime.
This will turn the current collection of model-specific scripts into a
paper-ready ablation study and reveal which of the five fused components
actually improve MPJPE.
