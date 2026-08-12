# Adaptive Temporal Window Pyramid for Multi-View 3D Pose

**Date:** 2026-08-06
**Author:** MotionFlow-MultiView research engineering
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP S2/Seq1 clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE
**Status:** Proposal + runnable skeleton implemented, CPU sanity check passed

---

## 1. Hypothesis

Replacing the single fixed-length spatio-temporal attention block with a **learned adaptive temporal-window pyramid** — short windows for fast motion, medium windows for smooth articulation, and a global window for long-range context — improves robustness to motion speed, occlusions, and calibration drift while keeping clean accuracy within 0.4 mm of the 9.32 mm anchor.

---

## 2. Related Existing Files/Modules

- **Anchor model:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`
  - `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` (clean MPJPE 9.32 mm)
- **Parent architecture:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py`
- **Per-frame / ray embedding:** `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py`
- **Principal-point correction:** `motionflow_mv/fusion/principal_point_correction.py`
- **Reprojection losses:** `motionflow_mv/losses/reprojection_consistency.py`
- **Fusion registry / wrapper:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_module.py`
- **Existing multi-scale temporal conv baseline:** `motionflow_mv/fusion/multiscale_temporal_conv_model.py` (conv-only, no cross-view attention)
- **Training example:** `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
- **Evaluation example:** `experiments/eval_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`

---

## 3. Proposed Code Changes

### 3.1 New model (primary skeleton)

**Create:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_adaptive_window_pyramid_model.py`

- New class: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointAdaptiveWindowPyramid`
  - Inherits from `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`
  - Constructor signature adds:
    - `temporal_scales: Tuple[int, ...] = (3, 7, 0)` — window sizes; `0` means full-clip global attention
    - `pyramid_layers: int = 1` — number of adaptive pyramid layers
    - `pyramid_dropout: float = 0.1`
    - `return_scale_weights: bool = False`
  - Keeps all anchor components intact: principal-point correction, residual refinement, weight head, ray embedding.
  - Replaces the single `st_transformer` path with:
    - `AdaptiveWindowPyramidLayer(nn.Module)` that, for each temporal scale, gathers local windows around every frame, runs a compact self-attention over `(window, views)` tokens, and projects the result back to a `(T, V, d)` feature map.
    - A learnable **scale-mixing gate** (per time, view, joint, and scale) that produces convex combinations of the multi-scale outputs.
  - All operations are causal-mask-free (offline clips) and batched with `unfold` / `pad` so the change is purely local to the temporal fusion stage.

### 3.2 New fusion-module wrapper

**Create:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_adaptive_window_pyramid_module.py`

- Class: `RayAttentionTemporalCrossviewResidualPrincipalPointAdaptiveWindowPyramidFusionModule`
  - Mirrors the existing `...PrincipalPointFusionModule`
  - Registers name: `ray_attention_temporal_crossview_residual_principal_point_adaptive_window_pyramid`

### 3.3 Registry update

**Modify:** `motionflow_mv/fusion/__init__.py`
- Import and register the new fusion-module wrapper.

### 3.4 Training script (smoke-only stub)

**Create:** `experiments/train_adaptive_window_pyramid_mpiinf3dhp.py`
- Copy of the anchor training script with model class swapped and `clip_len` increased to at least `max(temporal_scales)`.

### 3.5 Evaluation script (smoke-only stub)

**Create:** `experiments/eval_adaptive_window_pyramid_mpiinf3dhp.py`
- Copy of the anchor evaluation script with model class swapped.

---

## 4. Training / Smoke Plan (≤5 epochs, RTX 4090)

Use the existing MPI-INF-3DHP subset (`data/webbridge/mpi_inf_3dhp/`) with a small smoke split:

```bash
python experiments/train_adaptive_window_pyramid_mpiinf3dhp.py \\
  --config configs/benchmark_webbridge_mpi_smoke.yaml \\
  --epochs 5 \\
  --batch_size 8 \\
  --clip_len 27
```

**Smoke pass criteria:**

1. No `RuntimeError`, NaN, or gradient explosion.
2. Training completes in ≤ 45 min on RTX 4090 (single run).
3. Validation clean MPJPE ≤ 10.0 mm (allowing 0.7 mm smoke regression from anchor 9.32 mm).
4. The adaptive scale-mixing weights exhibit variance across clips (i.e., the gate is not collapsed to a single scale).

**Smoke fail criteria:**

- Val MPJPE > 11.0 mm or any instability.
- Runtime > 90 min (indicates windowing implementation is too expensive).
- Scale weights are uniform / collapsed (gate does not adapt).

---

## 5. Success Metrics

| Metric | Target | How measured |
|---|---|---|
| Clean MPJPE (MPI-INF-3DHP S2/Seq1) | ≤ 9.32 mm (anchor) or ≤ 9.60 mm if minor regression | `experiments/eval_adaptive_window_pyramid_mpiinf3dhp.py` |
| PA-MPJPE | ≤ 5.60 mm | same |
| Robustness to view dropout | ≥ 5% relative improvement over anchor under -view corruption | reuse `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` |
| Robustness to joint occlusion | ≥ 5% relative improvement over anchor under joint-dropout | same |
| Calibration drift (±5 px PP noise) | ≥ 5% relative improvement | same robustness matrix with PP perturbation |
| Runtime per clip (T=27) | ≤ 1.5× anchor latency | `experiments/benchmark_runtime.py` |

---

## 6. Risk and Fallback

| Risk | Mitigation / Fallback |
|---|---|
| **Window alignment complexity** introduces boundary artifacts or memory blow-up. | Keep windows symmetric with padding; if memory > 12 GB, reduce `d` to 32 or number of scales to two. |
| **Adaptive gate collapses** to a single scale and adds no value. | Remove the gate and use fixed equal-weight fusion as an ablation; if still no gain, drop the pyramid and report negative result. |
| **Longer clips (T=27)** hurt convergence on small smoke data. | Start from the anchor checkpoint, freeze the per-frame encoder for the first 2 epochs, and warm-up only the pyramid layers. |
| **Overfitting** because the pyramid head adds parameters. | Add 0.1 weight decay and dropout; reduce `pyramid_layers` to 1. |
| **No cross-dataset gain** on H36M or WebBridge. | Treat as an MPI-INF-3DHP-specific ablation; do not merge into the general anchor. |

---

## 7. ICRA/CVPR 2027 Angle

The contribution is a **calibrated multi-view video fusion** component:
- **Multi-view video → 3D skeleton:** fuses multi-scale temporal evidence from all calibrated views rather than collapsing the temporal axis once.
- **Calibration / alignment to physical space:** the existing principal-point and focal correction are preserved; the pyramid operates on ray-space features after intrinsic correction, so physical camera geometry remains explicit.
- **Robustness across views:** short windows reduce reliance on any single occluded frame, while the global window enforces temporal consistency; the scale gate adapts per-joint and per-view to the reliable temporal context.
