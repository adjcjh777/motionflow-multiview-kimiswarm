# v49: Per-View 2D Pose Extraction / Refinement

**Status:** Proposal / ready for design review  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #166 (proposed)  
**Depends on:** v46-SVG (#160), v47-temporal (#162), v48-domain (#164)

---

## 1. Problem statement

The current MotionFlow-MultiView pipeline treats **per-view 2D pose extraction as a fixed pre-processing step**: the canonical `.npz` already contains `points_2d (T, V, J, 2)` and `confidences (T, V, J)` from an upstream off-the-shelf detector. This creates three problems for the ICRA/CVPR 2027 story:

1. **2-D errors propagate directly into 3-D triangulation.** Occlusion, motion blur, domain shift, and joint mis-detection in any single view pollute the fused 3-D pose.
2. **The model has no control over its input.** v25 triangulation, v46 reliability, and v48 domain adaptation all consume the same noisy 2-D detections; none of them can correct systematic 2-D biases.
3. **No feedback from 3-D geometry back to 2-D extraction.** The self-evolution loop stops at the 3-D output (v37 SCVR, v39 RCAR, v43 adaptive residual). A true self-evolving system should also use 3-D consistency to refine its own 2-D inputs.

v49 therefore opens the first stage of the pipeline and adds a **lightweight, optional per-view 2-D pose refiner** that operates on the existing coordinate input and is supervised, in closed-loop, by multi-view reprojection.

---

## 2. Proposed approach

Add a `PerView2DRefinerV49` module inside `OmniMultiViewFusionV5` that refines `points_2d` and `confidences` **before** the v25 geometry-fusion block. The design is intentionally minimal:

- **Per-view only:** no cross-view attention, so the module is small and fast.
- **Temporal context:** a causal 1-D conv over `T` frames lets it smooth jitter and fill short occlusions.
- **Skeleton-aware joint MLP:** a tiny graph network over the kinematic tree corrects anatomically implausible local configurations.
- **Confidence recalibration:** outputs a refined confidence per `(view, joint)` that is consumed by v46/v45 weighted triangulation.
- **Identity at init:** output offsets and confidence rescaling start at zero/one, so the module is a no-op until trained.

### Fit with v46-v48 and the overall pipeline

```text
multi-view video / pre-extracted 2-D keypoints
        |
        v
[ PerView2DRefinerV49 ]
        |-- refines points_2d, confidences per view
        v
[ v25 Multi-View Geometry Fusion ]
        |
[ v46 Sparse-View Generalization ]  <-- uses refined confidences as reliability input
        |
[ v47 Temporal Aggregation ]         <-- sees temporally smoother 2-D trajectories
        |
[ v48 Domain Generalization ]        <-- can apply per-domain dropout/weights to refined 2-D
```

- **v46:** refined confidences replace or multiply the v46 reliability weights, giving sparse-view triangulation a stronger prior.
- **v47:** the temporal head receives 2-D trajectories that are already temporally smoothed, reducing the smoothing burden.
- **v48:** per-domain view dropout and DDWL can be applied to the refined 2-D detections; the refiner can also be domain-conditioned.

---

## 3. Concrete code-level changes

### New module

`motionflow_mv/extraction/per_view_2d_pose_extraction_v49.py` exposes:

```python
class PerView2DRefinerV49(nn.Module):
    def __init__(
        self,
        n_joints: int = 17,
        hidden: int = 64,
        n_layers: int = 2,
        temporal_kernel: int = 3,
        dropout: float = 0.1,
        recalibrate_confidence: bool = True,
    ):
        ...

    def forward(
        self,
        points_2d: torch.Tensor,      # (B, T, V, J, 2)
        confidences: torch.Tensor,    # (B, T, V, J)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return refined (points_2d, confidences)."""
```

### Integration points

| File | Change |
|------|--------|
| `motionflow_mv/extraction/per_view_2d_pose_extraction_v49.py` | New refiner module. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Insert refiner after input projection (around line 1085) when `use_v49_per_view_2d_refinement` is true. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add CLI flags; add closed-loop reprojection loss for the refiner. |
| `configs/benchmark_v49_per_view_2d_refinement_smoke.yaml` | Smoke config. |
| `scripts/run_v49_per_view_2d_refinement_smoke_local_4090.sh` | Smoke script. |
| `tests/test_per_view_2d_refinement_v49.py` | Unit tests for identity init, confidence range, and gradient flow. |

### New training flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v49_per_view_2d_refinement` | bool | `False` | Master switch. |
| `v49_2d_refiner_hidden` | int | `64` | Hidden dimension. |
| `v49_2d_refiner_layers` | int | `2` | Joint-MLP layers. |
| `v49_2d_refiner_temporal_kernel` | int | `3` | Causal conv kernel size. |
| `v49_2d_refiner_dropout` | float | `0.1` | Dropout. |
| `v49_2d_refiner_reproj_weight` | float | `0.01` | Weight of the closed-loop reprojection loss. |
| `v49_2d_refiner_recalib_conf` | bool | `True` | Recalibrate per-view confidences. |

---

## 4. Risks / failure modes

| Risk | Mitigation |
|------|------------|
| Refiner overfits to training 2-D noise and degrades val MPJPE. | Keep capacity small (`hidden=64`, `n_layers=2`); use strong weight decay; identity init. |
| Identity init never overcomes downstream 3-D loss gradient sparsity. | Add explicit auxiliary reprojection loss on the refined 2-D points. |
| Coordinate-only input cannot recover from severe occlusions. | Document as a limitation; leave image/heatmap input as a future extension. |
| Conflicts with v46 reliability head (both estimate confidences). | Multiply refined confidence with v46 reliability rather than replacing it. |
| Closed-loop reprojection loss is noisy early in training. | Warm it up after the first epoch; start with `v49_2d_refiner_reproj_weight=0`. |

---

## 5. Success metrics and recommended experiment

### Smoke experiment

| Field | Setting |
|-------|---------|
| Hardware | RTX 4090 |
| Config | `configs/benchmark_v49_per_view_2d_refinement_smoke.yaml` |
| Model | v48 base + `use_v49_per_view_2d_refinement=true` |
| Samples | 500, 2 epochs |
| Target | `val_MPJPE` finite and within 5% of the v48 smoke baseline; no NaN/OOM |

### Full experiment

| Field | Setting |
|-------|---------|
| Hardware | A800-D |
| Base | Best v48 checkpoint |
| Config | `configs/benchmark_v49_per_view_2d_refinement_full.yaml` |
| Target | ≥2% lower `val_MPJPE@full` than v48; ≥5% lower `MPJPE@2/3` because better 2-D directly helps few-view triangulation |

### Evaluation metrics

- `val_MPJPE` and `MPJPE@k` for `k ∈ {2, 3, 4, full}` (via `experiments/eval_variable_views.py`).
- Mean 2-D reprojection error on validation.
- Refined confidence calibration (ECE) per view.

---

## 6. Self-evolution feedback loop

v49 closes the outer loop of the v36–v48 self-evolution stack:

```
points_2d  --[refiner]-->  points_2d'
                              |
                              v
                         v25 triangulation
                              |
                              v
                         3-D pose P_t
                              |
                              v
                    project P_t back to each view
                              |
                              v
                    reprojection residual r_t
                              |
                              v
                    supervise points_2d' and confidences
```

- **Action:** the refiner proposes corrected 2-D detections.
- **Feedback:** the 3-D triangulation and reprojection residual act as the geometric verifier.
- **Retry:** high-residual `(view, joint)` pairs receive lower refined confidence, and the refiner is trained to reduce the residual on the next forward pass.

This is the same `action → feedback → retry` pattern mapped in `docs/qwen38_selfevolution_mapping.md`, now applied at the very first stage of the multi-view pipeline.

---

## 7. Next steps

1. Wait for v48-domain smoke results (#164) to stabilize the base.
2. Implement `PerView2DRefinerV49` and the closed-loop reprojection loss.
3. Wire the flags into `OmniMultiViewFusionV5` and the trainer.
4. Run smoke on RTX 4090 and compare `val_MPJPE` and reprojection error against v48.
5. If smoke shows no regression, queue a full A800 run warm-started from the best v48 checkpoint.
