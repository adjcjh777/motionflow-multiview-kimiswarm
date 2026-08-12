# v49: Online Self-Evolution Loop

**Status:** Proposal / design ready  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #167 (proposed)  
**Depends on:** #160 (v46-SVG), #162 (v47-temporal), #164 (v48-domain), #166 (v49 self-evolution with uncertainty/reliability/reprojection feedback)

---

## 1. Problem statement

By v48 the pipeline handles sparse views, temporal noise, and domain shift, but it still **triangulates once and stops**.  The per-view reliability from v45/v46 and the uncertainty gate from v36/v43 are learned offline; they never react to the geometric self-consistency of the *current* prediction.  This leaves two gaps:

1. **Residual information is discarded.**  After triangulation we know, for every view and joint, how well the predicted 3D point reprojects onto the original 2D detections.  Large residuals flag bad views or occlusions, yet the model does not feed them back to correct the next estimate.
2. **The v27 test-time self-evolution (TTE) loop is broken.**  Running the frozen triangulation head iteratively at inference produced ~90 mm failures and is hard to stabilize.  Any new self-evolution mechanism must be **gradient-safe during training**, **identity-at-init**, and **cheap at inference**.

v49 therefore proposes an **online self-evolution loop**: a differentiable, one- or two-step refinement that uses reprojection residuals to update per-view reliability/uncertainty and then re-triangulates, all inside the training graph.

---

## 2. Proposed approach and how it fits with v46-v48 and the overall pipeline

### Core idea

Add a lightweight `OnlineSelfEvolutionV49` head that takes the initial triangulated pose and the existing v46/v37 reliability/uncertainty estimates, and performs a short iterative refinement:

```text
P^0  = initial triangulated pose from v25/v45/v46/v48
for k = 1 .. K (K ≤ 2 by default):
    e^k_reproj = reprojection_residual(P^{k-1}, points_2d, cameras)
    r^k, u^k   = feedback_mlp(r^{k-1}, u^{k-1}, e^k_reproj)
    P^k        = weighted_triangulation(points_2d, cameras, weights=r^k)
P' = P^K
```

The loop is intentionally **short and differentiable**.  The output projection and the residual gate inside `OnlineSelfEvolutionV49` are zero-initialized, so the module is a strict no-op at the start of training.  The model therefore has to *learn* to improve; it cannot accidentally diverge from a warm-started v48 checkpoint.

### Where it lives in the pipeline

```text
2D keypoints + cameras
        |
        v
[v25 Multi-View Geometry Fusion]
        |
        v
[v46 Sparse-View Generalization reliability weights]
        |
        v
[v37 / v43 reliability & uncertainty maps]
        |
        v
[v48 Domain-invariant sparse-view refinement]  ->  P^0
        |
        v
[OnlineSelfEvolutionV49]
        |
        ├── Reprojection residual feedback
        ├── Updated per-view reliability r'
        └── Re-weighted triangulation  ->  P'
        |
        v
[v47 Temporal Aggregation (optional, on P')]
```

### Fit with v46-v48

- **v46 Sparse-View Generalization:** v49 consumes the v46 reliability head and lets it evolve from current reprojection residuals.  The v46 view-dropout augmentation is unchanged.
- **v47 Temporal Aggregation:** v47 can run either before or after the online loop; the default is after, so the temporal head smooths the already self-corrected trajectory.
- **v48 Domain Generalization:** v48 provides domain-invariant features and per-domain dropout; v49 is domain-agnostic and reuses the same pose/camera interface.
- **Overall multi-view pipeline:** v48 remains the backbone.  v49 is a small, optional post-triangulation self-correction that closes the feedback loop between 2D evidence, 3D prediction, and per-view uncertainty.

---

## 3. Concrete code-level changes

### New module

`motionflow_mv/fusion/online_self_evolution_v49.py`

```python
class OnlineSelfEvolutionV49(nn.Module):
    def __init__(
        self,
        n_joints: int = 17,
        n_views: int = 8,
        n_iters: int = 2,
        hidden: int = 64,
        sigma_reproj: float = 5.0,
        residual_thresh_mm: float = 0.5,
        residual_gate_init: float = 0.0,
        use_v37_reliability: bool = True,
    ):
        ...

    def forward(
        self,
        pose_3d: torch.Tensor,              # (B, T, J, 3)
        points_2d: torch.Tensor,            # (B, T, V, J, 2)
        K: torch.Tensor,                    # (B, T, V, 3, 3)
        R: torch.Tensor,                    # (B, T, V, 3, 3)
        t: torch.Tensor,                    # (B, T, V, 3)
        reliability: torch.Tensor | None,   # (B, T, V, J)
        uncertainty: torch.Tensor | None,   # (B, T, V, J)
        view_mask: torch.Tensor | None,     # (B, T, V)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            refined_pose: (B, T, J, 3)
            updated_reliability: (B, T, V, J)
        """
```

### Files to touch

| File | Change |
|------|--------|
| `motionflow_mv/fusion/online_self_evolution_v49.py` | New `OnlineSelfEvolutionV49` module. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Add v49 flags; instantiate module; call it after v48/v47 output and before the residual refinement head. |
| `motionflow_mv/fusion/test_time_self_evolution_v27.py` | Re-use `compute_reprojection_residual` and `triangulate_dlt_per_joint` helpers. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add CLI flags; wire model kwargs; add optional self-evolution reprojection consistency loss. |
| `experiments/eval_variable_views.py` | Report `MPJPE@k` with/without online self-evolution and mean reprojection residual. |
| `tests/test_online_self_evolution_v49.py` | Unit tests for shape, mask handling, identity at init, and iterative refinement. |
| `configs/benchmark_v49_ose_smoke.yaml` | Smoke config. |
| `scripts/run_v49_ose_smoke_local_4090.sh` | Smoke script. |

### New training / model flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_online_self_evolution_v49` | bool | `False` | Master switch. |
| `v49_ose_n_iters` | int | `2` | Number of self-evolution steps. |
| `v49_ose_sigma_reproj` | float | `5.0` | Cauchy kernel scale (pixels) for residual re-weighting. |
| `v49_ose_residual_thresh_mm` | float | `0.5` | Early-stop threshold (mm) per step. |
| `v49_ose_hidden` | int | `64` | Hidden dim of the reliability/uncertainty update MLP. |
| `v49_ose_use_v37_reliability` | bool | `True` | Seed the loop with v37 reliability when available. |
| `v49_ose_reproj_loss_weight` | float | `0.01` | Weight of the auxiliary reprojection-consistency loss. |
| `v49_ose_min_views_for_loop` | int | `2` | Skip the loop when fewer views are available. |

### Integration sketch in `omniview_fusion_v5.py`

Insert after the v48/v47 output and before the residual refinement / v22 KAP heads:

```python
if (
    self.use_online_self_evolution_v49
    and self.online_self_evolution_v49 is not None
):
    pred_3d, updated_rel = self.online_self_evolution_v49(
        pose_3d=pred_3d,
        points_2d=points_2d.view(B, T, V, J, 2),
        K=K_corrected.view(B, T, V, 3, 3),
        R=R.view(B, T, V, 3, 3),
        t=t.view(B, T, V, 3),
        reliability=scvr_reliability,          # from v37 if available
        uncertainty=v36_uncertainty,           # from v36 if available
        view_mask=view_mask_flat.view(B, T, V),
    )
    # Optionally cache updated_rel for logging.
```

### Auxiliary loss

A small self-supervised loss encourages the loop to actually reduce reprojection error:

```python
if self.use_online_self_evolution_v49 and v49_reproj_loss_weight > 0:
    initial_residual = compute_reprojection_residual(pred_3d_init, ...)
    final_residual   = compute_reprojection_residual(pred_3d, ...)
    loss = loss + v49_reproj_loss_weight * relu(final_residual - initial_residual).mean()
```

---

## 4. Risks / failure modes

| Risk | How it manifests | Mitigation |
|------|------------------|------------|
| **Loop divergence / oscillation** | Refined pose is worse than initial; training NaN. | Keep `n_iters` ≤ 2; early stop on `residual_thresh_mm`; clamp the residual gate to `[0, 1]`. |
| **Gradient instability through iterative triangulation** | `torch.linalg.lstsq` inside the loop breaks the graph or explodes. | Use the analytic weighted-DLT helper already in `test_time_self_evolution_v27.py`; add `epsilon` to weight sums. |
| **No measurable gain** | v48 baseline is already strong; loop learns the identity map. | Force a small auxiliary reprojection-consistency loss and monitor residual reduction. |
| **Re-introducing broken v27 TTE behavior** | Running the loop at inference with `K>1` causes instability. | Default inference path uses the same trained `n_iters=2` but under `torch.no_grad`; keep an fallback flag to disable at inference. |
| **Slowdown from iterative triangulation** | Training time grows with `n_iters`. | Cache projection matrices; default `n_iters=2`; benchmark on RTX 4090 smoke before A800. |

---

## 5. Success metrics and recommended smoke / full experiment

### Metrics

- `val_MPJPE@k` for `k ∈ {2, 3, 4, full}` (reuse `experiments/eval_variable_views.py`).
- `mean_reprojection_residual@k` before and after the online loop.
- `reliability_ranking_auc`: how well updated reliability ranks views by actual residual.

### Smoke experiment

| Stage | Hardware | Config | Goal |
|-------|----------|--------|------|
| Smoke | RTX 4090 | `configs/benchmark_v49_ose_smoke.yaml` | `val_MPJPE` finite; no NaN/OOM; mean reprojection residual decreases after the loop; `val_MPJPE@2` within 1 mm of v48 smoke baseline. |

```bash
bash scripts/run_v49_ose_smoke_local_4090.sh
```

Typical smoke config: `d=64`, `train_samples=500`, `clip_len=9`, warm-start from the best v48 smoke checkpoint.

### Full experiment

| Stage | Hardware | Config | Goal |
|-------|----------|--------|------|
| Full | A800-D | `v49_ose_all_train` manifest, warm-start from best v48 checkpoint | Improve `MPJPE@2`/`MPJPE@3` by ≥3 % over v48; no regression at full views. |

**Recipe**

1. Warm-start from the best v48-domain checkpoint.
2. Freeze v25/v45/v46/v47/v48 weights for 1 epoch; train only the v49 online-self-evolution head.
3. Unfreeze and fine-tune end-to-end with the v48 mixed manifest.
4. Validate on H36M/MPI/AIST val, 3DPW pseudo val, and 3DPW actual val.

### Success criteria

1. Smoke test passes with no NaN/OOM and finite per-domain `val_MPJPE`.
2. Mean reprojection residual after the loop is lower than before the loop on the val set.
3. `val_MPJPE@full` is within 1 mm of the v48 baseline (no regression).
4. `val_MPJPE@2`/`val_MPJPE@3` improve by ≥3 % over v48.
5. A800 full run completes ≥1 epoch.

---

## 6. The self-evolution feedback loop

The central idea of v49 is a **closed, differentiable feedback loop** between prediction, observation, and uncertainty:

```text
predict P^0  ->  measure reprojection residual e
      ^                            |
      |                            v
      |               update reliability r' = f(r, e)
      |                            |
      |                            v
      +------------------  re-weighted triangulation -> P'
```

- **Prediction:** v48 produces the initial 3D pose `P^0` and the v45/v46/v37 reliability/uncertainty maps.
- **Observation:** v49 computes per-view, per-joint reprojection residuals for `P^0`.
- **Self-critique:** a small MLP maps `(residuals, old reliability, old uncertainty)` to updated reliability.
- **Refinement:** the updated reliability re-weights a second-pass triangulation, producing `P'`.
- **Learning:** the final pose loss and the auxiliary reprojection-consistency loss train the network to produce *better* uncertainty estimates and more accurate poses on the next sample.

This is the multi-view analogue of Qwen-style self-improvement: the model learns to critique its own multi-view predictions and down-weight inconsistent evidence.  Unlike the broken v27 TTE loop, v49 stays **inside the training graph**, starts from an **identity mapping**, and uses only a **small number of iterations**, making it stable enough for both training and inference.

### Relation to existing variants

- **v27 TTE:** v27 iterated a frozen triangulation head at inference and broke; v49 learns a short differentiable loop and is safe at inference.
- **v37 SCVR:** v37 learned static per-view reliability; v49 *updates* that reliability online from geometric residuals.
- **v39 RCAGR / v43 adaptive residual:** those works coupled reliability to graph refinement; v49 generalises the feedback to a full re-triangulation step.
- **v45/v46:** v49 consumes and refines the per-view weights produced by v45-AGF and v46-SVG.

### Next steps

1. Wait for v48-domain smoke results (#164).
2. Implement `OnlineSelfEvolutionV49` and unit tests.
3. Wire v49 flags into `OmniMultiViewFusionV5` and the trainer.
4. Run smoke on RTX 4090 and verify reprojection-residual reduction.
5. Queue a full A800 run warm-started from the best v48 checkpoint.
