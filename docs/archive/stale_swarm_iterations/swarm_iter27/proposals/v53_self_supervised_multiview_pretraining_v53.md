# v53 Self-Supervised Multi-View Pre-Training Head (SS-MVP v53)

**Author:** design-swarm-agent-v53  
**Tracking issue:** #201  
**Depends on:** v45-AGF, v46-SVG, v50-SEFH, v51-CDSVR, v52-UWT  
**Target pipeline stage:** multi-view fusion and calibration, after v52 Uncertainty-Weighted Triangulation and before v46/v47/v48/v49/v50/v51 downstream heads.

## 1. Motivation

The MotionFlow pipeline now triangulates 3-D poses with v52 Uncertainty-Weighted Triangulation (UWT), which learns per-view per-joint precision from supervised 3-D labels. SS-MVP v53 exploits the redundancy of calibrated multi-view video itself: if the same pose is observed from several views, dropping a random subset of views should still produce a 3-D estimate that is consistent with the full-view estimate. By masking views (or joints, or short time windows) and asking v52 UWT to reproduce the full-view triangulation, the model learns robust precision weights without extra 3-D annotations. This directly refines the *multi-view fusion and calibration* stage of the paper story

```
multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized motionflow pipeline
```

and is expected to make v52 UWT generalize better under sparse views and domain shift.

## 2. Architecture

SS-MVP v53 is a pure auxiliary loss head. It does **not** change the forward pose estimate, so the main branch remains identical when the module is enabled.

### 2.1 Inputs from v52 UWT

After the v52 UWT call in `OmniMultiViewFusionV5.forward`, we have

* `pred_3d_full ∈ R^(B×T×J×3)` — full-view triangulated 3-D pose (the main-branch output).
* `uwt_weights ∈ R^(B×T×V×J)` — v52 triangulation weights, already sigmoid-activated.
* `uwt_log_precision ∈ R^(B×T×V×J)` — raw per-view per-joint log-precision.
* `points_2d ∈ R^(B×T×V×J×2)` — detected 2-D keypoints.
* Camera projection matrices `P ∈ R^(B×T×V×3×4)`.
* `view_mask ∈ {0,1}^(B×T×V)` — existing binary mask.

### 2.2 Masked-view sampling

For each training sample, draw a random corruption mask

```
M ∈ {0,1}^(B×T×V×J)
```

with per-token probability `v53_ssmvp_mask_ratio` (default 0.25). The mask is constrained so that every joint/time keeps at least `v53_ssmvp_min_visible_views` unmasked views and at most `1 - v53_ssmvp_min_visible_ratio` of all tokens are dropped, to avoid degenerate triangulation.

Three corruption modes are sampled with equal probability:

| Mode | Masked tokens | What must be reconstructed |
|------|---------------|----------------------------|
| `view` | one view for a joint/time | the 3-D point using the remaining views |
| `joint` | a joint across all views | the 3-D point of that joint from the other joints' views |
| `time` | a short contiguous window for one view | the 3-D track from temporal neighbours |

### 2.3 Masked triangulation

Using the existing v52 UWT weights, zero out the masked views:

```
w_mask = uwt_weights * M
```

Then run the same differentiable DLT used by v25/v45/v52:

```
pred_3d_mask = DLT(points_2d, P, w_mask) ∈ R^(B×T×J×3)
```

Joints/time steps with fewer than two visible views are skipped (`valid ∈ {0,1}^(B×T×J)` computed from `M.sum(dim=-1) >= 2`).

### 2.4 Self-supervised consistency loss

The full-view triangulation acts as a self-supervised pseudo-target (detached):

```
L_cons = (1 / (B T J)) Σ valid · || pred_3d_mask − stop_grad(pred_3d_full) ||²_2
```

A re-projection term anchors the masked prediction to the visible 2-D evidence:

```
L_reproj = (1 / (B T V J)) Σ M · || Π_v(pred_3d_mask) − points_2d ||²_2
```

Total auxiliary loss:

```
L_v53 = v53_ssmvp_loss_weight · [ λ_cons · L_cons + λ_reproj · L_reproj ]
```

Default: `λ_cons = 1.0`, `λ_reproj = 0.5`.

### 2.5 Identity-at-init / warm-start

SS-MVP v53 introduces no new parameters that modify the forward pass. Any internal mask-conditioning MLP (if used for adaptive weight scaling) is zero-initialised, so `w_mask` equals the v52 weights multiplied by the binary mask. The main 3-D output `pred_3d_full` is unchanged, so loading a v52 checkpoint with v53 enabled preserves the baseline val_MPJPE exactly.

## 3. Inputs / Outputs

### Inputs

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred_3d_full` | `(B, T, J, 3)` | v52 UWT full-view triangulated pose |
| `uwt_weights` | `(B, T, V, J)` | v52 triangulation weights |
| `uwt_log_precision` | `(B, T, V, J)` | raw log-precision from v52 |
| `points_2d` | `(B, T, V, J, 2)` | detected 2-D keypoints |
| `P` | `(B, T, V, 3, 4)` | camera projection matrices |
| `view_mask` | `(B, T, V)` | binary view availability mask |
| `domain_id` | `(B,)` | optional domain label for domain-conditional masking statistics |

### Outputs

| Symbol | Shape | Description |
|--------|-------|-------------|
| `v53_aux_loss` | scalar | `L_v53`, added to the training objective |

The main forward branch (`pred_3d_full` and all downstream v46/v47/v48/v49/v50/v51 heads) is unchanged.

## 4. Config flags

```yaml
use_v53_self_supervised_pretraining: false   # master toggle
v53_ssmvp_mask_ratio: 0.25                  # fraction of tokens masked per sample
v53_ssmvp_min_visible_views: 2               # minimum unmasked views per joint/time
v53_ssmvp_min_visible_ratio: 0.5           # never mask more than half of all tokens
v53_ssmvp_hidden: 64                       # optional mask-adaptation MLP hidden dim
v53_ssmvp_n_layers: 2                      # optional mask-adaptation MLP layers
v53_ssmvp_loss_weight: 0.05                # weight of L_v53 in total loss
v53_ssmvp_warmup_epochs: 1                 # epochs before loss is active; 0 = always
v53_ssmvp_temporal_window: 3               # window size for time-mode corruption
v53_ssmvp_use_reproj_term: true            # enable L_reproj
v53_ssmvp_use_consistency_term: true       # enable L_cons
v53_ssmvp_detach_full_target: true         # detach full-view pseudo-target
v53_ssmvp_identity_init: true              # zero-initialise any internal projections
```

## 5. Expected MPJPE impact

* **H36M / MPI-INF-3DHP (full supervision):** modest gain of ≈0.5–1.2 mm by regularising v52 UWT precision weights.
* **Sparse-view settings (v46):** larger gain of 1.5–3.5 mm because the model is explicitly trained to triangulate with missing views.
* **Cross-domain / limited-label scenarios (v48, v51):** 2–4 mm improvement when 3-D labels are scarce; the auxiliary loss only needs calibrated multi-view video.
* **3DPW actual / out-of-domain:** 1–2 mm gain from better generalised triangulation weights.

## 6. Risks

See `docs/swarm_iter27/reports/agent_self_supervised_multiview_pretraining_v53_risks.md` for the detailed risk matrix.

## 7. Five-step implementation plan

1. **Add the module file** `motionflow_mv/fusion/self_supervised_multiview_pretraining_v53.py` implementing `SelfSupervisedMultiViewPretrainingV53(nn.Module)` with the masked-view sampler, masked DLT call, and consistency/re-projection losses. Ensure any internal projection is zero-initialised.

2. **Wire into `OmniMultiViewFusionV5`**: add the `use_v53_*` flags in `__init__`, instantiate the head after the v52 UWT call, and invoke it with `pred_3d_full`, `uwt_weights`, `uwt_log_precision`, `points_2d`, `P`, and `view_mask`. Store the returned `v53_aux_loss`.

3. **Update the trainer** (`experiments/train_omniview_fusion_v5_webbridge_multi.py`) to add `v53_aux_loss` to the total loss with weight `v53_ssmvp_loss_weight`, and to respect `v53_ssmvp_warmup_epochs` so the loss is zero before warm-up. Apply the masking only during training; at evaluation the module is a no-op.

4. **Add smoke config** `configs/benchmark_v53_ssmvp_smoke.yaml` copied from the v52 UWT smoke config, enable `use_v53_self_supervised_pretraining`, and create `scripts/run_v53_ssmvp_smoke_local_4090.sh`. Acceptance: smoke finishes, val_MPJPE is within 0.1 mm of the v52-only baseline, and the auxiliary loss is finite and non-dominating.

5. **Ablate on the v52 URT baseline**: run a controlled 2-epoch comparison (v52 vs. v52+v53) on the local RTX 4090 with the same seed. If val_MPJPE is neutral or improved, add an A800 queue entry to `scripts/launch_v33_a800_queue.py` warm-starting from the best v52 checkpoint.
