# Proposal: Neural Implicit 3D Pose Field for Multi-View Skeleton Refinement

**Author:** MotionFlow-MultiView iter15 swarm — agent direction "neural-implicit-3d-pose-field"  
**Date:** 2026-08-06  
**Empirical anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — MPI-INF-3DHP clean **9.32 mm** MPJPE / **5.37 mm** PA-MPJPE  
**Related existing files/modules:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`, `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py`, `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py`, `motionflow_mv/losses/bone_length.py`, `motionflow_mv/losses/reprojection.py`.

---

## 1. Problem

The anchor model triangulates each joint independently with weighted DLT and then applies a per-joint MLP residual. This treats every joint as a free point in space and has no explicit notion of a coherent 3-D skeleton manifold: (1) noisy or occluded views can push a joint off the true surface, (2) the residual MLP operates independently per joint, and (3) there is no geometric prior that ties refined points back to multi-view reprojection consistency.

## 2. Hypothesis

Adding a lightweight, joint-conditioned neural implicit 3-D pose field that refines the DLT-initialized skeleton by walking along the field gradient toward the zero level-set will improve clean and corrupted-view accuracy, because the shared field encodes a per-joint geometric prior and can move points toward configurations that are consistent with all calibrated cameras.

## 3. Method

### 3.1 Architecture change

Create a new model that subclasses the anchor and replaces the dense residual MLP with a **neural implicit pose field refiner**.

- **New field module:** `motionflow_mv/fusion/neural_implicit_pose_field.py`
  - `SinusoidalPositionalEncoding`: maps a 3-D coordinate to a sinusoidal positional embedding (default 6 frequency octaves).
  - `NeuralImplicitPoseField(j, feat_dim, hidden_dim, num_layers, pe_freq)`:
    - Input: 3-D position `(N, J, 3)`, per-joint spatio-temporal feature `(N, J, d)`, and optional joint ids.
    - Joint embedding `nn.Embedding(J, 16)` to give the field a per-joint semantic identity.
    - MLP with `num_layers` hidden ReLU layers and a final scalar field head.
    - Output: scalar field value `f(p) ∈ ℝ` for each joint. The zero level-set `f(p)=0` is trained to coincide with the true 3-D joint location.
  - `NeuralImplicitPoseFieldRefiner(j, feat_dim, hidden_dim, num_layers, n_iters, step_size)`:
    - Input: the anchor's `residual_input = [pooled_spatiotemporal_feature; raw_3d]` of shape `(N, J, d+3)`.
    - Splits into feature `(N, J, d)` and initial position `X₀`.
    - Iterates `k = 1 … n_iters`:
      - Query `fₖ = field(Xₖ, feature)`.
      - Compute `∇ₓ fₖ` w.r.t. `Xₖ` via `torch.autograd.grad`.
      - Take a normalized Newton step: `X₊₁ = Xₖ - step_size · fₖ · ∇f / (‖∇f‖² + ε)`.
    - Returns the residual `Δ = Xₙ - X₀`, which the anchor adds to `X₀` exactly as it did with the MLP.

- **New model file:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_implicit_field_model.py`
  - Class: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointImplicitField`
  - Inherits from the 9.32 mm anchor `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
  - In `__init__`, after the parent init, replaces `self.residual_mlp` with a `NeuralImplicitPoseFieldRefiner`.
  - No other forward logic is changed, so the model remains a drop-in replacement in the existing trainer.

### 3.2 Loss / data changes

- No new dataset is required; reuse the MPI-INF-3DHP `.npz` clips as the anchor.
- The primary loss remains MPJPE (MSE on refined 3-D pose), so gradients flow from the final refined joints through the implicit-field Newton steps and into the field network weights.
- Optional auxiliary losses (to be enabled only if smoke shows drift):
  - **Eikonal regularization:** `(|∇f| - 1)²` on random 3-D samples around the raw DLT point, encouraging the field to behave like a signed distance field.
  - **Reprojection field loss:** after field refinement, apply the existing `motionflow_mv/losses/reprojection.py` on the refined joints to keep them camera-consistent.
  - These are exposed in `motionflow_mv/losses/implicit_pose_field_loss.py` but are not required for the smoke test.

### 3.3 Exact insertion point

In `ray_attention_temporal_crossview_residual_principal_point_implicit_field_model.py` the only change is:

```python
self.residual_mlp = NeuralImplicitPoseFieldRefiner(
    j=self.j,
    feat_dim=self.d,
    hidden_dim=field_hidden,
    num_layers=field_layers,
    n_iters=field_iters,
    step_size=field_step_size,
)
```

All ray embedding, PP correction, cross-view spatio-temporal attention, and DLT triangulation remain identical to the anchor.

### 3.4 Files to create / modify

| File | Action | Purpose |
|------|--------|---------|
| `motionflow_mv/fusion/neural_implicit_pose_field.py` | Create | `NeuralImplicitPoseField` and `NeuralImplicitPoseFieldRefiner` modules. |
| `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_implicit_field_model.py` | Create | New anchor subclass with the implicit-field residual refiner. |
| `motionflow_mv/losses/implicit_pose_field_loss.py` | Create | Optional eikonal and reprojection-consistent field losses. |
| `experiments/train_ray_attention_temporal_crossview_residual_principal_point_implicit_field_mpiinf3dhp.py` | Create | Smoke trainer (copy of the anchor trainer, instantiates the new class). |
| `tests/test_neural_implicit_pose_field.py` | Create | CPU sanity: forward + backward for the refiner and the full model. |

## 4. Smoke-Test Plan

Run a 5-epoch smoke on a 500-sample subset of MPI-INF-3DHP, matching the existing PP-graph / factorized smoke settings for comparability.

```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_implicit_field_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --field_hidden 64 --field_layers 2 --field_iters 1 --field_step_size 0.5 \
  --epochs 5 --batch_size 4 --lr 1e-3
```

| Setting | Value |
|---|---|
| Train | `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz` (500 random clips) |
| Val | `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz` |
| Clip length | 13 |
| Batch size | 4 |
| Backbone | `d=32`, `residual_hidden=64`, `n_st_layers=2` |
| Field refiner | `field_hidden=64`, `field_layers=2`, `field_iters=1`, `step_size=0.5` |
| Optimizer | Adam, `lr=1e-3` |
| Loss | MPJPE only (no auxiliary field losses in smoke) |
| Epochs | 5 |

**Pass / fail criteria:**

- **Pass:** Training completes 5 epochs without NaN/Inf; final val MPJPE ≤ 60 mm on the smoke-sized model.
- **Pass:** CPU sanity test `tests/test_neural_implicit_pose_field.py` runs in < 1 min and confirms finite gradients through the field refiner.
- **Pass:** Runtime per epoch is within 2× of the anchor PP trainer on the same smoke config.
- **Fail:** Any crash, NaN loss, epoch time > 3× anchor, or val MPJPE > 80 mm.

## 5. Evaluation Plan

After a successful smoke, evaluate with the existing harness:

- **Clean metrics:** `experiments/eval_full_metrics.py --model implicit_field_pp --checkpoint <smoke_checkpoint>`
  - Report MPJPE, PA-MPJPE, PCK@50/100/150, AUC.
  - Target: clean val MPJPE ≤ 9.6 mm (within ~3% of the 9.32 mm anchor).
- **Robustness matrix:** `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` on the new checkpoint.
  - Hypothesized win axes: `view_dropout_*`, `joint_dropout_*`, and `noise_*` corruptions, because the shared field prior can hallucate or correct missing/uncertain joints.
- **Ablation:** sweep `field_iters ∈ {0, 1, 3}` and `field_layers ∈ {2, 3}` to isolate the contribution of the field itself.

## 6. Estimated GPU/CPU Cost on RTX 4090

| Phase | Hardware | Time |
|---|---|---|
| CPU sanity (`tests/test_neural_implicit_pose_field.py`) | CPU | < 1 min |
| Smoke (5 epochs, d=32, ~500 clips) | RTX 4090 | ~20–30 min |
| Full validation run (S2/Seq1) | RTX 4090 | ~2–5 min |
| Full training if smoke passes (30 epochs, d=64) | RTX 4090 | ~6–10 h; ~1.2–1.5× the anchor due to the second-order field gradient. |

The field network adds only ~20–50k parameters (a few MLP layers), so memory overhead is negligible.

## 7. Risks & Fallback

| Risk | Likelihood | Mitigation / Fallback |
|---|---|---|
| The field Newton step requires `torch.autograd.grad` inside `forward`, which can be slow or numerically unstable. | Medium | Set `field_iters=1` for smoke; if unstable, disable the field step and use the field network as a direct residual MLP (`Δ = MLP([feat, X₀]) - X₀`), which falls back to a dense residual head. |
| The zero level-set collapses or the field overfits to the training distribution. | Medium | If smoke MPJPE is > 5 mm worse than the anchor, remove the eikonal loss and reduce `field_hidden` to 32; if still worse, archive the experiment. |
| Second-order gradients through the Newton step cause OOM on longer clips. | Low | Detach the field normal `∇f` before the update (done in the skeleton) so backward is first-order; if OOM persists, set `field_iters=0` to recover the anchor. |
| No clean improvement over the 9.32 mm anchor. | Medium-High | If the field matches the anchor within 0.3 mm it can be kept as a regularizer. If it consistently loses, abandon and keep the existing MLP residual head. |

## 8. Self-Evolution Mapping

- **Reflect:** The anchor triangulates and refines joints independently; there is no joint-level or manifold-level 3-D prior.
- **Hypothesize:** A joint-conditioned neural implicit field can serve as a lightweight 3-D prior that refines DLT output toward a coherent skeleton manifold.
- **Smoke-validate:** 5-epoch smoke checks stability, gradient health, and clean sanity metrics.
- **Integrate:** If clean accuracy holds and robustness improves on any axis, promote the implicit field as the default residual refiner and explore extending it to a full temporal implicit sequence model.
