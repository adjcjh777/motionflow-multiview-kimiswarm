# v33 SMPL-aware 3D Human Prior Integration

**Task identifier:** `smpl_prior_integration`

## Problem Statement and Motivation

Current MotionFlow-MultiView models estimate 3D joint positions directly from multi-view 2D keypoints. While v22 introduced a Kinematic Anthropometric Prior (KAP) that regularizes bone lengths and v22 also had a prototype `SMPLPriorFusionV22`, the SMPL-aware prior is **not wired into the main v5 training script** and the prototype subclasses the full model rather than integrating as a toggleable module. As a result, the model still does not exploit the rich anthropometric and reprojection constraints of a parametric body model during end-to-end training.

Adding a properly integrated SMPL-aware prior can:

* Provide a compact, interpretable 3D human representation (shape β, pose θ, translation γ) for downstream applications.
* Regularize implausible body configurations via parametric body constraints, especially under heavy occlusion or few views.
* Enable reprojection, bone-length, and temporal-consistency losses that operate on the parametric body rather than only on 3D joints.
* Serve as a robust fallback when the multi-view evidence is noisy (variable-view / outlier-view scenarios).

## Proposed Architecture Changes

### 1. New module: `motionflow_mv/fusion/smpl_prior_v33.py`

Create a self-contained `SMPLPriorV33` module that plugs into `OmniMultiViewFusionV5` **instead of subclassing it**.

* **Input:** pooled per-joint feature `feat_pooled` (B·T, J, d) and current 3D estimate `pred_3d` (B·T, J, 3), same as the existing v22 KAP.
* **Heads:**
  * `betas`: clip-level shape `(B·T, 10)` — averaged over the clip for temporal consistency.
  * `body_pose`: axis-angle per-frame pose `(B·T, 69)`.
  * `global_orient`: axis-angle root orientation `(B·T, 3)`.
  * `transl`: per-frame translation `(B·T, 3)`.
  * `blend`: per-frame scalar α ∈ (0, 1) controlling the fusion of SMPL prior joints with the triangulation estimate.
* **Optional SMPL forward:** when `smplx` is available and `--smpl_model_path` is set, run `smplx.SMPL` to obtain `smpl_joints` and map the first 17/28 joints to the project skeleton. When unavailable, the module predicts parameters only and is trained with a pseudo-SMPL loss computed from the predicted 3D joints.
* **Output:**
  * `pred_3d_fused` — weighted blend between the model's triangulation estimate and the SMPL prior joints.
  * `smpl_out` dict for loss computation.

### 2. Integration into `OmniMultiViewFusionV5`

Add the following constructor flags (default `False`) in `motionflow_mv/fusion/omniview_fusion_v5.py`:

```python
use_smpl_prior_v33: bool = False,
smpl_model_path: Optional[str] = None,
smpl_prior_loss_weight: float = 0.01,
smpl_prior_blend_weight: float = 1.0,
smpl_prior_warmup_epochs: int = 0,
smpl_use_shape_reg: bool = True,
smpl_temporal_smooth_weight: float = 1e-3,
```

In `OmniMultiViewFusionV5.__init__`, instantiate `SMPLPriorV33` after the residual refinement (after line ~1104) when the flag is enabled. In the forward pass, place the SMPL prior **after** the residual refinement but **before** the v32 trajectory consistency and v28 physical-space losses, so the physical losses can operate on the SMPL-regularized pose:

```
residual refinement → SMPL prior v33 → v32 trajectory consistency → v28 physical losses
```

### 3. Losses

In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, extend the loss mix to include (when `use_smpl_prior_v33`):

* **3D joint supervision:** `MSE(pred_3d_fused, y)` already covered by the main loss; the SMPL head is trained end-to-end through it.
* **SMPL reprojection loss (optional):** project `smpl_joints` back into each view using the known cameras and compute a robust 2D reprojection loss weighted by confidences.
* **Bone-length consistency:** encourage bone lengths of `pred_3d_fused` and `smpl_joints` to match the learned SMPL bone lengths.
* **Shape regularization:** L2 penalty on `betas` and on the variance of `body_pose` over the clip to prevent degenerate shapes.
* **Temporal smoothness (training):** velocity loss on `body_pose` and `transl` to match the v32 trajectory-consistency spirit but on the parametric pose space.

### 4. CLI / Training Flags

Add to `experiments/train_omniview_fusion_v5_webbridge_multi.py`:

```python
parser.add_argument("--use_smpl_prior_v33", action="store_true", help="Enable SMPL-aware 3D human prior")
parser.add_argument("--smpl_model_path", type=str, default="data/smpl/SMPL_NEUTRAL.pkl", help="Path to SMPL neutral model")
parser.add_argument("--smpl_prior_loss_weight", type=float, default=0.01, help="Weight for SMPL prior auxiliary losses")
parser.add_argument("--smpl_prior_blend_weight", type=float, default=1.0, help="Initial/fixed weight for SMPL-triangulation blend supervision")
parser.add_argument("--smpl_prior_warmup_epochs", type=int, default=0, help="Epochs to ramp up SMPL prior losses")
parser.add_argument("--smpl_use_shape_reg", action="store_true", help="Enable shape and pose regularization on SMPL parameters")
parser.add_argument("--smpl_temporal_smooth_weight", type=float, default=1e-3, help="Temporal smoothness weight on body_pose/transl")
```

### 5. Data / Preprocessing Requirements

* Download the gender-neutral SMPL model (`SMPL_NEUTRAL.pkl`) into `data/smpl/`.
* Ensure `smplx` is installed in the environment (`pip install smplx`).
* Optional: pre-process AMASS / SPIN pseudo-labels to provide weak supervision for shape/pose when real 3D mocap is scarce (out of scope for first smoke).

## Training Command / Ablation Flags

### Smoke test (RTX 4090)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --smoke \
  --use_smpl_prior_v33 \
  --smpl_model_path data/smpl/SMPL_NEUTRAL.pkl \
  --smpl_prior_loss_weight 0.01 \
  --smpl_use_shape_reg
```

### Full run on A800-D

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
  --use_mixed_loader \
  --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding \
  --use_deformable_cross_view_attention_v18 \
  --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
  --use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1 \
  --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 \
  --use_smpl_prior_v33 \
  --smpl_model_path data/smpl/SMPL_NEUTRAL.pkl \
  --smpl_prior_loss_weight 0.01 \
  --smpl_use_shape_reg \
  --smpl_temporal_smooth_weight 1e-3 \
  --d 64 --residual_hidden 128 --n_st_layers 2 --epochs 20
```

### Ablations

* `v33_smpl_prior_full`: full integration as above.
* `v33_smpl_prior_freeze_base`: train only the SMPL prior head with base model frozen.
* `v33_smpl_prior_no_reproj`: disable SMPL reprojection loss to isolate the effect of parametric blending.
* `v33_smpl_prior_no_temporal`: disable temporal smoothness on SMPL pose parameters.

## Expected Metrics and Baseline to Beat

### Primary metrics

* **val_MPJPE** on the H36M + MPI-INF-3DHP mixed validation set.
  * Baseline v32 (no SMPL prior): use the best reported v32 val_MPJPE from `outputs/omniview_fusion_v32_combined_a800` or local smoke equivalents.
  * Target: **reduce val_MPJPE by ≥3–5%** relative to the v32 baseline on the mixed validation set.
* **Variable-view robustness:** evaluate with `--variable_view_min_views 2 --variable_view_max_views 14`. Target: lower relative degradation than the v32 baseline as views drop below 4.
* **Outlier-view robustness:** evaluate with `--outlier_view_prob 0.5 --outlier_view_max_views 2`. Target: ≥5% relative improvement in MPJPE under heavy outliers.

### Secondary metrics

* **PA-MPJPE** on H36M protocol (if ground-trumocap alignment is available).
* **Shape parameter stability:** variance of `betas` across validation clips should be small and centered.
* **Inference latency:** measure FPS impact of the SMPL forward pass.

## Risks / Unknowns

1. **Dependency / licensing:** `smplx` and the SMPL model file add an runtime dependency and licensing constraint. The module must degrade gracefully when `smplx` or the model file is absent.
2. **Computational cost:** running the parametric SMPL body for every training step may be slow and memory-intensive. Consider caching or running only when `smpl_prior_loss_weight > 0`.
3. **Skeleton mismatch:** SMPL has 24/6890 vertices; mapping to the 17-joint H36M / 28-joint MPI skeleton is approximate. A learned joint regressor may be needed beyond the naive first-17-joint mapping.
4. **Gender / shape bias:** using only `SMPL_NEUTRAL` may limit shape expressiveness. Gendered models (`SMPL_FEMALE.pkl`, `SMPL_MALE.pkl`) could be added later but complicate data loading.
5. **Gradient instability:** blending SMPL joints with triangulation estimates can create non-smooth gradients, especially when SMPL forward is unavailable and only parameter predictions are trained. Warm-up and clamping are essential.
6. **Overlap with v22 KAP:** the SMPL prior and KAP both regularize anthropometry. Running both simultaneously may over-constrain the model; ablate `use_kinematic_anthropometric_prior_v22` together with `use_smpl_prior_v33`.

## References

* Loper et al., "SMPL: A Skinned Multi-Person Linear Model", SIGGRAPH Asia 2015.
* Existing v22 prototype: `motionflow_mv/fusion/smpl_prior_fusion_v22.py`.
* Existing v22 KAP: `motionflow_mv/fusion/kinematic_anthropometric_prior_v22.py`.
