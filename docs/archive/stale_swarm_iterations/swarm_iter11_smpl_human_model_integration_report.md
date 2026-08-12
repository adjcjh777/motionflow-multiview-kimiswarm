# SMPL Human Model Integration — Iter11+ Research Report

## 1. Current state

The best checkpoint on MPI-INF-3DHP is `RayAttentionFusionModelTemporalResidual`
at **11.17 mm MPJPE** (PA-MPJPE 8.24 mm). A more advanced architecture now
combines cross-view spatio-temporal attention, uncertainty-weighted DLT,
differentiable Gauss-Newton triangulation, and a residual MLP.

SMPL integration is still **post-hoc and offline**:

* `experiments/fit_smpl_multiview.py` fits SMPL to already-fused 3D joints but
  does not use raw 2D observations or learned per-view weights.
* `motionflow_mv/ir/human_motion_ir.py` stores SMPL parameters, but they are
  inherited from single-view detectors, not predicted by the fusion model.
* `experiments/generate_synthetic_multiview_dataset.py` discards GT SMPL
  parameters, so parametric evaluation is impossible.
* Skeleton-aware losses exist in `experiments/train_utils.py` but are not used
  by the current best model.

**Bottom line:** the project is a high-performing *joint triangulation* system;
it is not yet a *parametric body recovery* system. SMPL integration can become a
trainable inductive bias that improves MPJPE, enforces anatomical plausibility,
and yields robot-ready `HumanMotionIR` outputs.

## 2. Concrete, implementable improvements

### 2.1 Add a differentiable SMPL decoder to the fusion model

After spatio-temporal feature extraction, the pooled per-joint features encode
motion and multi-view evidence. Attach a small SMPL parameter head:

* **Shape head:** regress `betas` shared across the clip.
* **Pose head:** regress `global_orient` and `body_pose` per frame.
* **Translation head:** regress `transl` per frame (or reuse the triangulated root).

Run SMPL forward to obtain the canonical 24 joints, project the 17-joint subset,
and combine with the existing residual MLP. Training uses both 3D joint loss and
a weighted SMPL reprojection loss over the original 2D observations.

Why it may improve MPJPE: the SMPL layer is a strong 3D skeleton prior that
removes anatomically impossible configurations; shared `betas` regularizes the
skeleton across the clip, and the reprojection loss couples the parametric body
to raw 2D evidence.

### 2.2 Skeleton/bone-length regularizer on the residual head

Add an unsupervised bone-length consistency loss using helpers in
`experiments/train_utils.py`:

```python
L_bone = bone_length_loss(pred, gt, parents=SMPL17_PARENTS, weight=0.01)
L_temporal = temporal_bone_length_consistency_loss(pred, parents=SMPL17_PARENTS, weight=0.005)
```

This is the lowest-risk first experiment: it does not change the model
architecture, only the training loss.

### 2.3 End-to-end SMPL fitting as a post-processing stage

Replace the offline `fit_smpl_multiview.py` with a post-fusion stage that
consumes both fused 3D joints **and** the ray-attention per-view weights:

* Use the predicted weights to initialize a robust multi-view reprojection loss.
* Add a temporal smoothness term on `body_pose` and `transl`.
* Return valid SMPL parameters in the `HumanMotionIR`.

### 2.4 Synthetic ground-truth SMPL benchmark

Extend `experiments/generate_synthetic_multiview_dataset.py` to store the GT
`betas`, `global_orient`, `body_pose`, and `transl`. Create
`experiments/eval_smpl_param_synthetic.py` to evaluate SMPL-parameter recovery
(in addition to 3D MPJPE) so that SMPL integration gains can be measured before
real SMPL-annotated data is available.

## 3. Experiments to run

1. **Bone-length auxiliary loss smoke test** (1–2 days)
   * Copy `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py`
     and add `bone_length_loss` and `temporal_bone_length_consistency_loss`.
   * Train on MPI-INF-3DHP S1 Seq1+Seq2 → S2 Seq1, clip_len=13, 5 epochs.
   * Target: MPJPE < 11.17 mm on S2/Seq1 without divergence.

2. **SMPL decoder head on synthetic data** (3–5 days)
   * Generate a synthetic dataset that also saves GT SMPL parameters.
   * Implement `SMPLParamHead` and train the advanced model with both 3D MSE and
     SMPL reprojection loss.
   * Evaluate SMPL parameter error, 3D MPJPE, and per-view reprojection error.

3. **Post-fusion SMPL fitting with ray weights** (2–3 days)
   * Update `fit_smpl_multiview.py` to accept the fusion model's per-view weights
     and run robust reprojection fitting.
   * Evaluate on synthetic data and on H36M WebBridge `s_01_*_multiview_m.npz`.

4. **MPI-INF-3DHP end-to-end SMPL model** (1–2 weeks)
   * Add the SMPL decoder to the advanced fusion model.
   * Train on MPI-INF-3DHP; monitor MPJPE, PA-MPJPE, and SMPL parameter stability.

## 4. Metrics to track

* **3D pose:** MPJPE, PA-MPJPE, PCK@50/100/150 mm, AUC (already in
  `motionflow_mv/eval/metrics.py`).
* **Parametric accuracy:** `betas` L2 error vs. GT/pseudo-GT; mean per-joint
  body-pose axis-angle error; per-view 2D reprojection error in pixels.
* **Anatomical plausibility:** bone-length variance across a clip; percentage of
  frames with self-intersecting limbs.
* **Runtime:** ms per clip and throughput clips/s on RTX 4090.

## 5. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| **No real SMPL ground truth** | Rely on synthetic GT first; use ScoreHMR pseudo-GT only for cross-checking. |
| **Joint-set mismatch** (SMPL 24 vs. benchmark 17) | Define a fixed 17-joint regressor/mapping and document it. |
| **SMPL forward is slower** | Keep the decoder lightweight; profile before committing. |
| **Windows NumPy BLAS instability** | Continue using `torch.linalg.svd`; avoid NumPy SVD. |
| **A800-D read-only** | Train only on the local RTX 4090/WSL environment. |
| **Auxiliary loss dominates MSE** | Start with small weights (bone 0.01, temporal 0.005). |
| **Gradient instability through SMPL layer** | Use gradient clipping and warm-up the SMPL head frozen. |

## 6. Pseudo-code for the proposed SMPL decoder integration

```python
import torch.nn as nn
import smplx

class SMPLParamHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.betas_head = nn.Linear(d, 10)
        self.global_orient_head = nn.Linear(d, 3)
        self.body_pose_head = nn.Linear(d, 69)
        self.transl_head = nn.Linear(d, 3)

    def forward(self, feat: torch.Tensor, smpl_model: smplx.SMPL):
        z = feat.mean(dim=1)                                  # (B*T, d)
        betas = self.betas_head(z).mean(dim=0, keepdim=True)  # (1, 10)
        global_orient = self.global_orient_head(z)            # (B*T, 3)
        body_pose = self.body_pose_head(z)                    # (B*T, 69)
        transl = self.transl_head(z)                          # (B*T, 3)

        pred = smpl_model(
            betas=betas.expand(z.shape[0], -1),
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
        )
        pred_joints = pred.joints[:, :17, :]  # (B*T, 17, 3)
        return pred_joints, {
            "betas": betas,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "transl": transl,
        }

# Training loop
smpl_joints, smpl_params = smpl_head(feat, smpl_model)
loss = mse(pred_3d, gt_3d) + 0.5 * mse(smpl_joints, gt_3d) \
     + reproj_weight * reproj_loss(smpl_joints, points_2d, cameras, confidences) \
     + bone_weight * bone_length_loss(smpl_joints, gt_3d, parents=SMPL17_PARENTS)
```

## 7. Recommendation

Start with **Improvement 2.2** (bone-length auxiliary loss) because it is the
lowest-risk and reuses existing code. If the smoke test does not regress the
11.17 mm baseline, proceed to **Improvement 2.1** by adding a lightweight SMPL
decoder head on the synthetic dataset. Only after synthetic validation should the
SMPL decoder be trained on MPI-INF-3DHP. This staged plan keeps the project on
the ICRA/CVPR 2027 roadmap while producing measurable MPJPE or paper-quality
improvements at each step.
