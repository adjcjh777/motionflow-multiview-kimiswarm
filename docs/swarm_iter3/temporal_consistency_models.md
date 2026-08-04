I completed the research pass over the code and design docs. The report content is below. Note: this read-only exploration agent does not have file-editing tools, so I cannot persist it to `docs/swarm_iter3/temporal_consistency_models.md`; you will need to write the file manually or have a coding agent do it.

---

# Temporal Consistency Models for Multi-View Human Pose/SMPL Fusion

**Venue target:** CVPR / ICRA 2027  
**Scope:** temporal smoothing, motion priors, and sequence-level consistency for multi-view 3D human pose and SMPL fitting in the `motionflow-multiview` pipeline.

---

## 1. Problem statement

Per-frame multi-view fusion (DLT triangulation, attention, or robust triangulation) produces high-quality 3D skeletons at each isolated frame, but it does not enforce **temporal consistency**. In video sequences this leads to:

- **High-frequency jitter** in 3D joint trajectories, even when per-frame reprojection error is low.
- **Physically implausible motion** (foot sliding, abrupt limb accelerations, root translation noise).
- **Outlier propagation:** a single bad frame from occlusion or 2D detector failure can corrupt downstream SMPL fitting/robot retargeting.
- **No motion prior:** current learned fusion operates purely on 2D observations, not on the space of plausible human motions.

The goal of a *temporal consistency model* is to refine a sequence of per-frame 3D poses/SMPL parameters so that the output is smooth, physically plausible, and robust to transient noise, while preserving accuracy. This is especially important when lifting to SMPL, because body shape (betas) and pose (body_pose/global_orient) must vary coherently over time.

---

## 2. Key related work / methods

The literature splits roughly into **temporal 3D pose smoothing**, **motion-prior-based filtering**, and **temporal SMPL fitting**.

### 2.1 Temporal 3D pose models

**Pavllo et al., "3D Human Pose Estimation in Video with Temporal Convolutions and Semi-Supervised Training", CVPR 2019.**  
Introduces dilated temporal convolutions over 2D keypoint sequences to lift to 3D. Key insight: a fully convolutional temporal model outperforms RNNs and enforces smoothness by design. Relevant for designing a temporal head on top of the per-frame DLT or attention output.

**Zhang et al., "Adaptive Multi-view and Temporal Fusing Transformer for 3D Human Pose Estimation", arXiv 2110.05092.**  
Proposes MFT, a transformer that fuses multi-view and temporal information jointly. Directly applicable to our setting: instead of separating view-fusion and temporal-fusion, a single transformer can attend across both space and time.

### 2.2 Temporal SMPL / parametric shape models

**Kim et al., "Beyond Static Features for Temporally Consistent 3D Human Pose and Shape from a Single Video", CVPR 2021 (TCMR).**  
Builds a temporal encoder that predicts SMPL parameters over a video window rather than per-frame. Uses a motion encoder to produce coherent pose sequences. Most relevant for lifting `motionflow-multiview` from skeleton-level fusion to SMPL-level temporal fusion.

**Rempe et al., "HuMoR: 3D Human Motion Model for Robust Pose Estimation", CVPR 2021.**  
A learned motion prior (VAE-style) trained on AMASS. Can be used to regularize or sample plausible human motions, and to denoise/novel-fill corrupted frames. Strong candidate for a regularizer inside the temporal refiner.

### 2.3 Spatial–temporal transformers and motion representations

**Zhu et al., "MixSTE: Mixed Spatial-Temporal Transformer for 3D Human Pose Estimation", CVPR 2022.**  
Alternates spatial and temporal attention blocks, capturing joint-to-joint and frame-to-frame dependencies separately. Useful if we want to keep the per-frame multi-view attention module but add a temporal transformer on top.

**Bian et al., "MotionBERT: A Pre-trained Model for Motion Analysis with Arbitrary Sequential Modalities", ICCV 2023.**  
A general-purpose motion representation learned from large motion data. Could be used as a pre-trained temporal encoder or as a target representation for the fused 3D skeleton/SMPL sequence.

### 2.4 Real-time / online variants

For ICRA-style real-time robotics, causal temporal models are needed. **VIBE (Kocabas et al., CVPR 2020)** and **WHAM (Kocabas et al., CVPR 2024)** use recurrent temporal encoders for monocular video; the same causal GRU/Transformer idea can be applied to multi-view fused features.

---

## 3. Relation to the current `motionflow-multiview` codebase

The repository already has a **temporal refiner** plugin (`motionflow_mv/fusion/temporal_refiner.py` and `temporal_refiner_module.py`):

- **Architecture:** a per-frame view-attention module + bidirectional GRU over a sliding window (default 7 frames) that refines the center frame.
- **Input:** per-view 2D keypoints + confidence `(B, T, V, J, 3)` and a DLT baseline `(B, T, J, 3)`.
- **Output:** refined 3D joint positions `(B, J, 3)` as a **residual delta** on top of the DLT baseline.
- **Training:** reprojection loss on Shelf frames (`experiments/train_temporal_refiner_shelf.py`), and a synthetic pre-training script (`experiments/train_temporal_synthetic.py`).
- **Current results (design_v3):** `temporal_refiner` matches DLT at ~9.88 px reprojection on Shelf and 1.52 px on Campus, confirming that the model is not degrading the geometric baseline but also not dramatically improving it yet.

### Gaps vs. state-of-the-art

| Aspect | Current `temporal_refiner` | State-of-the-art target |
|---|---|---|
| Representation | 3D skeleton joints | SMPL parameters / latent motion |
| Motion prior | None | HuMoR/VPoser/AMASS prior |
| Temporal model | Bidirectional GRU | Transformer / causal RNN / diffusion |
| Long-range context | ~7 frames | 30–120+ frames |
| Loss | Reprojection only | Reprojection + smoothness + contact + prior |
| Output | 3D joints only | Full temporally consistent `HumanMotionIR` |
| Online/causal | No | Needed for ICRA robotics |

The `HumanMotionIR` data class (`motionflow_mv/ir/human_motion_ir.py`) already stores `body_pose`, `global_orient`, `transl`, and `betas`, but the multi-view adapter (`multiview_adapter.py`) only **aligns the root translation** and keeps the reference view's body_pose/betas. There is no temporal SMPL refinement yet.

---

## 4. Concrete recommendations

### 4.1 Near-term (next 2–4 weeks)

**A. Strengthen the existing skeleton-level temporal refiner**

1. Add a **temporal smoothness loss** to `train_temporal_refiner_shelf.py`:
   - Velocity loss: `L_vel = ||P_t - P_{t-1}||^2`
   - Acceleration loss: `L_acc = ||P_{t+1} - 2P_t + P_{t-1}||^2`
   - Bone-length consistency loss over time.
2. Replace the GRU with a **temporal transformer** (MixSTE-style) to capture long-range dependencies and joint-joint interactions.
3. Train on **pseudo-GT** from DLT + robust triangulation, but evaluate on true 3D GT if available, or use MPJPE/PA-MPJPE and acceleration error in addition to reprojection.

**B. Add causal / online variant for ICRA**

Implement a **causal temporal refiner** that only uses past frames (unidirectional GRU/Transformer) with a small latency budget. This is a separate plugin (e.g., `online_temporal_refiner`) so the offline CVPR version remains unchanged.

### 4.2 Medium-term (next 1–2 months)

**C. Lift temporal refinement to SMPL parameter space**

Build a new plugin `smpl_temporal_refiner` that operates on `HumanMotionIR.pose`:

- Input: per-frame SMPL parameters from GVHMR/ScoreHMR per view, fused per-frame via DLT or attention.
- Temporal model: transformer encoder over a window of `(body_pose, global_orient, transl)` features.
- Use **VPoser** or **HuMoR** latent space to regularize `body_pose`.
- Predict per-frame corrections to `transl` and `body_pose` (and optionally `betas` constrained to a single shared shape).
- Loss: multi-view reprojection + SMPL smoothness + latent prior + foot-contact/ground-plane constraints.

**D. Pre-train on AMASS**

Before touching real multi-view data, train the temporal model on AMASS sequences (projected to 2D with synthetic cameras) to learn a strong motion prior. Then fine-tune on Shelf/Campus. This mirrors the successful synthetic→real pre-training already used for `temporal_refiner`.

### 4.3 Datasets and training

- **Synthetic pre-training:** Use `experiments/train_temporal_synthetic.py` style data, but generate **SMPL sequences** from AMASS rather than random skeletons.
- **Real fine-tuning:** Shelf and Campus (VoxelPose) with pseudo-GT from DLT/robust triangulation; if 3D GT is sparse, use reprojection loss.
- **Evaluation:** Add MPJPE, PA-MPJPE, **acceleration error** (jerk), and **foot sliding** metrics alongside reprojection error.

### 4.4 What *not* to do

- Do not add a heavy motion prior before the skeleton-level temporal refiner is stable.
- Do not discard the geometry-based DLT baseline; keep it as the initialization for any temporal/smooth model.
- Avoid non-causal models for the ICRA track.

---

## 5. Open questions / risks

| Risk / Question | Impact | Mitigation |
|---|---|---|
| **Cross-dataset generalization** | Temporal models trained on Shelf may overfit to camera layout and subject motions. | Synthetic AMASS pre-training; scale/camera augmentation. |
| **Lack of 3D ground truth** | Shelf pseudo-GT from DLT has errors; temporal model may learn DLT biases. | Use Campus/Human3.6M/CMU Panoptic with true GT where available. |
| **SMPL temporal model complexity** | Fitting SMPL over time requires body model, ground contact, and physics. | Start with skeleton-level transformer; then add SMPL regularizer. |
| **Real-time latency (ICRA)** | Bidirectional GRU/Transformer needs future frames. | Implement causal variant with bounded buffer. |
| **End-to-end differentiability** | Current DLT is used as a hard baseline, not jointly trained. | Integrate differentiable robust triangulation (`RobustTriangulationModel`) with the temporal module. |
| **External resource access** | Could not verify paper details online for this report. | Verify citations and exact method names before submission. |

---

**Bottom line:** The current `temporal_refiner` is a solid skeleton-level baseline but is only scratching the surface. For CVPR/ICRA-level impact, the next step is to replace the GRU with a transformer, add explicit temporal smoothness losses, and ultimately lift the temporal model from 3D joints to SMPL parameter space with an AMASS-learned motion prior.