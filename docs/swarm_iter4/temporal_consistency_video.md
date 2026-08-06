# Temporal Consistency for Multi-View Video

**Target venues:** CVPR / ICRA 2027  
**Scope:** sequence models, temporal smoothing, and trajectory priors for calibrated multi-view 3D human pose.

---

## 1. Current state

The per-frame `ray_attention` plugin (`motionflow_mv/fusion/ray_attention_model.py`) triangulates with near-perfect metric accuracy on synthetic data and improves over earlier attention variants on Shelf/Campus. Its differentiable weighted-DLT layer provides a strong geometric inductive bias, but it is still strictly per-frame.

The existing `temporal_refiner` plugin (`motionflow_mv/fusion/temporal_refiner.py`) slides a 7-frame window over DLT baselines and refines the center frame with a bidirectional GRU plus per-frame view attention. It matches DLT on Shelf (~10 px reprojection) and generalizes to Campus (1.52 px), but it does not yet improve over the geometric baseline. Missing pieces: explicit temporal losses, longer context, and a motion prior.

`HumanMotionIR` already stores `body_pose`, `global_orient`, `transl`, and `betas`, but no temporal SMPL refinement exists yet.

---

## 2. Brief survey

**Temporal 3D pose smoothing.** Pavllo et al. (CVPR 2019) used dilated temporal convolutions over 2D keypoint sequences. MixSTE (Zhu et al., CVPR 2022) alternates spatial and temporal transformers, which is a natural fit on top of `ray_attention`.

**Motion priors.** HuMoR (Rempe et al., CVPR 2021) and VPoser are AMASS-trained priors that can regularize SMPL sequences or impute corrupted frames. TCMR (Kim et al., CVPR 2021) predicts temporally coherent SMPL parameters over a window.

**Online variants.** VIBE/WHAM-style causal recurrent encoders are the ICRA-relevant counterpart to offline bidirectional models.

The high-impact opportunity for this project is a *multi-view temporal refiner* that exploits calibrated cameras and the per-view weights from `ray_attention`.

---

## 3. Concrete recommendations

### 3.1 Add explicit temporal losses to the skeleton-level refiner

`experiments/train_temporal_refiner_shelf.py` currently uses only a 3D MSE or reprojection loss. Add:

- **Velocity loss:** `L_vel = ||P_t - P_{t-1}||_2`
- **Acceleration loss:** `L_acc = ||P_{t+1} - 2P_t + P_{t-1}||_2`
- **Bone-length consistency:** `L_bone = ||b_t - b_{t-1}||_2`

Start with small weights (~0.01× the main loss) so the model still fits 3D GT. This targets jitter without adding parameters.

### 3.2 Upgrade the temporal backbone from GRU to transformer

Replace the bidirectional GRU in `temporal_refiner.py` with a small transformer encoder (MixSTE-style) and increase the window from 7 to 31–61 frames. Keep the per-frame `ray_attention` view encoder but allow cross-frame self-attention over fused per-joint features. This addresses the GRU's limited context and independent per-joint processing.

### 3.3 Feed `ray_attention` view weights into the temporal model

Currently `temporal_refiner` ignores the per-view weights that `ray_attention` computes. Modify the temporal model to consume both the per-frame 3D DLT baseline and the `ray_attention` weight maps `(B, T, V, J)`. The temporal model can then learn to trust or distrust specific views over time, e.g., down-weighting a view during occlusion and propagating that information across frames.

### 3.4 Pre-train on AMASS with synthetic multi-view projection

Before fine-tuning on Shelf/Campus, pre-train the temporal transformer on AMASS sequences projected through randomized calibrated rigs. This extends the existing synthetic pipeline (`experiments/generate_synthetic_multiview_dataset.py`) from random poses to real human motion. Expected benefit: stronger motion prior and better cross-dataset generalization.

### 3.5 Add a causal online variant for ICRA

For robotics, implement a causal version (unidirectional GRU/Transformer or a small state-space model) with bounded latency. Register it as a separate plugin, e.g., `online_temporal_refiner`, so the offline CVPR model remains unchanged. Evaluate with per-frame latency and jitter metrics, not just MPJPE.

---

## 4. Potential risks and mitigations

| Risk | Mitigation |
|------|------------|
| **Over-smoothing** | Keep a residual connection to the per-frame DLT/ray-attention output and weight temporal losses low initially. |
| **Dataset bias** | AMASS pre-training; scale and camera augmentation; cross-dataset validation on Campus. |
| **Lack of 3D ground truth** | Use DLT/robust triangulation pseudo-GT; evaluate on Campus and, if available, Human3.6M/CMU Panoptic. |
| **Real-time latency** | Use a causal variant with a bounded buffer; benchmark latency on target hardware. |
| **SMPL complexity** | Defer full SMPL temporal fitting until skeleton-level temporal refiner is stable. |

---

## 5. Fit into the paper plan

Temporal consistency is the natural *third act* of the paper:

1. **Geometry-aware fusion:** `ray_attention` with weighted DLT (done).
2. **Robust multi-view fusion:** occlusion/outlier handling via learned view weights (in progress).
3. **Temporal refinement:** smooth, physically plausible 3D trajectories over time (this topic).
4. **SMPL lifting:** from 3D skeletons to temporally coherent `HumanMotionIR` (next phase).

The ICRA angle emphasizes the causal online variant and latency; the CVPR angle emphasizes the offline transformer with AMASS pre-training and strong MPJPE/PA-MPJPE numbers.

**Next immediate step:** implement recommendation 3.1 (temporal losses) on the existing `temporal_refiner` and measure whether it still matches DLT on reprojection while reducing acceleration error.
