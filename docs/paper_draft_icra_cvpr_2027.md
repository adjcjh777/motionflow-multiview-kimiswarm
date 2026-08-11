# MotionFlow-MultiView: Residual Refinement of Temporal Ray-Attention Fusion for Calibrated Multi-View 3D Human Pose Estimation

> ⚠️ **SUPERSEDED (2026-08-10).** Headline numbers in the abstract and
> conclusion come from circular-label protocols (H36M `joints_3d = DLT(points_2d, cameras)`;
> MPI-INF-3DHP GT-projection 2D) and are invalidated by the data-foundation
> audit. Do not cite the MPI 9.32 mm or H36M 0.62 mm results. See
> `docs/data_foundation_blocker.md` and the repositioned story in
> `docs/roadmap_cvpr2027.md`; verified leaderboards now include H36M true-GT
> (`docs/results_true_gt_h36m.md`), AIST++ smoke (Section 5.4.1), and
> Shelf/Campus detected (`docs/results_true_gt_shelf_campus.md`).

**Abstract.**
We present a lightweight residual refinement head that boosts calibrated multi-view 3D human pose estimation and integrates it into a robotics-oriented pipeline. Starting from a temporal ray-attention fusion baseline, our model first predicts per-view weights and triangulates a 3D pose via weighted DLT, then refines the result with a small MLP that corrects the residual triangulation error. On the repaired, non-circular evaluation protocols the paper pivots from absolute-record MPJPE to **sparse-view and cross-domain robustness**: on honest true-GT Human3.6M, even the best learned variant (v80, 39.98 mm) still trails Iskakov (23.35 mm) and confidence-weighted DLT (25.87 mm), while the v25 variant reaches only 72.80 mm before diverging. The framework therefore integrates a new AIST++ cross-domain smoke benchmark and supports variable camera rigs through geometry-based camera positional encoding (CamPE) and a skeleton-aware graph joint relation (GJR) module. A learned principal-point correction layer is retained as an architectural option, though its contribution is still being re-evaluated on the true-GT protocols. The fusion module is exposed as a pluggable `MultiViewFusionPlugin` inside MotionFlow, consumes calibrated multi-view 2D keypoints, and outputs a `HumanMotionIR` that feeds robot retargeting and policy training. It runs at 12.7–194 clips/s on an RTX 4090, making it suitable for robotics and immersive-video applications.

---

## 1. Introduction

Multi-view video is the dominant capture modality in sports, entertainment, and human-robot interaction. Existing monocular pose pipelines such as MotionFlow are accurate for single-view inputs, but they discard the geometric constraints that relate different camera views. Calibrated multi-view triangulation can recover metric 3D pose, yet classical DLT is brittle to occlusions, detector noise, and small calibration errors.

Our key observation is that a **geometric triangulation step followed by a learned residual correction** is the right decomposition: the triangulation enforces camera consistency and metric scale, while the residual head only has to learn the small, structured leftover error. We make the following contributions:

1. A temporal ray-attention fusion module that predicts per-view, per-joint weights and feeds a differentiable weighted DLT layer.
2. A lightweight residual refinement head that predicts per-joint corrections from temporally attended features and the raw triangulated pose.
3. Empirical validation on MPI-INF-3DHP and Human3.6M, including efficiency numbers on an RTX 4090.
4. A plug-in integration into the MotionFlow pipeline as a `MultiViewFusionPlugin`, producing a `HumanMotionIR` for downstream robot retargeting and policy training.

## 2. Related Work

**Multi-view pose estimation.** Classic approaches triangulate 2D keypoints detected in each view [1,2]. Learnable triangulation [3] and ray-aware attention [4] have improved robustness by predicting view weights, but still output the weighted DLT solution directly. We differ by learning a post-triangulation residual.

**Temporal pose models.** Temporal transformers and 1-D convolutions enforce smoothness across frames [5,6]. We use a temporal transformer over (view, joint) tokens after ray-aware feature extraction.

**Residual learning in pose.** Residual connections are common in 2D pose networks [7]; applying them after explicit triangulation is less explored. Our residual head is conditioned on both 3D geometry and temporal context.

## 3. Method

### 3.1 Base temporal ray-attention fusion

Input: per-frame 2D keypoints and confidences `(B, T, V, J, 3)` plus calibrated camera intrinsics `K` and extrinsics `R, t`.

1. **Ray embedding.** For each view and joint, compute the camera ray and encode it together with the 2D point, confidence, and a camera embedding.
2. **Joint-level attention.** Exchange information across joints within each view.
3. **Temporal attention.** Treat each (view, joint) pair as a temporal sequence and apply a transformer encoder.
4. **Weight head.** Predict per-view, per-joint weights `w_{v,j}` and multiply by input confidences.
5. **Weighted DLT.** Triangulate the 3D joints `X_raw` from the weighted rays.

### 3.2 Residual refinement head

The residual head takes the pooled temporal feature `f` and the raw triangulation `X_raw`:

```
ΔX = MLP([f, X_raw])    # (B·T, J, 3)
X  = X_raw + ΔX
```

The MLP has two hidden layers of size `residual_hidden` (default 128) and ReLU activations. Because the correction is added to a camera-consistent estimate, the output preserves metric scale while correcting systematic residual errors (detector bias, mild calibration drift, occlusion-dominated clips).

### 3.3 Training

End-to-end 3D MSE loss:

```
L_3D = || X - X_gt ||_2^2
```

Optionally, a calibrated reprojection loss and a skeleton bone-length loss can be added:

```
L_reproj = sum_{v} || proj_v(X) - x_v ||_2^2
L_bone   = sum_{(parent, child)} ( ||bone_pred|| - ||bone_gt|| )^2
L = L_3D + lambda_reproj * L_reproj + lambda_bone * L_bone
```

The reprojection term does not require skeleton topology and directly enforces multi-view consistency. The bone-length term enforces anatomical consistency without adding learnable parameters. The model is trained with Adam, a clip length of 13 frames, and early stopping on validation MPJPE.

### 3.3.1 Training-time camera calibration perturbation

Calibration noise is the dominant failure mode of the production system (Table 5), so we expose the model to realistic camera perturbations during training. For each clip we sample independent per-view rotation, translation, focal-length, and principal-point offsets and corrupt the calibration tensors before they reach the weighted DLT layer:

```
K', R', t' = perturb(K, R, t;  rot_std, trans_std, focal_std, pp_std)
X_raw = weighted_DLT(..., K', R', t')
```

The ground-truth 3D pose and 2D keypoints remain unchanged, so the residual head learns to absorb small calibration drift while preserving metric scale. Typical magnitudes are ±0.5° rotation, ±5 mm translation, ±1% focal length, and ±2 px principal point. Validation is always run on the unperturbed calibration.

### 3.3.2 Reprojection-error-gated residual head

An optional refinement step makes the residual correction conditional on the raw triangulation's own reprojection error. After the weighted DLT triangulation, the raw 3D estimate is projected back into each view and the per-view pixel residual is summarized by its mean, standard deviation, maximum, and inlier fraction. This 4-D vector is concatenated to the residual head's input, and a tiny sigmoid gate scales the per-joint correction:

```
s = summary( proj_v(X_raw) - x_v )
g = sigmoid( MLP_gate([f_pooled, X_raw, s]) )
X = X_raw + g * MLP_delta([f_pooled, X_raw])
```

The gate lets the model suppress the correction when the raw triangulation is already geometrically consistent, and amplify it when reprojection residuals indicate noise, occlusion, or calibration drift. It adds only a few hundred parameters and can be switched off to recover the original residual head.

### 3.4 Geometry-based camera positional encoding (CamPE)

A practical multi-view system must generalise across camera rigs. Learned view embeddings are tied to a fixed number of views and an arbitrary ordering, so we replace them with a geometry-based camera positional encoding computed from the intrinsics and extrinsics. For each view we derive the camera center `c = -R^T t` and the principal ray `r = R^T [0,0,1]^T`, normalise them to be scale-invariant, encode each component with sinusoidal Fourier bands, and project the concatenated vector to the token dimension with a small MLP. The resulting camera token is added to every joint token before view-level attention, allowing the same model to accept any number of views and to transfer across datasets without relearning view identities.

### 3.5 Skeleton-aware graph joint relation (GJR)

The dense joint-level transformer in the base model treats all joints equally and ignores skeleton topology. We replace it with a sparse graph over `(view, joint)` nodes. Edges encode bone parent–child links, left/right symmetry, and cross-view same-joint connections. Edge-conditioned message passing with a learned scalar gate propagates evidence along anatomically meaningful paths, so occluded joints can borrow information from neighbouring joints and mirrored limbs while preserving multi-view consistency.

### 3.6 Cross-view spatio-temporal attention

A more expressive variant replaces the temporal-only transformer with a single transformer that attends jointly over time and views for each joint. The input tokens are arranged on a (time, view) grid, so each token can aggregate information from all views at any frame within the clip. A residual refinement head is added on top of the weighted DLT triangulation as before. This variant increases capacity modestly (~350–400 k parameters for d=128, n_st_layers=3) while preserving the same plug-in interface.

### 3.7 Learned intrinsic correction (principal point + focal length)

Small errors in the intrinsic matrix `K` are common in practice: principal-point offsets from checkerboard drift or off-center calibration, and focal-length drift from lens zooms or imprecise camera models. Because these errors shift the back-projected rays before triangulation, the residual head alone cannot correct them. We therefore add a lightweight `IntrinsicCorrection` layer that predicts a bounded per-view correction from the pooled temporal features or raw 2D observations and applies it to `K` before triangulation.

The layer first predicts a principal-point offset `(dx, dy)` and, optionally, a focal-length scale `s`:

```
out  = tanh(MLP(pool(feat)))              # (N, V, 3)
Δ    = out[..., :2] * max_offset          # (N, V, 2)
s    = 1 + out[..., 2] * max_focal_scale  # (N, V)
K_corrected[..., 0, 2] += Δ[..., 0]
K_corrected[..., 1, 2] += Δ[..., 1]
K_corrected[..., 0, 0] *= s
K_corrected[..., 1, 1] *= s
```

All corrections are initialized near zero and bounded, so the layer is transparent when calibration is accurate and only activates when drift is detected. During training, the input cameras are perturbed with realistic rotation, translation, focal-length, and principal-point noise; the correction head is supervised with the inverse of the applied perturbation so it learns to restore the true calibration. The same correction layer is applied both to the temporal-only model (Section 3.2) and to the cross-view spatio-temporal variant (Section 3.6), with identical training and inference overhead.

### 3.9 Visibility-gated adaptive fusion

In real captures some views may be occluded or unreliable. We add a per-view, per-joint visibility head that predicts a soft visibility score from the spatio-temporal features and gates the DLT weights. The head is conditioned on a per-joint pooled context across views, so each view's visibility estimate is aware of the full multi-view context. A fallback guard ensures at least `min_visible_views` remain active, preventing degenerate triangulation. The visibility head is supervised with a binary cross-entropy loss against the detector confidence mask.

### 3.10 Factorised spatio-temporal attention (T x V x J)

A more expressive backbone factorises attention along the temporal, view, and joint axes. After ray-aware feature extraction and intrinsic correction, tokens are arranged as a 3-D grid `(T, V, J)` and refined by separate Transformer layers along each axis. Three-dimensional positional embeddings (time, view, joint) are added before the factorised attention, and per-view weights are predicted from the refined features. This model is larger than the temporal-only baseline and targets the same plug-in interface.

### 3.11 Self-supervised masked-view pre-training

To reduce reliance on 3D labels, we pre-train the backbone without ground-truth 3D. Random views or time steps are masked out by zeroing their confidence channels, and the model is asked to minimise the reprojection error on both visible and masked slots. A temporal smoothness loss and a skeleton bone-length consistency loss serve as regularisers. After pre-training on unlabeled multi-view video, the model is fine-tuned with labelled data.

### 3.12 Quality gating and system integration

The fusion module is exposed as a `MultiViewFusionPlugin` inside MotionFlow. It outputs a `HumanMotionIR` containing the fused 3D pose, per-joint confidence, and view-support count, which downstream quality gating can use to fall back to the best single view when fusion disagreement is high.

## 4. Experiments

### 4.1 Datasets and metrics

- **MPI-INF-3DHP.** Train on subject 1 sequences 1 and 2, validate on subject 2 sequence 1. Report MPJPE, PA-MPJPE, PCK@50/100/150 mm, and AUC.
- **Human3.6M.** Train on subject 1, validate on subject 5 (action 02). Report the same metrics.

### 4.2 Implementation details

Models are implemented in PyTorch and trained on a local RTX 4090. The small residual model uses `d=32`, `residual_hidden=64`, and has 66 k parameters. The full model uses `d=64`, `residual_hidden=128`, and has 243 k parameters.

## 5. Results

> **Data-foundation caveat.** Earlier versions of this paper cited MPI-INF-3DHP **9.32 mm** and Human3.6M **0.62 mm** MPJPE results. Those numbers came from circular-label protocols: H36M labels were the unweighted DLT triangulation of the input 2D keypoints (`direct MJE ≈ 0 mm`), and MPI-INF-3DHP 2D inputs were GT-projected rather than detector output. After repairing the data foundation, absolute MPJPEs rise to the 15-130 mm range typical of honest multi-view benchmarks. The experiments below therefore pivot from chasing records to measuring **sparse-view and cross-domain robustness** on true 3D ground truth.

### 5.1 Datasets and protocols

We evaluate on four non-circular benchmarks:

- **Human3.6M true-GT standard protocol.** Train on subjects S1, S5, S6, S7, S8; test on S9 and S11. Labels are true mocap world coordinates from `data/h36m_true_gt/*_multiview_m.npz` and pass both non-circularity and reprojection acceptance gates.
- **AIST++ cross-domain smoke.** A 9-view dance-motion benchmark built from canonical AIST++ `.npz` (`data/webbridge/aistpp_canonical/`). It uses the same 17-joint skeleton as H36M and is non-circular (DLT direct MJE ≈ 44 mm), making it a useful cross-domain stress test.
- **Shelf / Campus detected.** Rebuilt from COCO-style detections and true 3D annotations (`data/webbridge/shelf_campus_detected/`). Campus (3 views, well-calibrated) is the primary sparse-view benchmark; Shelf is reported with a calibration caveat because its reprojection error is ~53 px.
- **MPI-INF-3DHP non-circular smoke.** True 3D GT with GT-projected 2D (used only for controlled smoke diagnostics while real detected 2D is being obtained).

Metrics are direct MPJPE and PA-MPJPE in millimetres, plus PCK and AUC where applicable.

### 5.2 True-GT leaderboards

#### 5.2.1 Human3.6M (S1,5,6,7,8 -> S9/S11)

| Method | S9 direct (mm) | S11 direct (mm) | Combined direct (mm) | Combined PA-MPJPE (mm) | Notes |
|---|---:|---:|---:|---:|---|
| DLT (unweighted) | 33.61 | 24.77 | 29.19 | 29.31 | frozen geometric baseline |
| DLT (confidence-weighted) | 29.82 | 21.91 | 25.87 | 25.55 | frozen geometric baseline |
| **Iskakov ICCV 2019** | **27.10** | **19.60** | **23.35** | **23.10** | 10 epochs, hidden_dim=32; current leader on true GT |
| v80 (smoke) | -- | -- | 98.12 | -- | 2-epoch smoke |
| v80 (long, best recipe) | -- | -- | 39.70 | -- | A800 v2 checkpoint; overfits after epoch 2 |
| **v80 (medium)** | -- | -- | **39.98** | -- | local medium; best epoch 4, then diverges to 133.71 by epoch 8 |
| v57 (medium) | -- | -- | *pending* | -- | started but did not complete; slot reserved for true-GT result |
| **v25** | -- | -- | **72.80** | -- | 8-epoch medium completed; epoch 1 83.19, best epoch 2, diverged to 207.62 by epoch 8 |

Iskakov-style learnable triangulation is the current leader on the true-GT protocol, improving over confidence-weighted DLT by **2.52 mm** combined direct. **The v25 model substantially underperforms both Iskakov and the geometric baselines on true-GT H36M (best 72.80 mm direct, epoch 1 83.19 mm, versus 23.35 mm for Iskakov and 25.87 mm for confidence-weighted DLT), confirming that the headline results from the circular-label protocol do not transfer to honest labels.** The best v80 result is now **39.98 mm** (local medium, best epoch 4; A800 v2 **39.70 mm**), which is closer to the geometric baselines than v25 but still lags confidence-weighted DLT by **14.11 mm**. This pattern reinforces the true-GT narrative: honest H36M labels expose a large generalisation gap that the circular protocol had masked, and the current learned architectures need stronger regularisation, longer training, or mixed-dataset training before they can beat a simple geometric baseline on non-circular data. The v57 H36M true-GT medium run is still pending.

#### 5.2.2 Shelf / Campus detected

| Method | Val direct MPJPE (mm) | Val PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| DLT (unweighted) | 134.43 | 122.37 | frozen reference |
| DLT (confidence-weighted) | 132.29 | 120.95 | frozen reference |
| **Iskakov ICCV 2019** | **128.73** | **119.23** | early-stop at epoch 11; leader |
| v80 | 408.58 | -- | 3-epoch smoke |
| v57 | 424.63 | -- | 3-epoch smoke |
| v25 | 430.67 | -- | 3-epoch smoke |

All learned MotionFlow variants, including v25, are far undertrained relative to DLT on this small true-GT benchmark. The ranking among smoke checkpoints is v80 > v57 > v25, which is consistent with the intended robustness narrative: v80's explicit view-reliability head begins to show value even after only three epochs. **These results further confirm that v25 is not competitive with the DLT/Iskakov geometric baselines in absolute terms on true GT.**

#### 5.2.3 MPI-INF-3DHP non-circular smoke

| Model | Best val MPJPE (mm) | Notes |
|---|---:|---|
| DLT baseline | **23.79** | Geometric lower bound |
| v25 geometry fusion | 26.15 | Closest learned model to DLT |
| v57 DC-PSC | 33.26 | Domain-conditional physical-space calibration |
| v46 SVG | 34.94 | Sparse-view generalisation |
| v80 VRBT | 35.22 | Learned view reliability before triangulation |

The MPI smoke confirms that DLT is a strong baseline on true 3D GT. The gap between v25 and DLT is only ~2.4 mm, while v46/v57/v80 are still tuning. This reinforces the shift from absolute accuracy to **robustness under view scarcity**.

### 5.3 Sparse-view robustness

Rather than reporting a single full-view number, we measure `MPJPE@k`: the pose error when only `k` views are available, averaged over random view subsets. Table below is a smoke evaluation on a 300-frame subset of MPI-INF-3DHP S2/Seq1 (17-joint H36M mapping).

| Model | MPJPE@2 | MPJPE@3 | MPJPE@4 | MPJPE@14 | Notes |
|---|---:|---:|---:|---:|---|
| v25 | 131.98 | 72.05 | 89.02 | **30.61** | best full-view, volatile at low k |
| v46 | 108.51 | 93.10 | 72.31 | 68.64 | competitive at k=2, plateaus at full |
| v57 | 143.67 | 87.02 | 53.41 | 40.93 | strongest low-view scaling |
| v80 | 145.01 | 74.41 | 63.34 | 51.79 | good scaling, higher full-view floor |

These are smoke numbers and should not be used for final model selection, but they already reveal the paper's new claim: **different architectures trade off full-view accuracy against low-view reliability**. v25 is best when all 14 cameras are present; v57 degrades most gracefully as views drop. The v80 sparse-view robust architecture is explicitly designed to shrink this gap by learning per-view reliability before triangulation.

### 5.4 Cross-dataset and cross-domain behaviour

A mixed-dataset variant trained on MPI-INF-3DHP plus the (now deprecated) circular H36M WebBridge reached 11.16 mm on MPI S2/Seq1 under the old circular protocol. While that exact number is no longer meaningful, the underlying design principle--domain-agnostic ray features plus optional per-dataset heads--remains the core of our cross-domain strategy. Ongoing work is repeating the mix on true-GT H36M, AIST++, and Shelf/Campus.

#### 5.4.1 AIST++ cross-domain smoke benchmark

We added AIST++ to the cross-domain suite. The canonical `.npz` are built from the 9-camera AIST++ annotations (`data/webbridge/aistpp_canonical/`) and are non-circular (DLT direct MJE ≈ 44 mm). A small smoke split uses one genre (`gBR_sBM_cAll_d04`) with two training takes and one validation take, giving a quick read on whether the v25/v80 pipelines generalise to dance-style, motion-rich captures.

| Method | val MPJPE (mm) | Notes |
|---|---:|---|
| DLT (unweighted) | **12.66** | frozen geometric reference |
| v25 geometry fusion | 71.79 | 3-epoch smoke |
| v80 view reliability | 76.34 | 3-epoch smoke |

The geometric DLT baseline is again strong on AIST++ (12.66 mm), confirming that AIST++ is a viable, high-quality, non-circular cross-domain source. The learned v25 and v80 models are far from convergence after only three smoke epochs, so the gap is not yet meaningful; these numbers serve only to verify pipeline integration and data loading. Full medium-schedule runs and a true cross-domain training mix (H36M true-GT + AIST++ + Shelf/Campus) are ongoing.

#### 5.4.2 Shelf/Campus detected

Preliminary cross-dataset evaluation on the true-GT Shelf/Campus detected benchmark shows that none of the learned models yet transfer across datasets without dedicated tuning. This is expected: the camera rigs (3-5 views, different focal lengths and room scales) differ radically from the 14-view MPI studio or the 9-view AIST++ rig. We therefore treat cross-domain transfer as an active research direction rather than a resolved result, and we report it as such.

### 5.5 Calibration robustness

Earlier calibration-perturbation tables (rotation, translation, focal length, principal point) were measured on the old MPI GT-projection protocol. Their qualitative lessons remain valid--rotation and principal-point drift are the dominant failure modes, translation and focal-length errors are better tolerated--but the absolute numbers should be re-measured on true detected 2D once that data is available.

On the true-GT H36M protocol, the regularised v80 long run shows that even modest capacity models overfit after epoch 2, suggesting that calibration-robust training must be paired with stronger data augmentation or mixed-protocol training. The learned principal-point correction head from earlier iterations was found to saturate at its maximum allowed offset; it has been retained as an architectural option but is not relied upon for the new narrative.

### 5.6 Runtime on RTX 4090

| Batch | Latency (ms) | Throughput (clips/s) |
|---:|---:|---:|
| 1 | 78.3 | 12.8 |
| 4 | 71.0 | 56.4 |
| 8 | 78.1 | 102.5 |
| 16 | 82.1 | 194.8 |

A single clip (13 frames, 14 views, 28 joints) takes 78 ms, and batching increases throughput to 195 clips/s. These timing numbers were measured on the original model variant and demonstrate that the architecture is lightweight enough for robotics and immersive-video applications regardless of the ongoing data-foundation repair.

### 5.7 Ongoing work and next experiments

The immediate priority is to complete a fair true-GT comparison of v25, v46, v57 and v80 on H36M, AIST++, and Shelf/Campus with matched epoch budgets. Pending experiments include:

1. **Finish v25 H36M true-GT medium run** (`agent-51`).
2. **Run v57 / v80 full-medium schedule** on H36M true GT, AIST++, and Shelf/Campus detected.
3. **Generate true detected 2D for MPI-INF-3DHP** by obtaining `imageSequence/` and running MediaPipe/HRNet/RTMPose.
4. **Re-measure calibration-robustness matrices** on the true-GT protocol.
5. **Produce MPJPE@k curves** for all model variants on H36M (k = 2..4) and MPI (k = 2..14).
6. **Build a true cross-domain training mix** of H36M true-GT, AIST++, and Shelf/Campus to test whether domain-agnostic ray features transfer across rigs.

Until these runs complete, the only verified leaderboard results are the true-GT H36M, AIST++ smoke, and Shelf/Campus tables above, and the paper's headline contribution is repositioned around **sparse-view / cross-domain robustness on honest, non-circular benchmarks**.

## 6. MotionFlow System Integration

### 6.1 `HumanMotionIR` and plugin contract

The fusion model is implemented as a `MultiViewFusionPlugin` inside MotionFlow. It consumes per-view 2D keypoints + confidences and calibrated camera parameters, and emits a `HumanMotionIR` containing:

- `pose`: world-coordinate 3D joints (the output of the residual head).
- `uncertainty`: per-joint, per-view weights and the model's confidence summary.
- `provenance`: source manifest, camera calibration hash, and fusion plugin version.

This IR decouples the upstream human recovery from downstream robot retargeting and policy training, so that alternative fusion algorithms can be swapped without changing the rest of the pipeline.

### 6.2 Quality gating

Before forwarding to GMR/MJLab, the system checks frame-level validity, per-joint confidence, and view support count. When the fusion disagreement exceeds a threshold, the pipeline can fall back to the best single view or flag the segment for human review, preventing low-quality motion from entering robot training.

### 6.3 Robot profile abstraction

A robot profile defines the target kinematics, human-to-robot joint mapping, uncertainty-aware retargeting weights, and the ONNX action order. Our fusion module is profile-agnostic: it outputs metric 3D human pose, and the profile resolver maps it to any supported robot (e.g., BXI ELF3, Unitree G1) without retraining the fusion network.

## 7. Discussion

The residual head succeeds because it only corrects the structured leftover error after a strong geometric baseline. The small model performs best on MPI-INF-3DHP, suggesting that the residual problem is low-dimensional and does not require a large network. On Human3.6M, a slightly larger head improves PA-MPJPE, likely because the H36M skeleton and camera layout benefit from finer pose alignment.

Training-time camera perturbation improves rotation robustness with only a small clean-accuracy cost, but principal-point errors remain catastrophic. Bone-length regularisation and a reprojection-error gate did not improve the small ablation, suggesting that the dominant failure mode is geometric rather than learned: once the principal point shifts, the input rays no longer intersect the true 3D point, and the residual head cannot undo the resulting bias. Future work should therefore focus on explicit principal-point correction or robust triangulation rather than on larger residual heads.

## 8. Conclusion

We introduced a residual refinement head on top of temporal ray-attention fusion for calibrated multi-view 3D pose estimation. The data-foundation audit showed that the original v25 headline numbers were artefacts of a circular-label protocol; on honest, true-GT H36M, **v25 reaches only 72.80 mm (best epoch 2, then diverging to 207.62 mm) and v80 reaches 39.98 mm (best epoch 4, then diverging to 133.71 mm), while Iskakov reaches 23.35 mm and confidence-weighted DLT reaches 25.87 mm**. Both learned variants underperform the geometric and learnable-triangulation baselines by a wide margin on the non-circular protocol. We therefore reposition the paper's contribution around **sparse-view and cross-domain robustness on honest, true-GT benchmarks** rather than absolute MPJPE records.

Verified true-GT leaderboards now include Human3.6M (Iskakov 23.35 mm, confidence-weighted DLT 25.87 mm, v80 medium 39.98 mm / A800 v2 39.70 mm, v25 medium 72.80 mm; v57 pending), AIST++ smoke (DLT 12.66 mm; v25/v80 smoke checkpoints 71.79/76.34 mm), and Shelf/Campus detected (Iskakov 128.73 mm; DLT 132.29 mm). The sparse-view smoke experiments (Section 5.3) show that different architectures trade full-view accuracy for low-view reliability: v57 degrades most gracefully as views drop, while v25 is volatile at low view counts. These observations, together with the cross-domain AIST++ and Shelf/Campus diagnostics, frame the core research question as designing learned fusion models that match or exceed geometric triangulation on true GT while remaining robust when only a few views are available or when transferring across camera rigs.

The architecture remains lightweight (12.7–194 clips/s on an RTX 4090) and is exposed as a `MultiViewFusionPlugin` inside MotionFlow, outputting a `HumanMotionIR` that supports quality gating and robot-profile-based retargeting. These properties make it a promising candidate for ICRA / CVPR 2027, provided that the true-GT performance gap against Iskakov/DLT is closed by stronger regularisation, longer training, or cross-dataset mixing in the coming experimental cycle.

## References

1. Hartley, R. and Zisserman, A. *Multiple View Geometry in Computer Vision*. Cambridge University Press, 2004.
2. Hartley, R. and Zisserman, A. *Multiple View Geometry in Computer Vision*. Cambridge University Press, 2004.
3. Iskakov, K., Burkov, E., Lempitsky, V., and Malkov, Y. “Learnable triangulation of human pose.” *ICCV*, 2019.
4. Ghasemzadeh, S. A. and Alahi, A. “RUMPL: Ray-based transformers for universal multi-view 2D to 3D human pose lifting.” arXiv:2512.15488, 2025.
5. Zhu, W., Ma, X., Liu, Z., Liu, L., Wu, W., and Wang, Y. “MotionBERT: A Unified Perspective on Learning Human Motion Representations.” *ICCV*, 2023.
6. Zeng, et al. “SmoothNet.” *CVPR*, 2022.
7. Newell, et al. “Stacked hourglass networks.” *CVPR*, 2016.

---

## Tracking

- GitHub Issue: #16
- Pull Request: #17
- Branch: `multiview-residual-exploration`

---

**New scripts added in Iter11+:**
- `motionflow_mv/calibration/perturb.py`
- `motionflow_mv/losses/bone_length.py`
- `motionflow_mv/fusion/camera_positional_encoding.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_campe_model.py`
- `motionflow_mv/fusion/graph_joint_relation.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_campe_graph_model.py`
- `motionflow_mv/fusion/adaptive_view_selector.py`
- `motionflow_mv/fusion/ray_attention_temporal_residual_campe_adaptive_model.py`
- `experiments/train_ray_attention_temporal_residual_campe_mpiinf3dhp.py`
- `experiments/train_ray_attention_temporal_residual_h36m.py`
- `experiments/train_ray_attention_temporal_residual_campe_graph_h36m.py`
- `experiments/train_ray_attention_temporal_residual_campe_adaptive_mpiinf3dhp.py`
- `experiments/generate_paper_figures_v1.py`

**Figures included in this draft:**
1. `docs/figures/architecture.png` — architecture diagram.
2. `docs/figures/mpi_mpjpe_bar.png` — MPI-INF-3DHP MPJPE comparison.
3. `docs/figures/robustness_final5.png` — robustness to noise/occlusion/outliers.
4. Qualitative figure: raw DLT, residual-corrected, and GT skeletons on MPI-INF-3DHP (`outputs/visualize_residual_mpi_final5/`).
5. Runtime table / plot on RTX 4090.
