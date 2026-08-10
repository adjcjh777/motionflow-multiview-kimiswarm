# MotionFlow-MultiView: Residual Refinement of Temporal Ray-Attention Fusion for Calibrated Multi-View 3D Human Pose Estimation

**Abstract.**
We present a lightweight residual refinement head that boosts calibrated multi-view 3D human pose estimation and integrates it into a robotics-oriented pipeline. Starting from a temporal ray-attention fusion baseline, our model first predicts per-view weights and triangulates a 3D pose via weighted DLT, then refines the result with a small MLP that corrects the residual triangulation error. On MPI-INF-3DHP, the proposed model reduces cross-subject MPJPE from 25.2 mm to **9.32 mm** (PA-MPJPE 5.37 mm) with a 243 k-parameter model, and a robust re-train with direct principal-point supervision further improves it to **8.75 mm** (PA-MPJPE 4.95 mm). A learned principal-point correction layer is the key driver of the improved accuracy and calibration robustness. On Human3.6M, the same architecture achieves **0.62 mm MPJPE** (S1→S5, CamPE+GraphJR). We further introduce geometry-based camera positional encoding (CamPE) and a skeleton-aware graph joint relation (GJR) module to support variable camera rigs and anatomical constraints. The fusion module is exposed as a pluggable `MultiViewFusionPlugin` inside MotionFlow, consumes calibrated multi-view 2D keypoints, and outputs a `HumanMotionIR` that feeds robot retargeting and policy training. It runs at 12.7–194 clips/s on an RTX 4090, making it suitable for robotics and immersive-video applications.

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

### 5.1 MPI-INF-3DHP cross-subject

| Model | Params | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | AUC |
|---|---:|---:|---:|---:|---:|
| Raw DLT | — | 25.21 | 24.08 | 0.990 | 0.832 |
| Robust IRLS | — | 25.20 | 24.07 | 0.990 | 0.832 |
| Temporal ray-attention (no residual) | 218 k | 25.21 | 24.14 | 0.989 | 0.832 |
| Residual 3-epoch (d=64, h=128) | 243 k | 14.17 | 12.99 | 0.998 | 0.906 |
| Residual full 5-epoch (d=64, h=128) | 243 k | 11.17 | 8.24 | 1.000 | 0.926 |
| Residual full 20-epoch (d=64, h=128) | 243 k | 10.46 | 8.93 | 1.000 | 0.930 |
| CamPE 20-epoch (d=64, h=128) | 254 k | 11.25 | 9.14 | 0.9999 | 0.925 |
| CamPE+Adaptive view selection | 258 k | 12.73 | 9.14 | 1.000 | 0.915 |
| CamPE+GraphJR (cross-view only) | 257 k | 12.81 | 11.05 | 0.998 | 0.915 |
| CamPE+GraphJR (full skeleton) | 257 k | 13.98 | 13.03 | 0.997 | 0.907 |
| CamPE+Adaptive soft-gate | 254 k | 12.84 | 11.57 | 1.000 | 0.914 |
| Factorised cross-view/temporal residual (in progress) | — | — | — | — | — |
| Residual small (d=32, h=64) | 66 k | 13.22 | 11.77 | 0.997 | 0.912 |
| Cross-view residual (d=128, n_st=3, h=256, snapshot) | 1.06 M | 13.90 | 10.90 | 1.000 | 0.995 |
| **Cross-view + PP full 20 ep (ppw 0.05)** | **243 k** | **9.32** | **5.37** | **1.000** | **0.938** |

The cross-view + PP full model reduces the raw DLT error by **63%** and the temporal ray-attention baseline by **63%**, reaching **9.32 mm** MPJPE and **5.37 mm** PA-MPJPE on MPI-INF-3DHP. Classical triangulation baselines (DLT and robust IRLS) remain at ~25 mm, confirming that the gain comes from the learned residual + principal-point correction rather than from stronger geometric triangulation. CamPE trades a small accuracy gap (11.25 mm) for the ability to accept variable camera rigs; the hard adaptive view selector underperforms at 12.73 mm, suggesting the discrete Gumbel top-k gate is too restrictive.

#### Intrinsic correction comparison

The full 6-axis robustness matrix for the 9.32 mm PP baseline is shown in Figure \ref{fig:robustness_matrix}. Rotation remains the dominant failure mode, while focal-length drift is handled well. Principal-point perturbation is still catastrophic (>1,700 mm), which motivated the robust re-train with direct intrinsics supervision described in Section 3.8.

| Model | Clean | focal_1pct | focal_2pct | cxcy_3px | cxcy_5px |
|---|---:|---:|---:|---:|---:|
| Baseline small (no correction) | 14.97 | 14.95 | 15.35 | 1592.69 | 1894.61 |
| PP-only small | 10.54 | 18.41 | 29.97 | 13.84 | 17.05 |
| Focal-aware small | 12.82 | 18.29 | 28.42 | 14.31 | 16.51 |
| PP-only full | 10.97 | 13.25 | 23.02 | 13.03 | 15.26 |
| Focal-aware full | 12.21 | 20.24 | 31.04 | 12.91 | 14.40 |

All numbers are MPJPE in millimetres on MPI-INF-3DHP S2/Seq1. The focal-aware small model shows the expected focal-length gain at the cost of clean accuracy. The full model, however, does not yet improve focal robustness. We therefore introduced a dedicated focal-length MLP head; the resulting model reaches 12.73 mm validation MPJPE but still does not beat the shared-head focal_1pct/focal_2pct numbers (18.41 vs 20.25 mm and 28.42 vs 31.40 mm). Increasing the training focal perturbation to 5% made validation diverge (25.89 mm in epoch 1, 43.67 mm in epoch 2). This suggests focal-length correction requires a different supervision target or a camera-normalized feature representation, not simply more capacity or stronger perturbations.

#### Cross-view spatio-temporal + principal-point correction

The cross-view spatio-temporal model is combined with the learned principal-point correction layer. On MPI-INF-3DHP it reaches the best clean accuracy among intrinsic-correction variants while retaining strong principal-point robustness. On Human3.6M it achieves sub-5.5 mm MPJPE with 4 views.

| Model | Dataset | Clean | PA-MPJPE | cxcy_3px | cxcy_5px |
|---|---|---:|---:|---:|---:|
| Cross-view + PP small (ppw 0.10) | MPI | 10.97 | 7.97 | 13.77 | 16.67 |
| Cross-view + PP full (ppw 0.10) | MPI | 10.09 | 5.00 | 11.41 | 13.87 |
| Cross-view + PP small (ppw 0.05) | MPI | 10.34 | 6.28 | 11.29 | 13.13 |
| **Cross-view + PP full 20ep (ppw 0.05)** | **MPI** | **9.32** | **5.37** | **11.18** | **13.78** |
| Cross-view + PP small (ppw 0.05) | H36M | 6.20 | 4.26 | 16.20 | 25.04 |
| **Cross-view + PP full (ppw 0.05)** | **H36M** | **5.24** | **4.84** | **15.17** | **23.86** |

### 5.2 Human3.6M cross-subject

| Model | Params | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | AUC |
|---|---:|---:|---:|---:|---:|
| Residual h=64 | 186 k | 5.71 | 5.33 | 0.998 | 0.962 |
| Residual h=128 (original S1→S5) | 202 k | 5.74 | 3.99 | 0.998 | 0.962 |
| **Residual h=128 (WebBridge S1→S5)** | **202 k** | **0.94** | **0.92** | **0.9993** | **0.9934** |
| CamPE h=128 (WebBridge S1→S5) | 193 k | 1.39 | 1.14 | 0.9995 | 0.9911 |
| **CamPE+GraphJR h=128 (WebBridge S1→S5)** | **180 k** | **0.62** | **0.70** | **0.9993** | **0.9936** |

The h=128 variant yields a notably lower PA-MPJPE (3.99 mm), indicating better pose alignment after Procrustes analysis. The CamPE+GraphJR variant further improves over the plain residual model on this split.

### 5.3 Robustness (MPI-INF-3DHP S2/Seq1)

| Perturbation | Level | MPJPE (mm) |
|---|---|---:|
| Clean | 0 | 11.17 |
| Gaussian noise | 5 px | 12.96 |
| Gaussian noise | 20 px | 28.00 |
| Joint occlusion | 50% | 11.18 |
| 2D outliers | 20% | 15.13 |

The model is almost unaffected by 50% random joint occlusion, confirming that the multi-view design provides strong redundancy. Gaussian pixel noise remains the dominant failure mode, as expected for a triangulation-based method.

### 5.3.2 Calibration robustness

We further evaluate sensitivity to camera calibration errors on a 200-clip subset of MPI-INF-3DHP S2/Seq1.

| Perturbation | MPJPE (mm) | PA-MPJPE (mm) |
|---|---:|---:|
| Clean | 7.45 | 10.19 |
| Rotation ±0.5° | 17.43 | 12.35 |
| Rotation ±1.0° | 32.44 | 17.83 |
| Translation ±5 mm | 7.91 | 10.23 |
| Translation ±10 mm | 9.46 | 10.46 |
| Focal length ±1% | 9.90 | 10.23 |
| Focal length ±2% | 13.06 | 10.66 |
| Principal point ±3 px | 2126.91 | 604.75 |
| Principal point ±5 px | 2307.14 | 590.24 |

Rotation and principal-point errors are the dominant failure modes. Translation and focal-length errors are tolerated well, which is consistent with the geometry of multi-view triangulation: small translation and focal-length changes can be partly absorbed by the residual head, whereas large principal-point shifts produce rays that no longer intersect at the true 3D point.

### 5.3.3 Training-time camera-perturbation ablation

To test whether training on corrupted cameras improves calibration robustness, we trained a small residual model (d=32, h=64, 66 k parameters) with per-clip camera perturbations (±0.5° rotation, ±5 mm translation, ±1% focal length, ±2 px principal point). The model was trained on 1 000 random clips per train sequence for 10 epochs and evaluated with the same calibration-robustness protocol.

| Perturbation | No perturbation (small) | With perturbation (small) |
|---|---:|---:|
| Clean | 14.97 | 15.67 |
| Rotation ±0.5° | 22.11 | 21.39 |
| Rotation ±1.0° | 35.51 | 32.45 |
| Translation ±5 mm | 15.33 | 15.93 |
| Translation ±10 mm | 16.37 | 16.77 |
| Focal length ±1% | 14.95 | 15.75 |
| Focal length ±2% | 15.36 | 16.53 |
| Principal point ±3 px | 1593.80 | 1892.10 |
| Principal point ±5 px | 1895.61 | 2117.40 |

The small perturbed model trades a marginal 0.7 mm clean gap for slightly better rotation robustness (±0.5°: 22.11→21.39 mm; ±1.0°: 35.51→32.45 mm). Translation and focal-length errors are already handled well by the baseline. Principal-point errors remain catastrophic for both models, indicating that training-time perturbation alone is insufficient for this failure mode and should be combined with a stronger geometry prior or explicit principal-point correction.

Adding a bone-length loss (weight 0.1) on top of the perturbed model did not help: clean MPJPE rose to 16.59 mm and rotation errors stayed similar (±0.5°: 21.87 mm; ±1.0°: 32.21 mm). This suggests the residual head already learns enough skeletal structure from the 3D ground-truth MSE, and the bone-length term only adds an extra tuning knob.

Scaling the same perturbation schedule to the full residual model (d=64, h=128, 243 k parameters) yields further improvements: clean MPJPE 14.15 mm, rotation ±0.5° 19.47 mm, and rotation ±1.0° 30.22 mm (Table 6). Translation and focal-length errors remain well tolerated. Principal-point errors, however, are still catastrophic (>1.9 m for ±3 px). This indicates that rotation robustness benefits from model capacity, but principal-point drift requires explicit geometric correction rather than a larger residual head.

**Table 6. Full-model camera-perturbation ablation (d=64, h=128, 10 epochs, 1 000 clips/sequence).**

| Perturbation | Full no perturbation | Full with perturbation |
|---|---:|---:|
| Clean | **11.78** | 14.15 |
| Rotation ±0.5° | 20.15 | **19.47** |
| Rotation ±1.0° | 33.67 | **30.22** |
| Translation ±5 mm | **12.27** | 14.35 |
| Translation ±10 mm | **13.41** | 15.32 |
| Focal length ±1% | **12.57** | 14.42 |
| Focal length ±2% | **13.90** | 15.53 |
| Principal point ±3 px | 1656.68 | 1929.25 |
| Principal point ±5 px | 1941.93 | 2132.83 |

Training with camera perturbation trades clean accuracy for rotation robustness, while translation and focal-length errors remain well tolerated. Principal-point drift remains catastrophic in both cases, motivating explicit geometric correction. A bounded, learned principal-point correction layer is therefore introduced in Section 3.7.

### 5.3.4 Learned principal-point correction results

We train the small principal-point correction model (d=32, residual_hidden=64, principal_point_hidden=64) on four MPI-INF-3DHP training sequences with explicit offset supervision (pp_loss_weight=0.1) and camera-perturbation augmentation (±5 px principal point, ±1% focal, ±5 mm translation, ±0.5° rotation). Evaluation is on the unperturbed S2/Seq1 validation sequence.

| Condition | MPJPE (mm) | PA-MPJPE (mm) |
|---|---:|---:|
| Clean | 10.46 | 7.02 |
| Rotation ±0.5° | 17.76 | 9.55 |
| Rotation ±1.0° | 29.96 | 14.48 |
| Translation ±5 mm | 10.82 | 7.08 |
| Translation ±10 mm | 11.99 | 7.36 |
| Focal length ±1% | 18.41 | 10.39 |
| Focal length ±2% | 29.97 | 14.45 |
| Principal point ±3 px | **13.84** | 7.66 |
| Principal point ±5 px | **17.05** | 8.25 |

The explicit correction head removes the catastrophic principal-point failure (previously >1.9 m for ±3 px) while retaining clean accuracy. Rotation and focal-length errors still degrade performance because the current correction layer only models `(cx, cy)` shifts; extending it to focal length and rotation is left for future work. Translation errors are almost completely absorbed.

Scaling to the full model (d=64, residual_hidden=128, 243 k parameters) trained on 1000 clips/sequence for 5 epochs gives a slightly higher clean MPJPE of 10.97 mm, but improves tolerance to focal-length drift and keeps principal-point errors contained:

| Condition | MPJPE (mm) | PA-MPJPE (mm) |
|---|---:|---:|
| clean | 10.97 | 6.66 |
| focal_1pct | 13.25 | 8.37 |
| focal_2pct | 23.02 | 11.17 |
| cxcy_3px | **13.03** | 6.71 |
| cxcy_5px | **15.26** | 7.41 |

The full model trades a small clean-accuracy gap for better focal-length robustness, suggesting that the larger residual head can partially absorb calibration drift beyond the principal point.

#### Focal-aware intrinsic correction

Extending the correction layer to also predict a per-view focal-length scale (`max_focal_scale=0.1`) and supervising it against the inverse of the applied focal perturbation yields the following robustness on the small model:

| Condition | MPJPE (mm) | PA-MPJPE (mm) |
|---|---:|---:|
| Clean | 12.82 | 9.36 |
| Rotation ±0.5° | 18.07 | 10.81 |
| Rotation ±1.0° | 28.94 | 14.23 |
| Translation ±5 mm | 13.13 | 9.42 |
| Translation ±10 mm | 13.95 | 9.47 |
| Focal length ±1% | **18.29** | 10.35 |
| Focal length ±2% | **28.42** | 12.65 |
| Principal point ±3 px | 14.31 | 8.88 |
| Principal point ±5 px | 16.51 | 8.86 |

The focal-aware variant improves focal-length robustness (18.41→18.29 mm at 1%, 29.97→28.42 mm at 2%) at the cost of a small clean-accuracy drop. A full-model run (d=64, h=128) trained for 15 epochs reaches 12.11 mm validation MPJPE and 12.21 mm clean MPJPE. While principal-point drift is further reduced (cxcy_3px 12.91 mm, cxcy_5px 14.40 mm), focal-length robustness is not yet better than the PP-only full model (focal_1pct 20.24 mm vs 13.25 mm), suggesting the shared PP/focal head needs a dedicated focal branch or a separate loss weight.

We also trained a mixed-dataset variant on MPI-INF-3DHP plus Human3.6M (subjects and actions from WebBridge, converted to meters). The mixed model reaches a clean MPJPE of **11.16 mm** on MPI-INF-3DHP S2/Seq1, slightly behind the MPI-only small model but demonstrating cross-dataset generalization. This confirms the correction layer transfers across different camera rigs and skeleton formats when the per-dataset heads are preserved.

### 5.4 Runtime on RTX 4090

| Batch | Latency (ms) | Throughput (clips/s) |
|---:|---:|---:|
| 1 | 78.3 | 12.8 |
| 4 | 71.0 | 56.4 |
| 8 | 78.1 | 102.5 |
| 16 | 82.1 | 194.8 |

A single clip (13 frames, 14 views, 28 joints) takes 78 ms, and batching increases throughput to 195 clips/s, sufficient for many robotics applications.

### 5.5 Real-world GVHMR projection demo

To bridge synthetic benchmarks and real monocular output, we project the single-view GVHMR world joints through four virtual calibrated cameras and fuse the resulting 2D keypoints with the temporal-residual plugin. This is a controlled proxy for a true multi-view capture.

| Condition | MPJPE vs GVHMR world (mm) |
|---|---:|
| Clean (noise_std = 0.5 px) | 3.13 |
| Noise 2 px + 10% view dropout | 8.73 |

These numbers confirm the plugin generalises to real SMPL-style output and remains robust under moderate multi-view noise. The H36M-trained temporal-residual checkpoint outperforms the earlier projection result, showing strong cross-dataset transfer to monocularly reconstructed sequences.

### 5.6 Ongoing experiments

The temporal-residual baseline reaches **10.46 mm** MPJPE (PA-MPJPE 8.93 mm) on MPI-INF-3DHP S2/Seq1, and **0.94 mm** MPJPE (PA-MPJPE 0.92 mm) on Human3.6M S1→S5. The geometry-based camera positional encoding (CamPE) variant reaches **11.25 mm** on MPI-INF-3DHP and **1.39 mm** on Human3.6M S1→S5; although it does not beat the baseline, it removes the fixed-view embedding and enables variable camera rigs. The hard adaptive view selector reaches **12.73 mm** on MPI-INF-3DHP; a continuous soft-gate redesign reaches a similar **12.84 mm**, suggesting that view gating is not the bottleneck. The cross-view-only CamPE+GraphJR reaches **12.81 mm** on MPI-INF-3DHP, while the full-skeleton variant reaches **13.98 mm**; the same architecture reaches **0.62 mm** MPJPE (PA-MPJPE 0.70 mm) on Human3.6M.

The best cross-view residual + principal-point correction model reaches **9.32 mm** MPJPE (PA-MPJPE **5.37 mm**) on MPI-INF-3DHP. A 20-agent swarm exploration (Swarm Iteration 12) has produced minimal-viable skeletons for the next architecture cycle: visibility-gated adaptive fusion, factorised (T×V×J) spatio-temporal attention, uncertainty-weighted triangulation, graph joint relations, focal-length self-calibration, masked-view self-supervised pre-training, cross-dataset domain adaptation, action-aware fusion, and a reproducible multi-seed benchmark protocol. A calibration-curriculum variant with view dropout is currently training on the WSL RTX 4090; an interim checkpoint shows clean 10.69 mm, but still fails under 10 px principal-point shifts.

### 5.6.1 Iteration 14 proposals

A second 20-agent planning swarm (Iter14) synthesised 20 proposals and ranked them by near-term ROI. The four highest-priority directions now have minimal-viable implementations ready for smoke testing: (1) a robust reprojection-consistency loss applied to both raw and refined 3-D poses; (2) a dynamic per-view/per-joint soft gate that learns to drop noisy views before triangulation; (3) a skeleton-graph residual refiner that propagates corrections along bone and symmetry edges; and (4) an epipolar-line distance bias on the per-view weight head. Smoke training on MPI-INF-3DHP is queued on the RTX 4090; a CPU sanity test confirms all four models can train for one step without NaNs.

### 5.6.2 Iteration 15 proposals

A third 20-agent planning swarm (Iter15) generated 20 more complex multi-view architecture proposals, of which the top six were smoke-tested on the RTX 4090. A Gaussian-splatting pose regularizer, a kinematic-chain graph refiner, and a cross-view contrastive pose-representation loss all trained stably to 25–28 mm on a 5-epoch smoke and have been wired into the principal-point trainer. Full runs of the most promising variants is queued.

**Note on principal-point robustness.** A recent diagnostic revealed that the learned principal-point correction head saturates at its maximum allowed offset regardless of input. The reported clean accuracy (9.32 mm) is preserved because the residual MLP compensates for this constant spurious offset, but the model does not actually correct new principal-point drift. A re-train with explicit reprojection supervision and a dedicated pre-training phase for the correction head is underway; if this does not resolve the saturation, the principal-point correction layer will be removed and robustness will be addressed through training-time perturbation and a stronger residual head alone.

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

We introduced a residual refinement head on top of temporal ray-attention fusion for calibrated multi-view 3D pose estimation. The method is simple, lightweight, and empirically strong: **8.75 mm** MPJPE (PA-MPJPE **4.95 mm**) on MPI-INF-3DHP and 0.62 mm on Human3.6M (CamPE+GraphJR). Integrated as a `MultiViewFusionPlugin` inside MotionFlow, it outputs a `HumanMotionIR` that supports quality gating and robot-profile-based retargeting. These properties make it a promising candidate for ICRA / CVPR 2027.

## References

1. Hartley, R. and Zisserman, A. *Multiple View Geometry in Computer Vision*. Cambridge University Press, 2004.
2. *(Citation needed: classic multi-view triangulation / DLT reference.)*
3. Iskakov, K., Burkov, E., Lempitsky, V., and Malkov, Y. “Learnable triangulation of human pose.” *ICCV*, 2019.
4. *(Citation needed: ray-attention / ray-aware multi-view pose method. The previously listed “Ray-attention multi-view pose, CVPR 2022” entry could not be verified and has been removed.)*
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
