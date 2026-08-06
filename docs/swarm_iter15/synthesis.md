# Swarm Iter-15 Synthesis: 20-Agent Proposal Review

Synthesis of the 20 proposals in `docs/swarm_iter15/`. No training was run; this document ranks the ideas by near-term feasibility and expected ROI for an ICRA/CVPR 2027 submission.

---

## 1. Proposal Inventory

| # | Proposal File | Title | Key Idea (1 sentence) |
|---|-------------|-------|----------------------|
| 1 | `proposal_unified-multidataset-canonical-skeleton.md` | Unified Multi-Dataset Canonical Skeleton Prior | Replace the dense per-joint residual MLP in the PP anchor with a dataset-conditional graph residual refiner on a shared skeleton graph, regularized by a canonical bone-length consistency loss. |
| 2 | `proposal_cross-view-contrastive-pose-representation.md` | Cross-View Contrastive Pose Representation | Add an auxiliary cross-view joint-level contrastive loss on per-joint spatio-temporal features to make the multi-view representation more view-invariant and physically consistent. |
| 3 | `proposal_adaptive-temporal-window-pyramid.md` | Adaptive Temporal Window Pyramid | Replace the single fixed-length spatio-temporal attention block with a learned adaptive temporal-window pyramid (short/medium/global windows) that selects scales per joint and view. |
| 4 | `proposal_epipolar-geometry-transformer-bias-v2.md` | Epipolar-Geometry Transformer Bias v2 | Inject calibrated multi-view epipolar geometry as a relative-position bias inside the spatio-temporal transformer, with a learnable sigmoid gate to blend the bias. |
| 5 | `proposal_semantic-action-conditional-fusion.md` | Semantic Action-Conditional Fusion | Condition the spatio-temporal transformer and residual refinement on a discrete action label via additive embedding, per-layer FiLM, and an action-aware residual head. |
| 6 | `proposal_multiperson-multiview-association-graph.md` | Multi-Person Multi-View Association Graph | Add a lightweight `(view, person, joint)` association graph between the ST transformer and triangulation to jointly fuse multi-view video of multiple people. |
| 7 | `proposal_lightweight-realtime-multiview-fusion.md` | Lightweight Real-Time Multi-View Fusion | Combine a closed-form epipolar-line-distance bias with a lightweight per-view/per-joint dynamic selection gate to improve cross-view robustness within real-time runtime constraints. |
| 8 | `proposal_neural-implicit-3d-pose-field.md` | Neural Implicit 3D Pose Field | Replace the dense residual MLP with a joint-conditioned neural implicit 3D pose field that refines DLT output by walking along the field gradient toward the zero level-set. |
| 9 | `proposal_uncertainty-aware-multiview-triangulation.md` | Uncertainty-Aware Multi-View Triangulation | Predict anisotropic 2D image-space covariances per view/joint and use them to drive covariance-conditioned adaptive Gauss-Newton refinement with an epipolar consistency loss. |
| 10 | `proposal_hierarchical-view-temporal-joint-attention.md` | Hierarchical View→Temporal→Skeleton-Joint Attention | Replace the flat ST transformer with a coarse-to-fine hierarchy: camera-group view attention, then temporal attention, then skeleton-graph joint attention. |
| 11 | `proposal_failure-driven-hard-negative-mining.md` | Failure-Driven Hard Negative Mining | Add online hard-negative mining (OHEM + FIFO memory bank) and a synthetic hard-negative generator that corrupts the most confident views. |
| 12 | `proposal_gaussian-splatting-pose-regularizer.md` | Gaussian-Splatting Pose Regularizer | Add a differentiable 3D Gaussian-splatting regularizer that projects predicted joints with learned per-joint anisotropic covariance back into each calibrated view and penalizes 2D keypoint negative log-likelihood. |
| 13 | `proposal_physics-informed-skeleton-dynamics-prior.md` | Physics-Informed Skeleton Dynamics Prior | Add a lightweight physics-informed skeleton dynamics prior (temporal smoothness, bone-length ratios, soft ground-contact, COM stability) plus a small bidirectional GRU temporal refiner. |
| 14 | `proposal_learnable-camera-centric-coordinate-transform.md` | Learnable Camera-Centric Coordinate Transform | Replace the fixed camera-to-world mapping with a learned per-view residual SE(3)+scale transform conditioned on deep features and applied after principal-point correction. |
| 15 | `proposal_kinematic-chain-graph-convolutional-refiner.md` | Kinematic-Chain Graph Convolutional Refiner | Add a final skeleton-aware kinematic-chain graph convolutional refiner on the triangulated 3-D skeleton, trained with a kinematic-chain regularization loss. |
| 16 | `proposal_camera-parameter-conditioned-fusion.md` | Camera-Parameter-Conditioned Fusion | Inject calibrated camera intrinsics and extrinsics into the view-selection weight head and 3D residual refinement head to down-weight geometrically inconsistent views. |
| 17 | `proposal_self-supervised-masked-view-completion.md` | Self-Supervised Masked-View Completion | Add a masked-view 2D completion head trained only on reprojection error of masked-out views/timesteps to force view-invariant 3D skeleton learning. |
| 18 | `proposal_multi-scale-cross-view-spatial-pyramid.md` | Multi-Scale Cross-View Spatial Pyramid (MSCVSP) | Add a multi-scale spatial pyramid inside the per-frame cross-view encoder so attention operates over fine joint and coarse limb/torso scales. |
| 19 | `proposal_differentiable-multiview-bundle-adjustment.md` | Differentiable Multi-View Bundle Adjustment | Add a lightweight differentiable structure-only bundle-adjustment (Gauss-Newton/LM) refinement on top of the learned triangulation to minimize reprojection error. |
| 20 | `proposal_occlusion-robust-visibility-transformer.md` | Occlusion-Robust Visibility Transformer | Replace the per-view MLP visibility gate with a small geometry-aware cross-view visibility transformer that reasons jointly over views, joints, and camera rays. |

---

## 2. Tier Ranking

### Tier 1: Run Next — Minimal Code, High ROI

These changes either reuse the existing 9.32 mm anchor directly or add a tiny module/loss, and can be smoke-tested in ≤5 epochs on a single RTX 4090.

| # | Proposal | Rationale |
|---|----------|-----------|
| 12 | Gaussian-Splatting Pose Regularizer | Adds a tiny per-joint covariance head and a closed-form NLL reprojection loss to the existing PP trainer; no change to the attention/triangulation path. Smoke: 5 epochs, ~20–30 min. |
| 7 | Lightweight Real-Time Multi-View Fusion | Recombines two existing ideas (epipolar-line bias + dynamic view-selection gate) into a single model subclass; mostly wiring existing modules. Smoke: 5 epochs, ~15–25 min. |
| 15 | Kinematic-Chain Graph Convolutional Refiner | Single post-triangulation graph refiner + small regularization loss; operates on 3D output and preserves the anchor backbone. Smoke: 5 epochs, ~15–30 min. |
| 2 | Cross-View Contrastive Pose Representation | Drop-in auxiliary loss and projection head; no change to architecture or inference. Smoke: 5 epochs, ~30–45 min. |
| 11 | Failure-Driven Hard Negative Mining | Mostly a training-script change (OHEM + small synthetic generator hook); small model subclass. Smoke: 1–5 epochs, ~20–30 min for a 1-epoch smoke. |
| 16 | Camera-Parameter-Conditioned Fusion | Replaces the weight and residual heads with camera-conditioned variants; moderate but well-scoped and directly attacks calibration robustness. Smoke: 5 epochs, ~25–35 min per epoch, so plan ~2 h for a full 5-epoch smoke. |

### Tier 2: Queue After Tier 1 — New Module, Low/Medium Risk

These add a modest new module or architecture branch but stay close to the existing anchor or its visibility/uncertainty variants.

| # | Proposal | Rationale |
|---|----------|-----------|
| 20 | Occlusion-Robust Visibility Transformer | Builds on the existing visibility-gated PP model; replaces a single MLP with a small transformer, reuses the same synthetic occlusion pipeline. Smoke: 5 epochs, ~45–60 min. |
| 4 | Epipolar-Geometry Transformer Bias v2 | New transformer encoder layer with epipolar relative-position bias; v1 already exists, so this is an incremental extension. Smoke: 5 epochs, ~45–60 min. |
| 18 | Multi-Scale Cross-View Spatial Pyramid | New per-frame cross-view encoder module; minimal change to the rest of the pipeline and an existing ablation template exists. Smoke: 2–5 epochs, ~30–45 min. |
| 8 | Neural Implicit 3D Pose Field | Replaces the residual MLP with an implicit field + Newton refiner; moderate complexity but localized to the residual path. Smoke: 5 epochs, ~20–30 min. |
| 9 | Uncertainty-Aware Multi-View Triangulation | Adds a covariance head and adaptive Gauss-Newton refinement; moderate complexity and some numerical risk, but a natural extension of the anchor. Smoke: 5 epochs, ~10–15 min. |
| 13 | Physics-Informed Skeleton Dynamics Prior | Adds a physics loss + small bidirectional GRU refiner; low/medium risk but the auxiliary loss can dominate if not carefully weighted. Smoke: 5 epochs, ~30–45 min. |

### Tier 3: High Risk / High Reward — Major Architecture or Data Work

These require significant new architecture, data engineering, or training pipelines. Run only after Tier 1 and Tier 2 have delivered stable baselines and free GPU time.

| # | Proposal | Rationale |
|---|----------|-----------|
| 3 | Adaptive Temporal Window Pyramid | Replaces the core ST transformer with an adaptive multi-scale window pyramid; large memory/clip-handling implications and must prove the gate actually adapts. Smoke: 5 epochs, ~30–45 min. |
| 10 | Hierarchical View→Temporal→Skeleton-Joint Attention | Replaces the flat ST transformer with a three-stage hierarchy; major architecture change, must beat the 9.32 mm anchor to justify complexity. Smoke: 5 epochs, ~20–30 min. |
| 19 | Differentiable Multi-View Bundle Adjustment | Differentiable Gauss-Newton/LM optimization layer; high implementation risk (NaN/instability) but high reward for calibration robustness. Smoke: 3–5 epochs, ~20–35 min. |
| 1 | Unified Multi-Dataset Canonical Skeleton Prior | Requires mixed-dataset loader integration and canonical skeleton mapping across 17/28 joints; major data-engineering component. Smoke: 5 epochs, ~15–25 min once data wiring is ready. |
| 14 | Learnable Camera-Centric Coordinate Transform | Learns residual SE(3)+scale extrinsics; risk of degenerate camera configurations and scale collapse. Smoke: 5 epochs, ~45–90 min. |
| 5 | Semantic Action-Conditional Fusion | Requires action labels, action-aware dataloader, and FiLM integration; risk of overfitting to action classes and no labels at inference. Smoke: 5 epochs, ~20–30 min. |

### Tier 4: Paper / Eval Infrastructure

This batch is architecture-heavy, so Tier 4 contains the proposals whose main deliverable is a new capability, evaluation protocol, or two-stage training pipeline rather than a direct 9.32 mm improvement. They should run concurrently with RTX 4090 smokes on CPU or A800-D read-only where possible.

| # | Proposal | Rationale |
|---|----------|-----------|
| 6 | Multi-Person Multi-View Association Graph | Synthetic multi-person data generation, association-accuracy metrics, and new multi-person eval protocol are the dominant work; the model itself is a graph module. Can run synthetic P=1 backward-compatibility eval on CPU/A800 after a checkpoint exists. |
| 17 | Self-Supervised Masked-View Completion | SSL pre-training infrastructure (masked completion head + two-stage pre-train/fine-tune pipeline); checkpoint generation is GPU, but downstream eval can be CPU/A800 read-only once the model is fine-tuned. |

---

## 3. Recommended RTX 4090 GPU Queue Order

Single-GPU queue, respecting dependencies and smoke turnaround time. Estimated total active RTX 4090 time to clear all smokes serially: ~8–14 hours; many Tier 1/Tier 2 smokes are short enough to batch across a day.

1. **Tier 1 block (parallelize in order of shortest runtime):**
   1. `Lightweight Real-Time Multi-View Fusion` — ~15–25 min
   2. `Kinematic-Chain Graph Convolutional Refiner` — ~15–30 min
   3. `Gaussian-Splatting Pose Regularizer` — ~20–30 min
   4. `Failure-Driven Hard Negative Mining` — ~20–30 min (1-epoch smoke)
   5. `Cross-View Contrastive Pose Representation` — ~30–45 min
   6. `Camera-Parameter-Conditioned Fusion` — ~2–3 h for 5 epochs
2. **Tier 2 block:**
   1. `Uncertainty-Aware Multi-View Triangulation`
   2. `Multi-Scale Cross-View Spatial Pyramid`
   3. `Neural Implicit 3D Pose Field`
   4. `Physics-Informed Skeleton Dynamics Prior`
   5. `Epipolar-Geometry Transformer Bias v2`
   6. `Occlusion-Robust Visibility Transformer`
3. **Tier 3 block (after Tier 1/Tier 2 checkpoints exist):**
   1. `Hierarchical View→Temporal→Skeleton-Joint Attention`
   2. `Adaptive Temporal Window Pyramid`
   3. `Differentiable Multi-View Bundle Adjustment`
   4. `Learnable Camera-Centric Coordinate Transform`
   5. `Semantic Action-Conditional Fusion`
   6. `Unified Multi-Dataset Canonical Skeleton Prior`
4. **Tier 4 block (can overlap with Tier 1/Tier 2 training, mostly CPU/A800):**
   1. `Self-Supervised Masked-View Completion` pre-training smoke
   2. `Multi-Person Multi-View Association Graph` synthetic data generation and P=1 eval

---

## 4. CPU / A800-D Read-Only Recommendations

The following work can be done on CPU or A800-D in read-only mode because they rely only on existing checkpoints or do not require gradient validation on the local RTX 4090:

- **`proposal_gaussian-splatting-pose-regularizer.md`**: After the RTX 4090 smoke produces a checkpoint, clean/robustness evaluation and covariance-diagnostics can run on A800-D read-only.
- **`proposal_cross-view-contrastive-pose-representation.md`**: Inference and robustness eval on the trained checkpoint are read-only; the CPU unit test `tests/test_crossview_pose_contrast.py` can run on CPU.
- **`proposal_kinematic-chain-graph-convolutional-refiner.md`**: CPU sanity of the refiner and eval on a saved checkpoint are read-only; the kinematic-chain loss is lightweight.
- **`proposal_lightweight-realtime-multiview-fusion.md`**: Runtime benchmarking and eval on existing checkpoints can run on CPU/A800 read-only.
- **`proposal_camera-parameter-conditioned-fusion.md`**: Eval on the saved checkpoint and ablation of raw-2D vs deep-feature conditioning are read-only.
- **`proposal_multiperson-multiview-association-graph.md`**: Generating synthetic P=2 validation clips, computing association accuracy, and P=1 backward-compatibility eval can run on CPU/A800 read-only once a checkpoint is available.
- **`proposal_self-supervised-masked-view-completion.md`**: Generating frozen anchor outputs and computing masked-view reprojection metrics are CPU/A800 friendly; only the SSL pre-training smoke needs an RTX 4090.

All other proposals require a training smoke on the local RTX 4090 first because they introduce new model parameters, loss terms, or optimization layers that must be validated with gradients before CPU/A800 integration.

---

## 5. Self-Evolution Loop Mapping

For every proposal, the loop is:

1. **Reflect:** Review the 9.32 mm anchor (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`) and the iter14 robustness matrix (`docs/results_iter14.md`, `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py`) to identify the weakest axis (calibration robustness, occlusion, temporal coherence, multi-view association, or evaluation depth).
2. **Hypothesize:** Propose a minimal, architecture-preserving change (new loss, small head, new module, or eval harness) that addresses the weak axis while reusing the anchor backbone.
3. **Smoke-validate:** Run ≤5 epochs on a tiny subset (typically 500 samples, `d=32`, `residual_hidden=64`) on the RTX 4090; check for NaNs, finite val MPJPE, and that the change does not regress the anchor by more than the proposal's stated threshold.
4. **Integrate:** If the smoke passes, register the new model/loss in `motionflow_mv/fusion/__init__.py` or `motionflow_mv/losses/__init__.py`, add the eval script to the paper harness, and queue the full run; if it fails, apply the fallback and either re-smoke or archive the proposal.

---

## 6. Bottom-Line Recommendation

Start with the **Tier 1** block: `Lightweight Real-Time Multi-View Fusion`, `Kinematic-Chain Graph Convolutional Refiner`, `Gaussian-Splatting Pose Regularizer`, `Cross-View Contrastive Pose Representation`, and `Failure-Driven Hard Negative Mining`. These five are the closest to the 9.32 mm anchor, add the fewest lines of code, and are most likely to improve or preserve accuracy while generating paper-ready ablations. Queue `Camera-Parameter-Conditioned Fusion` right after if the first five leave GPU time.

Run the **Tier 4** infrastructure items (`Multi-Person Multi-View Association Graph` synthetic-data protocol, `Self-Supervised Masked-View Completion` pre-training smoke) on CPU/A800-D read-only or during RTX 4090 downtime so that new evaluation protocols and pre-trained weights are ready before the high-risk Tier 3 experiments finish.

Defer **Tier 3** high-risk items until at least two Tier 2 architectures have produced stable checkpoints, because several Tier 3 ideas (adaptive windows, hierarchy, bundle adjustment, camera-centric transforms) depend on a validated base architecture and can destabilize if rushed.
