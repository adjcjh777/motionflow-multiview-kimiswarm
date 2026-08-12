# Phase 1 20-Agent Exploration Synthesis

## Scope

The 20-agent parallel exploration (Phase 1, Iteration 2) surveyed 2024–2026 multi-view human pose/mesh estimation, the existing MotionFlow IR/fusion stack, datasets, training strategies, evaluation protocols, and integration paths. This document synthesizes the top findings, the most promising directions, and the key risks that should shape the next design.

---

## Top Findings

1. **DLT is the dominant baseline on the current data.** On Shelf frames 300–600 (5 views), confidence-weighted DLT achieves a mean/median reprojection error of ~9.9 / 5.5 px. Every lightweight learned variant—`RobustTriangulationModel`, `ResidualRefinerModel`, `TemporalRefinerModel`, and `AttentionFusionModelV2`—only matches or marginally improves it. A much larger temporal model (A800-D, hidden=256, d=128, window=15) still fails to beat DLT.

2. **Reprojection-only supervision is the bottleneck.** DLT already minimizes a closely related geometric objective, so models trained only on 2D reprojection loss cannot easily outperform it. This is the single most consistent finding across the swarm and the repository's own Iterations 5–9.

3. **3D ground truth is required for a clear learned win.** The literature (e.g., Iskakov et al., ICCV 2019; MVGFormer, CVPR 2024; MV-SSM, CVPR 2025) and the project's own experiments agree that supervised 3D losses, bone-length priors, or strong motion priors are needed to surpass triangulation.

4. **The existing stack is modular and ready.** `HumanMotionIR` is a clean integration contract. `motionflow_mv/fusion/` already contains DLT, attention fusion, robust triangulation, residual refinement, and temporal refinement. `MultiViewPipeline` triangulates per-frame 2D keypoints and can be wrapped by an IR-level adapter.

5. **Per-view estimator quality matters as much as fusion.** Several agents identify **ScoreHMR** (CVPR 2024) as the strongest MIT-licensed per-view SMPL/SMPL-X estimator, while **GVHMR** (SIGGRAPH Asia 2024) remains the existing upstream because it already produces world-grounded SMPL. **HeatFormer** (CVPR 2025) is flagged as a future multi-view SMPL refiner.

6. **Temporal and physical priors are the next levers.** Temporal refiners give only marginal gains under reprojection loss, but methods that add bone-length, symmetry, and temporal smoothness constraints—such as Bragagnolo et al. (ECCVW 2024) and COMETH (arXiv 2025)—are the natural next step once 3D labels are available.

7. **Uncertainty should be a first-class IR citizen.** The fusion modules already compute per-view weights, attention scores, and reprojection residuals, but these are discarded. Standardizing and storing them in `HumanMotionIR.uncertainty` enables downstream robot retargeting to reason about confidence.

---

## Most Promising Directions

### Methods

- **ScoreHMR** (CVPR 2024): MIT-licensed, diffusion-based SMPL/SMPL-X recovery from single images, video, or multiple uncalibrated views. Best near-term upgrade for the per-view estimator.
- **MVGFormer** (CVPR 2024): interleaves learning-free geometry with learnable appearance modules. Directly inspires upgrading `AttentionFusionModelV2` from a shallow camera-embedding model to a geometry-aware transformer.
- **RapidPoseTriangulation** (arXiv 2025): fast whole-body multi-person triangulation in milliseconds. Useful as a multi-person baseline and for engineering speed comparisons.
- **Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation** (ECCVW 2024): fuses monocular 3D skeletons with reprojection and limb-length symmetry constraints. Directly applicable to the residual/temporal refiner loss terms.
- **MV-SSM** (CVPR 2025): replaces full cross-view attention with state-space scanning, potentially generalizing better to new camera layouts.
- **GNN skeleton/view graphs**: several agents suggest replacing or augmenting `ViewAttentionFusion` with graph message passing over views and/or the human skeleton topology to enforce anatomical and occlusion consistency.

### Datasets

- **Human3.6M**: the most practical large-scale source of real 3D GT for supervised 3D training. Requires registration.
- **CMU Panoptic**: excellent for multi-person and temporal evaluation; research-only license.
- **Shelf / Campus**: small, already loaded via `VoxelPoseShelfLoader`, ideal for fast reprojection sanity checks but too small for training large models from scratch.
- **AMASS**: mocap-only source for realistic synthetic pre-training; can be rendered through virtual calibrated rigs to inject motion priors.
- **3DPW**: in-the-wild validation with SMPL ground truth.

The swarm's dataset agent was asked to find WebBridge-accessible sources, but no WebBridge-specific data feed was identified. The practical path is direct dataset registration/download (Human3.6M, Panoptic) and synthetic generation (AMASS).

### Architectural Directions

1. **Extend `HumanMotionIR` for multi-view provenance** with optional per-view 2D/3D observations, camera parameters, and uncertainty fields while keeping the downstream interface stable.
2. **Add a `MultiViewFusion` adapter** that turns per-view `HumanMotionIR`s into a single fused IR, reusing the existing GVHMR adapter for SMPL fields.
3. **Keep DLT as the default geometric backend**, and expose learned variants as plugin fusion heads behind a common interface.
4. **Switch training to 3D-supervised losses** (MPJPE + reprojection + bone-length + temporal) once real 3D GT is available.
5. **Instrument uncertainty**: store per-view weights, reprojection residuals, and per-joint standard deviations in the IR.

---

## Key Risks and Blockers

- **3D ground-truth access.** Without real 3D labels, learned fusion is unlikely to beat DLT. Human3.6M and Panoptic require registration and preprocessing.
- **Reprojection-only ceiling.** Continuing to train on reprojection or DLT pseudo-GT will likely keep tying DLT.
- **License constraints.** EasyMocap and CMU Panoptic are non-commercial. ScoreHMR is MIT; GVHMR's license should be verified before redistribution.
- **Calibration sensitivity.** DLT and the current fusion modules assume accurate intrinsics/extrinsics. In-the-wild capture will need robust calibration (COLMAP, DUSt3R/MASt3R) or a calibration-free fallback.
- **Cross-view person matching.** `select_best_person_group` is combinatorial and only validated for single-person scenes. Multi-person scenes need appearance-based matching or epipolar/Hungarian methods.
- **Per-view estimator cost.** ScoreHMR diffusion inference is heavier than GVHMR. Batch processing or A800-D offloading may be needed.
- **Domain gap.** Synthetic pre-training on random skeletons did not transfer to Shelf. Future synthetic data must use realistic motion (AMASS) and camera/noise distributions.

---

## Bottom Line

The most credible next move is not to build a bigger reprojection-only model, but to **acquire real 3D GT and train a lightweight, geometry-aware fusion head with 3D + bone-length + reprojection losses**, while keeping DLT as the strong baseline and default. The repository already has the right modular skeleton; Phase 2 should extend the IR, add a multi-view adapter, and validate on Human3.6M with proper 3D metrics.
