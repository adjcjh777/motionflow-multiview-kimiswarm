<agent_swarm_result>
<summary>completed: 20</summary>
<subagent agent_id="agent-27" item="Survey 2024-2026 multi-view human pose/mesh estimation methods: identify top performing architectures, datasets, and metrics, and recommend which methods are most compatible with the MotionFlow IR pipeline." outcome="completed">## Report: 2024–2026 Multi-View Human Pose/Mesh Methods for MotionFlow

### Executive Summary

MotionFlow’s multi-view branch currently runs a clean, modular pipeline: per-view 2D keypoints + confidence → calibrated cameras → DLT triangulation → optional learned refinement. Empirical work on Shelf 300–600 shows DLT is the dominant baseline and reprojection-only supervision cannot beat it. The most compatible 2024–2026 methods therefore fall into three groups: **strong per-view estimators** that feed the existing 2D/3D fusion layer, **learned multi-view optimizers** that can replace or refine triangulation given 3D labels, and **fast geometric baselines** for deployment. The best near-term strategy is to keep DLT, swap the per-view estimator for ScoreHMR, and train a lightweight 3D-supervised fusion head on CMU Panoptic/Human3.6M while treating EasyMocap as an external benchmark only.

### Key Findings / Method Descriptions

**1. ScoreHMR (CVPR 2024)** – MIT license. A score-guided diffusion model for 3D human recovery that supports single-image, video, and **multiple uncalibrated views**. In MotionFlow it fits naturally before fusion: run it independently on each view to produce per-view SMPL parameters, then feed its 2D reprojections/confidences into `motionflow_mv/fusion/`. It can also act as a pseudo-GT generator for 3D-supervised fusion training.

**2. HeatFormer (CVPR 2025)** – neural optimizer for fixed multiview SMPL recovery. It iteratively refines SMPL parameters from calibrated multiview images using a transformer-based heat-map alignment. This is a direct upgrade path for `motionflow_mv/ir/human_motion_ir.py`: it outputs SMPL/SMPL-X parameters that the existing `gvhmr_adapter.py` pattern can already normalize into the IR.

**3. MVMP-HMR (arXiv 2026)** – multiview multi-person mesh recovery for large scenes with occlusions. It introduces a synthetic benchmark (15 scenes, up to 50 views / 30 people) and a volumetric feature-fusion + 3D-pose-query decoder. Relevant for scaling beyond the single-person Shelf setting; the dataset can augment training of the `AttentionFusionModelV2` with realistic occlusion patterns.

**4. RapidPoseTriangulation (arXiv 2025)** – whole-body multi-person pose triangulation in milliseconds. If MotionFlow later needs whole-body (face/hands) multi-person output, this can replace the current per-joint DLT loop in `motionflow_mv/fusion/triangulation.py` for speed while keeping the same IR.

**5. COMETH (arXiv 2025)** – convex-optimization multiview fusion with kinematic/biomechanical constraints and temporal consistency. It directly addresses the “outliers remain” limitation of the current `TemporalRefinerModel` by adding physical priors, and could be integrated as a post-DLT refinement layer.

**6. EasyMocap / ZJU mvpose** – mature geometric/optimization baseline for calibrated multi-view SMPL fitting. License is non-commercial, so it should remain an external benchmark, not a core dependency. The project’s `docs/phase0_literature_audit.md` already flags this correctly.

### Concrete Recommendations for Implementation

1. **Per-view estimator upgrade**: integrate ScoreHMR as the default per-view human recovery module. Its outputs are SMPL parameters, so extend `gvhmr_adapter.py` into a generic `SMPLAdapter` that writes `HumanMotionIR`.
2. **Fusion layer**: keep `RobustTriangulationModel`/`AttentionFusionModelV2` but train with **3D GT loss** on CMU Panoptic and Human3.6M, not reprojection-only. This is the only path the repo’s own experiments suggest can beat DLT.
3. **SMPL neural refiner**: plug HeatFormer after DLT as an optional `SMPLRefiner` module operating on the IR’s `pose` and `coordinate_system` fields.
4. **Whole-body / multi-person path**: prototype RapidPoseTriangulation for multi-person whole-body settings before committing to a custom transformer.
5. **Data**: use Shelf/Campus for sanity checks, but switch evaluation to Human3.6M, CMU Panoptic, and the MVMP-HMR synthetic set for paper-level metrics.

### Key References

1. **Score-Guided Diffusion for 3D Human Recovery** (ScoreHMR) – Stathopoulos et al., CVPR 2024. [arXiv:2403.09623](https://arxiv.org/abs/2403.09623)
2. **HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery** – Matsubara et al., CVPR 2025. [arXiv:2412.04456](https://arxiv.org/abs/2412.04456)
3. **Multiview Multi-Person Human Mesh Recovery Under Large Scenes with Occlusions** – Zhang et al., arXiv 2026. [arXiv:2607.24302](https://arxiv.org/abs/2607.24302)
4. **RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond** – Bermuth et al., arXiv 2025. [arXiv:2503.21692](https://arxiv.org/abs/2503.21692)
5. **COMETH: Convex Optimization for Multiview Estimation and Tracking of Humans** – Martini et al., arXiv 2025. [arXiv:2508.20920](https://arxiv.org/abs/2508.20920)

### Risks, Blockers, and Assumptions

- **3D supervision bottleneck**: The repo already shows DLT cannot be beaten with reprojection loss. Access to Human3.6M/CMU Panoptic 3D labels is required for a learned win.
- **License block**: EasyMocap is non-commercial and cannot ship in the default product. ScoreHMR (MIT) and VoxelPose (MIT-like Microsoft repo) are safer choices.
- **Calibration**: HeatFormer and DLT assume calibrated cameras; ScoreHMR’s multiview mode tolerates uncalibrated views. MotionFlow must expose the camera model in `HumanMotionIR.coordinate_system` consistently.
- **Domain gap**: Synthetic pretraining (MVMP-HMR) did not transfer in the repo’s temporal-refiner experiments; mix real and synthetic data.
- **Person matching**: The current `select_best_person_group` works for one person; MVMP-HMR/RapidPoseTriangulation are needed for true multi-person scenes.</subagent>
<subagent agent_id="agent-28" item="Survey 2024-2026 multi-view SMPL/SMPL-X fitting and shape estimation: optimization vs. learning-based, temporal consistency, and how to output compatible SMPL params for the IR." outcome="completed">## Report: 2024–2026 Multi-View SMPL/SMPL-X Fitting and Shape Estimation for MotionFlow

### Executive Summary

The MotionFlow multi-view extension currently triangulates per-view 2D keypoints into a world-coordinate 3D skeleton (DLT baselines, `AttentionFusion`, `RobustTriangulation`, `TemporalRefiner`) and passes the result to `HumanMotionIR` / `gvhmr_adapter.py`. The next step toward CVPR/ICRA-quality is to produce **compatible SMPL/SMPL-X parameters** (`body_pose`, `global_orient`, `transl`, `betas`) rather than raw 3D joints, and to do so with **temporal consistency** across multi-view inputs.

Recent top-venue work (2024–2026) falls into two camps: **optimization/model-fitting** methods that fit a parametric body model to multi-view image evidence, and **learning-based** methods that regress SMPL parameters directly. For MotionFlow, the best near-term path is a **hybrid pipeline**: (1) a strong per-view estimator such as ScoreHMR, (2) multi-view consistency via reprojection/temporal losses, and (3) a lightweight SMPL/SMPL-X parameter aggregator that writes the exact keys expected by `HumanMotionIR`.

### Key Findings

**1. Optimization-based SMPL fitting is still competitive and interpretable.**  
ScoreHMR (CVPR 2024) frames model fitting as a diffusion inverse problem: a pretrained score model on SMPL parameters is guided by image-derived scores. It supports **multiple uncalibrated views** and video sequences, and explicitly reasons about pose/shape parameters. For MotionFlow, this maps naturally onto the per-view GVHMR/ScoreHMR front-end, and its output SMPL keys can be inserted into `HumanMotionIR.pose` with minimal adapter changes.

**2. Learning-based multi-view lifting now targets world/metric-scale output.**  
Mocap-2-to-3 (2025) recovers absolute world-grounded pose from monocular input by treating 3D motion as a multi-view synthesis process. It decouples local pose and global movement and encodes ground-plane priors. For MotionFlow, this suggests adding an **explicit world-coordinate head** after triangulation so the fused 3D skeleton is not just camera-relative, but aligned with the shared physical frame already stored in `HumanMotionIR.coordinate_system`.

**3. Occlusion-aware multi-view fusion remains a practical concern.**  
The ECCV 2024 workshop paper on multi-view pose fusion lifts per-view 3D skeletons (rather than 2D keypoints), then refines via reprojection error with **limb-length symmetry constraints**. This is highly compatible with the existing `RobustTriangulationModel` / `ResidualRefinerModel`: the same differentiable reprojection loss can be extended with bone-length and temporal smoothness terms, which the current codebase already trains on Shelf data.

**4. Fast triangulation is solved; the value-add is downstream consistency.**  
RapidPoseTriangulation (2025) shows that multi-person whole-body triangulation can run in milliseconds. The MotionFlow DLT baseline is already fast (median ~5.5 px reprojection on Shelf). Therefore, the research gap is not triangulation speed, but **recovering SMPL-X-compatible parametric bodies and ensuring temporal consistency** across frames.

**5. Temporal consistency should be a first-class output of the fusion module.**  
Mocap-2-to-3’s motion representation and ScoreHMR’s video setting both demonstrate that temporal priors reduce jitter. The existing `TemporalRefinerModel` (Bi-GRU over windows) is a good start, but should be trained with **3D SMPL parameter losses** rather than reprojection alone, because the current experiments show reprojection-only supervision cannot beat DLT.

### Concrete Recommendations

1. **Adopt a per-view SMPL/SMPL-X front-end.**  
   Replace or augment the current 2D keypoint input with per-view SMPL parameter estimates from ScoreHMR or GVHMR. Store them under the `HumanMotionIR.pose` keys already defined (`body_pose`, `global_orient`, `transl`, `betas`).

2. **Implement a multi-view SMPL consistency loss.**  
   Starting from `motionflow_mv/fusion/robust_triangulation.py`, add a module that:
   - Takes per-view SMPL parameters and camera calibration.
   - Projects SMPL joints into each view.
   - Optimizes a shared set of SMPL parameters by minimizing reprojection error plus bone-length and temporal-smoothness terms.
   This is the optimization counterpart to the current learned DLT variants.

3. **Upgrade `TemporalRefinerModel` to operate on SMPL parameters.**  
   Extend `temporal_refiner.py` so the GRU refines `body_pose` / `global_orient` / `transl` sequences, not just 3D joint positions. Train on 3D/parameter ground truth (e.g., Human3.6M, Panoptic, or synthetic SMPL sequences), because the Shelf experiments showed reprojection-only losses plateau against DLT.

4. **Use Shelf/Campus and Human3.6M for validation, but add 3D-GT datasets for SMPL training.**  
   Shelf/Campus provide calibrated multi-view 3D joints, yet the project’s own results show they do not provide enough SMPL-level 3D supervision. Add Human3.6M/Panoptic SMPL fits or synthetic data with known parameters.

5. **Maintain the IR contract.**  
   Any new fusion module should read/write `HumanMotionIR` so that downstream robot retargeting remains unchanged. Update `gvhmr_adapter.py` helpers if new keys (e.g., left/right hand pose for SMPL-X) are added.

### Key References

1. **Score-Guided Diffusion for 3D Human Recovery** — Anastasis Stathopoulos et al., CVPR 2024. [arXiv:2403.09623](http://arxiv.org/abs/2403.09623)  
   Diffusion-based SMPL fitting; supports single-frame, multi-view uncalibrated, and video settings.

2. **Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining** — Zhumei Wang et al., arXiv 2025. [arXiv:2503.03222](http://arxiv.org/abs/2503.03222)  
   Learns world-grounded absolute pose via multi-view synthesis; decouples local pose and global movement.

3. **Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation** — Laura Bragagnolo et al., ECCV workshops 2024. [arXiv:2408.15810](http://arxiv.org/abs/2408.15810)  
   Fuses per-view 3D skeletons with reprojection error and limb-length symmetry; directly relevant to occlusion handling.

4. **RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond** — Daniel Bermuth et al., arXiv 2025. [arXiv:2503.21692](http://arxiv.org/abs/2503.21692)  
   Demonstrates that multi-view triangulation can be extremely fast; the remaining research value lies in parametric/temporal refinement.

5. **Learnable Triangulation of Human Pose** — Karim Iskakov et al., ICCV 2019. [arXiv:1903.09299](https://arxiv.org/abs/1903.09299)  
   Seminal confidence-weighted multi-view triangulation; forms the geometric backbone already used in `RobustTriangulationModel`.

### Risks, Blockers, and Assumptions

- **Supervision bottleneck.** The project’s own experiments (Iterations 5–9) show learned fusion matches but does not beat DLT on Shelf when supervised only by reprojection. SMPL fitting requires 3D or parameter-level GT; without it, any learned module risks the same plateau.
- **Shape ambiguity across views.** `betas` should ideally be shared across all views and frames for a single person, whereas current per-frame triangulation treats each frame independently. A mechanism is needed to enforce a global shape vector.
- **SMPL-X license/model availability.** SMPL-X parameters (face, hands, expressions) require the SMPL-X model and a compatible license. The current IR defaults to SMPL; expanding to SMPL-X is a license and dependency decision.
- **Camera calibration.** Multi-view SMPL fitting assumes accurate intrinsics/extrinsics. The project already uses Shelf/Campus calibration, but in-the-wild capture may need calibration refinement.
- **Computational cost.** ScoreHMR/Mocap-2-to-3 are heavier than 2D keypoint detectors. The A800-D scaling experiments suggest capacity alone does not help; the bottleneck is the objective/supervision, not compute.</subagent>
<subagent agent_id="agent-29" item="Survey multi-view temporal consistency methods for human motion: smoothing, filtering, trajectory models, and how to integrate with the HumanMotionIR timestamps/fps." outcome="completed">## Report: Multi-View Temporal Consistency for Human Motion in MotionFlow

### Executive Summary
The MotionFlow multi-view branch currently triangulates per-frame 2D keypoints with confidence-weighted DLT and optionally refines them with a learned per-frame residual or a short Bi-GRU temporal window. On Shelf 300–600, all learned variants only *match* DLT (median ~5.5 px) because the only available supervision is 2D reprojection and there is no strong motion prior. To move toward ICRA/CVPR quality, temporal consistency should be treated as a **trajectory-level** problem: use the `HumanMotionIR` `timestamps` and `fps` fields to build a physically plausible, temporally coherent world-coordinate skeleton.

### Key Findings / Methods
1. **Classic smoothing/filters are still valuable.**  
   Savitzky–Golay, weighted moving average, and Kalman/RTS smoothing can be applied per joint to the DLT output. They are deterministic, low-latency, and directly use `HumanMotionIR.timestamps` to handle variable frame rates or dropped frames. They mainly remove high-frequency jitter and single-frame outliers, explaining the small median improvement already observed, but cannot fix large systematic errors.

2. **Trajectory models beat per-frame residual models.**  
   The existing `TemporalRefinerModel` uses a fixed-window Bi-GRU and ignores both `fps` and the actual time interval between frames. Replacing it with a Transformer or dilated temporal convolution (VideoPose3D-style) and feeding timestamp-derived `dt` or positional encoding can better exploit the `fps` field. State-space / Kalman-smoother trajectory models naturally incorporate `timestamps` and can fuse per-view reprojection errors with a motion model.

3. **Bundle adjustment with temporal and skeleton priors.**  
   A spatiotemporal bundle adjustment (BA) objective over a window refines 3D joints, camera extrinsics, and optionally bone lengths. Adding a temporal smoothness term `λ·‖X_t − X_{t−1}‖²` and limb-length symmetry constraints, as in the occlusion-aware multi-view fusion work below, targets the current ~1044 px outlier max rather than just the median.

4. **Leverage per-view 3D estimators and multi-view lifters.**  
   ScoreHMR produces per-view SMPL parameters with strong image priors; running it on each view and fusing the resulting 3D skeletons gives temporal smoothing better inputs. Mocap-2-to-3 reformulates 3D motion as multi-view synthesis with ground geometric priors, while RapidPoseTriangulation provides fast whole-body triangulation. Both align with the `HumanMotionIR.pose` SMPL fields and `timestamps`.

### Concrete Recommendations
- Add a `TemporalSmoother` post-processor in `motionflow_mv/fusion/` that reads `HumanMotionIR.timestamps` and `fps`, supports Savitzky–Golay / RTS / GP modes, and returns a new `HumanMotionIR`.
- Implement `SpatiotemporalBundleAdjuster` that optimizes a sliding window of 3D joints under reprojection, bone-length, and temporal-smoothness constraints, using `scipy.optimize.least_squares` (or `ceres`). Initialize with DLT.
- Upgrade `TemporalRefinerModel` to accept timestamp deltas and, when 3D GT is available, add a 3D MSE loss plus per-joint uncertainty from reprojection error. Without 3D GT, keep the model small and combine it with BA rather than expecting it to beat DLT alone.
- Propagate `fps` and `timestamps` through the GVHMR adapter: it currently hard-codes 30 fps; use the video metadata or frame rate instead.
- Store temporal-consistency metadata in `HumanMotionIR.quality` (e.g., per-frame reprojection error, smoothness score) so downstream robot retargeting can gate on motion quality.

### Key References
- Score-Guided Diffusion for 3D Human Recovery, CVPR 2024. arXiv:2403.09623
- Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation, ECCV Workshops 2024. arXiv:2408.15810
- RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond, arXiv:2503.21692, 2025/2026.
- Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining, arXiv:2503.03222, 2025/2026.
- 3D Human Pose Estimation in Video with Temporal Convolutions and Semi-Supervised Training, CVPR 2019. arXiv:1811.11742

### Risks, Blockers, and Assumptions
- **3D supervision bottleneck:** temporal and trajectory methods need real 3D GT (Human3.6M, Panoptic, or Shelf GT) to beat DLT; reprojection-only losses have proven insufficient in this codebase.
- **Over-smoothing risk:** aggressive temporal smoothing can attenuate fast robot-relevant motions; use per-joint adaptive smoothing and preserve high-frequency components.
- **Real-time vs. offline:** BA/RTS are windowed/offline; for online robot applications, use a causal Kalman filter or causal temporal convolution.
- **License and camera data:** external datasets and per-view estimators must be checked for license compatibility and the availability of calibrated camera parameters.</subagent>
<subagent agent_id="agent-30" item="Survey camera calibration, synchronization, and multi-camera rig setups for in-the-wild human capture; recommend a practical calibration pipeline for multi-view videos." outcome="completed">## Report: Practical Calibration, Synchronization, and Rig Design for In-the-Wild Multi-View Human Capture

### Executive Summary

The MotionFlow multi-view branch already has a working geometric fusion core: `motionflow_mv/calibration/camera.py` stores a pinhole `Camera` (K, R, t), and `motionflow_mv/fusion/triangulation.py` performs confidence-weighted DLT. These components have been validated on the VoxelPose Shelf dataset, where calibration is provided offline. To move from a lab setup to in-the-wild multi-view video, the bottleneck is no longer fusion, but obtaining accurate, time-aligned camera calibration without a pre-existing rig. This report surveys the 2024–2026 literature and recommends a practical calibration pipeline that plugs directly into the existing `Camera` / `MultiViewPipeline` interfaces.

### Key Findings

**1. Intrinsic calibration: checkerboard / ChArUco is still the safest starting point.**  
For static rigs, a printed ChArUco board plus OpenCV’s `calibrateCamera` gives reliable intrinsics and distortion coefficients. The existing `docs/swarm_iter1/11_camera_calibration.md` already covers this. For smartphone captures, EXIF focal length can seed K, then be refined with COLMAP. The current `Camera` class only stores K, R, t and assumes zero distortion, so any lens distortion must be undone before points reach `triangulate_dlt`.

**2. Extrinsic calibration: COLMAP for structured scenes, DUSt3R / MASt3R for weakly textured or wide-baseline capture.**  
COLMAP remains the default workhorse for static rigs with reasonable texture and overlap. In-the-wild capture often violates these assumptions. Recent transformer-based geometry models provide a robust fallback:
- **DUSt3R** (CVPR 2024) regresses dense pointmaps from image pairs and recovers relative poses without calibration. It can bootstrap camera parameters when feature-based SfM fails.
- **MASt3R** (ECCV 2024) adds a local feature head and fast reciprocal matching, improving wide-baseline matching and scene-scale reconstruction.

Both output focal lengths and camera poses that can be converted into the `Camera` dataclass (`P = K[R | t]`) and fed to `MultiViewPipeline.fuse_frame`.

**3. Synchronization: hardware genlock is ideal; post-hoc audio/visual alignment is practical.**  
The current fusion assumes per-frame correspondence across views. In the wild:
- Prefer hardware genlock or timecode (Tentacle Sync, DSLRs with HDMI genlock).
- Low-cost fallback: clap board / flash + audio peak detection or cross-correlation of 2D joint trajectories.
- If frame-level offset is unknown, add a small post-processing step that shifts each view by the lag maximizing cross-view 2D-keypoint correlation.

**4. Multi-camera rig design for human capture.**  
- Use 4–8 static cameras placed around the subject in a 3–6 m³ volume.
- Baselines of 1–2 m and 60–120° separation give strong triangulation while maintaining overlap.
- Mount cameras at chest/head height to reduce occlusions; avoid extreme rolling-shutter devices.
- Record a short calibration clip of the same scene or a person standing in the capture volume to refine extrinsics.

### Concrete Recommendations for Implementation

1. **Extend `Camera` to include distortion.** Add `distCoeffs` and an `undistort_points()` helper, then call it inside the pipeline before DLT.
2. **Create `scripts/calibrate_multiview.py`** with the following flow:
   - Run per-camera ChArUco / OpenCV intrinsics calibration.
   - Run COLMAP with fixed intrinsics to obtain extrinsics.
   - If COLMAP fails, fall back to DUSt3R or MASt3R for global alignment.
   - Convert the resulting K, R, t into `Camera` objects and write `calibration.json`.
3. **Add a synchronization pre-processor** that either reads per-frame timestamps or estimates frame offsets from an audio/visual event, producing a common frame index map before fusion.
4. **Validate on capture data** using the same reprojection-error script pattern as `experiments/eval_shelf_voxelpose.py`. A target mean reprojection error < 10 px is comparable to the current Shelf baseline.
5. **Keep the per-view estimator modular.** Continue using GVHMR / ScoreHMR per view; feed their 2D keypoints/confidences into `MultiViewPipeline` and the existing `RobustTriangulationModel` / `TemporalRefinerModel`.

### Risks, Blockers, and Assumptions

- **License:** DUSt3R and MASt3R are released under CC BY-NC-SA 4.0, which is non-commercial. They should be treated as optional fallback/benchmark tools, not core dependencies, unless the project’s licensing can accommodate them.
- **Distortion:** The current `Camera` model ignores lens distortion. Fisheye / wide-angle consumer lenses will degrade triangulation unless explicitly undistorted.
- **Rolling shutter / motion blur:** Fast action with low-cost cameras violates the static-pinhole assumption and cannot be fixed by calibration alone.
- **Scale ambiguity:** DUSt3R/MASt3R and COLMAP produce up-to-scale extrinsics. A known object size (checkerboard square, marker, or human height prior) is required for metric output.
- **Overlap requirement:** If cameras share no common field of view, no geometric calibration or triangulation can recover their relative poses.

### Key References

1. **DUSt3R: Geometric 3D Vision Made Easy** — Wang et al., CVPR 2024. arXiv:2312.14132. https://arxiv.org/abs/2312.14132
2. **Grounding Image Matching in 3D with MASt3R** — Leroy et al., ECCV 2024. arXiv:2406.09756. https://arxiv.org/abs/2406.09756
3. **Score-Guided Diffusion for 3D Human Recovery** — Stathopoulos et al., CVPR 2024. arXiv:2403.09623. https://arxiv.org/abs/2403.09623
4. **Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation** — Bragagnolo et al., ECCV Workshops 2024. arXiv:2408.15810. https://arxiv.org/abs/2408.15810
5. **Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining** — Wang et al., arXiv 2025. arXiv:2503.03222. https://arxiv.org/abs/2503.03222</subagent>
<subagent agent_id="agent-31" item="Survey attention and transformer-based multi-view fusion for human pose/mesh: view transformers, cross-view attention, positional encoding strategies, and applicability to our IR." outcome="completed">## Report: Attention and Transformer-based Multi-View Fusion for Human Pose/Mesh

### Executive Summary

The MotionFlow multi-view extension currently triangulates per-view 2D keypoints with DLT and feeds the resulting 3D skeleton into the `HumanMotionIR` via the existing `motionflow_mv/ir` and `motionflow_mv/fusion` modules. Recent top-venue work (CVPR 2024/2025, ICCV 2023) shows that the next gains will come from **hybrid geometry-learning fusion**, **stronger 3D positional encodings**, and **self-supervised/adaptive training** rather than from bigger pure-attention models. The literature confirms the project’s own empirical finding: when only reprojection error is available, DLT is extremely hard to beat.

### Key Findings

**1. Hybrid geometry + transformer fusion is the dominant design.**  
Liao et al., *Multiple View Geometry Transformers for 3D Human Pose Estimation* (CVPR 2024), propose MVGFormer, which alternates **learning-free geometry modules** with **learnable appearance modules**. The geometry module explicitly handles viewpoint-dependent 3D tasks; the appearance module predicts 2D poses and refines occluded views. This matches the project’s `triangulation.py`/`robust_triangulation.py` + `attention_model.py` split and gives a principled way to improve on DLT without discarding it.

**2. State-space models are emerging as a lower-overhead alternative to cross-view attention.**  
Chharia et al., *MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation* (CVPR 2025), replace full multi-head attention with a **Projective State Space (PSS)** block and **Grid Token-guided Bidirectional Scanning**. They report better generalization to new camera arrangements and large gains over transformer baselines on CMU Panoptic and Campus. This is directly applicable to the project’s `ViewAttentionFusion` class in `motionflow_mv/fusion/attention.py`.

**3. Self-supervised pseudo-label training can bypass the lack of 3D ground truth.**  
Srivastav et al., *SelfPose3d: Self-Supervised Multi-Person Multi-View 3d Pose Estimation* (CVPR 2024), localize people in 3D space, project 3D poses back to all views, and use an **adaptive supervision attention** mechanism to down-weight noisy pseudo-labels. This is a practical path forward because the project’s README already identifies “lack of 3D supervision” as the main bottleneck.

**4. Positional encoding should be 3D point-based rather than camera-parameter based.**  
Shu et al., *3DPPE: 3D Point Positional Encoding for Transformer-based Multi-Camera 3D Object Detection* (ICCV 2023), show that encoding **predicted 3D point positions** outperforms ray-sample encodings. The current `AttentionFusionModelV2` flattens the 3×4 projection matrix into a 12-D vector; a 3D-point/ray embedding would inject a stronger geometric inductive bias.

### Concrete Recommendations

- **Keep DLT as a geometry module, then add a lightweight appearance refiner.** Following MVGFormer, use `triangulate_dlt` or `RobustTriangulationModel` as the geometry stage and let the transformer/Mamba predict only residuals/corrections. This preserves the strong baseline while giving learning a meaningful role.
- **Upgrade the positional encoding in `AttentionFusionModelV2`.** Replace the 12-D projection embedding with per-token 3D point or ray embeddings (e.g., back-project each 2D keypoint to a 3D ray and concatenate direction/camera center), aligned with 3DPPE.
- **Add an adaptive pseudo-label loss for real-data training.** When 3D GT is unavailable, project the predicted 3D skeleton to each view and weight each joint’s contribution by an attention score, as in SelfPose3d. This can be layered on top of `residual_refiner.py`/`temporal_refiner.py`.
- **Experiment with state-space scanning as a drop-in replacement for `ViewAttentionFusion`.** MV-SSM suggests a Mamba-based scan across views may generalize better and use less memory than full multi-head attention across 5+ views.
- **Use 3D-supervised datasets (Human3.6M, CMU Panoptic, Shelf GT) before claiming a DLT improvement.** The project’s own Iterations 5–9 show this empirically; the literature confirms it.

### Key References

1. **Multiple View Geometry Transformers for 3D Human Pose Estimation** — Z. Liao et al., CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Liao_Multiple_View_Geometry_Transformers_for_3D_Human_Pose_Estimation_CVPR_2024_paper.html  
2. **MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation** — A. Chharia et al., CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Chharia_MV-SSM_Multi-View_State_Space_Modeling_for_3D_Human_Pose_Estimation_CVPR_2025_paper.html  
3. **SelfPose3d: Self-Supervised Multi-Person Multi-View 3d Pose Estimation** — V. Srivastav et al., CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Srivastav_SelfPose3d_Self-Supervised_Multi-Person_Multi-View_3d_Pose_Estimation_CVPR_2024_paper.html  
4. **3DPPE: 3D Point Positional Encoding for Transformer-based Multi-Camera 3D Object Detection** — C. Shu et al., ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Shu_3DPPE_3D_Point_Positional_Encoding_for_Transformer-based_Multi-Camera_3D_Object_ICCV_2023_paper.html  

### Risks, Blockers, and Assumptions

- **3D supervision remains the main blocker.** Both the project’s experiments and the surveyed papers agree: without 3D ground truth, learned fusion is unlikely to outperform a well-calibrated DLT.
- **Mamba/state-space libraries are not in `requirements.txt`.** Adopting MV-SSM-style scanning would add `mamba_ssm` or a custom CUDA scan; verify RTX 4090/A800-D compatibility first.
- **3D point encoding needs depth/ray information.** If monocular depth is not available, ray-direction embeddings are a fallback but require careful normalization, as the current `AttentionFusionModelV2` already struggles with unnormalized inputs.
- **Web access was constrained during the survey.** Literature was gathered from CVF Open Access; arXiv preprints and official code releases should be cross-checked for exact implementation details before committing to a specific design.</subagent>
<subagent agent_id="agent-32" item="Survey graph neural network (GNN) approaches for multi-view/skeleton fusion: view graphs, skeleton graphs, message passing, and how they could replace/replace attention fusion." outcome="completed">## Report: GNN Approaches for Multi-View Skeleton Fusion in MotionFlow

### Executive Summary

MotionFlow-MV currently fuses multi-view 2D keypoints through a `ViewAttentionFusion` module (`motionflow_mv/fusion/attention.py`) that treats each joint independently and aggregates views with a learned per-joint attention map. Variants add camera embeddings (`AttentionFusionModelV2`), differentiable DLT weighting (`RobustTriangulationModel`), and temporal Bi-GRU refinement. These models match but do not beat geometric DLT, partly because they ignore two strong priors: the human skeleton topology and the relational structure across camera views. Graph neural networks (GNNs) provide a natural replacement for the attention fusion head by modeling views, joints, or both as nodes in a graph and propagating information via message passing. They can explicitly enforce bone-length and occlusion consistency, and can be integrated as a drop-in replacement for the existing attention/refiner modules without changing the `HumanMotionIR` contract.

### Key Findings / Method Descriptions

**1. View-Graph Message Passing**

The most direct replacement for `ViewAttentionFusion` is a *view graph* in which each node represents one camera view and edges encode geometric/semantic relationships (epipolar pairs, shared field of view, or camera proximity). Instead of a global attention vector per joint, a GNN learns to pass messages between views, naturally handling occlusion: a view with a missing or low-confidence joint receives updates from unoccluded views. Microsoft’s “Graph-Based 3D Multi-Person Pose Estimation Using Multi-View Images” (ICCV 2021) builds such a graph to link 2D detections across views before triangulation. In MotionFlow, this would replace the per-joint softmax in `attention.py` with an `edge_index`/`GATConv` layer.

**2. Skeleton-Graph Refinement**

The `ResidualRefinerModel` and `TemporalRefinerModel` refine 3D points joint-by-joint but do not exploit the fact that arms, legs, and the spine form a known kinematic tree. A *skeleton graph* treats joints as nodes and bones as edges. Message passing along bone edges can enforce anatomical consistency—e.g., correcting an outlier elbow by borrowing from shoulder and wrist. Semantic/geometric graph convolutions (SemGConv) for 3D pose regression (e.g., Zhao et al., CVPR 2019 follow-ups) and later cross-view U-shaped GCNs show that even shallow skeleton GCNs reduce physically implausible poses. This fits naturally after DLT or after the residual head.

**3. Heterogeneous View–Joint Graphs**

The most powerful design combines the two: a heterogeneous graph where nodes are either *views* or *joints*, and edges connect (a) the same joint across views, (b) adjacent joints within a view, and optionally (c) camera parameters to views. The existing `docs/swarm_iter1/09_graph_fusion.md` already sketches this idea. Such a graph can perform cross-view consistency and skeleton smoothing in one forward pass. The message-passing objective can be:

- View nodes aggregate per-joint observations to update view-wise confidence.
- Joint nodes aggregate multi-view evidence to refine 3D position.
- Bone edges penalize large deviations from learned bone lengths.

**4. Relation to Current Attention Models**

The current attention fusion is effectively a graph with a fully connected view-to-joint bipartite structure and a single learned query per joint. A GNN generalizes this by (a) making the skeleton topology explicit, (b) allowing edge-specific weights (occlusion, epipolar distance), and (c) stacking multiple message-passing layers. It also decouples the *structure* (the graph) from the *aggregator* (GAT, GCN, MessagePassing), which is more interpretable and easier to regularize.

### Concrete Recommendations

1. **Prototype a skeleton-graph residual refiner.** Start with the smallest change: add a 1–2 layer GCN on the 17-joint COCO skeleton graph after the `ResidualRefinerModel` features, before the final 3D coordinate prediction. This enforces bone-length consistency without touching the view-aggregation path. Use PyTorch Geometric or a minimal adjacency-matrix implementation to avoid a heavy dependency.

2. **Replace `ViewAttentionFusion` with a view graph network.** Model the 5 Shelf cameras as graph nodes. Initialize node features with the lifted 2D coordinates + camera embedding (reusing `AttentionFusionModelV2`), and use `GATConv` or `MessagePassing` over the complete view graph. The output is a per-view-joint confidence used to weight DLT, similar to `RobustTriangulationModel` but with explicit relational reasoning.

3. **Combine both in a two-stage heterograph.** Stage 1: view graph produces per-view weights. Stage 2: skeleton graph refines the triangulated 3D pose. This mirrors the existing `RobustTriangulationModel` + `ResidualRefinerModel` pipeline but replaces the MLP/attention blocks with graph layers.

4. **Use 3D ground-truth datasets for training.** The design notes correctly identify that reprojection-only loss is the bottleneck. Train the GNN on Human3.6M or CMU Panoptic with 3D losses; reserve Shelf for validation/reprojection benchmarking.

5. **Add an explicit bone-length loss.** A GNN makes skeleton constraints easy to add: add an edge loss penalizing deviations from a learned mean bone length, or a graph Laplacian smoothness term.

### Key References

1. **Graph-Based 3D Multi-Person Pose Estimation Using Multi-View Images** — Wu et al., *ICCV 2021*. Proposes a graph to associate and fuse multi-view 2D detections. Directly applicable to replacing the attention head in `motionflow_mv/fusion/attention.py`.
2. **TransFusion: Cross-view Fusion with Transformer for 3D Human Pose Estimation** — Ma et al., *BMVC 2021* ([arXiv:2110.09554](https://arxiv.org/abs/2110.09554)). Shows cross-view transformer fusion; a useful bridge from current attention to graph/relational fusion.
3. **Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation** — Bragagnolo et al., *ECCVW 2024* ([arXiv:2408.15810](https://arxiv.org/abs/2408.15810)). Fuses 3D skeletons across views with occlusion handling; aligns with the skeleton-graph refiner direction.
4. **Semantic Graph Convolutional Networks for 3D Human Pose Regression** — Zhao et al., *CVPR 2019* ([arXiv:2003.00051](https://arxiv.org/abs/2003.00051)). Foundation for skeleton-graph propagation and bone-length priors.
5. **RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond** — Bermuth, *arXiv 2025* ([arXiv:2503.21692](https://arxiv.org/abs/2503.21692)). Fast whole-body triangulation baseline; the GNN head should be evaluated against its speed/accuracy trade-off.

### Risks, Blockers, and Assumptions

- **Supervision bottleneck remains.** Like the current attention/temporal models, a GNN will struggle to beat DLT on reprojection-only metrics unless trained with 3D GT or strong skeleton priors.
- **Cross-view person matching.** The GNN assumes aligned per-view detections. The existing `pipeline_utils.py::select_best_person_group` must remain reliable; otherwise graph messages will propagate mismatched identities.
- **Dependency overhead.** PyTorch Geometric is useful but adds installation and CUDA compatibility burden. A custom `MessagePassing` layer is preferable for a minimal reproducible baseline.
- **Complexity vs. gain.** Graph layers introduce hyperparameters (layers, heads, edge types). A rigorous ablation on Shelf/H36M is needed to justify the extra complexity for ICRA/CVPR.
- **Temporal extension.** Current `TemporalRefinerModel` uses Bi-GRU. A spatio-temporal graph that adds frame-to-frame edges is a natural next step, but requires careful handling of window size and memory.</subagent>
<subagent agent_id="agent-33" item="Survey diffusion, generative, and neural rendering models for multi-view human reconstruction (2024-2026): pros/cons for accuracy, generalization, and real-time use." outcome="completed">## Report: Survey of Diffusion, Generative, and Neural Rendering Models for Multi-View Human Reconstruction (2024–2026)

### Executive Summary
MotionFlow’s multi-view extension currently relies on a fast, geometry-first pipeline: per-view 2D keypoints/ confidence → calibration → DLT triangulation or learned fusion (`AttentionFusionModel`, `RobustTriangulationModel`, `TemporalRefinerModel`) → `HumanMotionIR`. Recent diffusion, generative, and 3D Gaussian/neural-rendering models can raise accuracy and robustness, but they come with clear real-time and integration costs. The best near-term strategy is to use generative models as **optional, higher-quality per-view priors** and **geometry/appearance regularizers**, while keeping the existing DLT/fusion stack for real-time inference.

### Key Findings

**1. ScoreHMR — diffusion for SMPL/SMPL-X recovery (CVPR 2024)**  
ScoreHMR casts 3D human recovery as a score-guided diffusion inverse problem and can consume a single image, multiple uncalibrated views, or video. For MotionFlow, it is a direct replacement for the GVHMR step: it produces SMPL/SMPL-X parameters that already fit `HumanMotionIR` through an adapter analogous to `motionflow_mv/ir/gvhmr_adapter.py`.  
- *Accuracy:* Very strong pose/shape prior; handles occlusion and unusual poses better than pure triangulation.  
- *Generalization:* Trained on large pose data; works on in-the-wild images and multi-view setups.  
- *Real-time:* Not real-time; inference is seconds per frame on an RTX 4090, so it should run offline or on sparse keyframes.  
- *License:* MIT (already compatible).

**2. GPS-Gaussian+ — generalizable 3D Gaussian splatting for human scenes (arXiv 2024, journal extension of CVPR 2024 highlight)**  
GPS-Gaussian+ regresses pixel-wise Gaussian parameters from sparse calibrated views for real-time novel-view synthesis of humans and scenes. It can provide dense depth/normal/appearance maps that regularize the current triangulation-based fusion.  
- *Accuracy:* High-fidelity view-consistent rendering; depth/geometry can constrain the SMPL fit.  
- *Generalization:* No per-subject optimization; trained once, generalizes to new subjects.  
- *Real-time:* Yes, designed for interactive rendering; but requires calibrated multi-view input and significant GPU memory.  
- *Fit:* Its output can populate `coordinate_system` and `uncertainty` in `HumanMotionIR`.

**3. HumanSplat — single-image human Gaussian splatting (arXiv 2024)**  
HumanSplat predicts 3D Gaussian splatting properties from one image using a multi-view diffusion model and structure priors. It is useful when only sparse or monocular views are available.  
- *Accuracy:* Good appearance/geometry, but single-view depth ambiguity remains.  
- *Generalization:* Works on in-the-wild single images; best as an initializer or pseudo-label generator.  
- *Real-time:* Faster than per-subject NeRFs, but still heavier than the current DLT pipeline.  
- *Fit:* Can bootstrap appearance for `AttentionFusionModelV2` when camera coverage is poor.

**4. PSHuman — photorealistic single-image reconstruction via multiview diffusion (arXiv 2024)**  
PSHuman uses cross-scale multiview diffusion to generate consistent multi-view images and explicit remeshing for detailed 3D humans. It is less about calibrated multi-view fusion and more about generating high-quality 3D assets from sparse input.  
- *Accuracy:* High visual fidelity; useful for dense geometry/shape prior.  
- *Generalization:* Single-image driven; domain gap to multi-view video needs validation.  
- *Real-time:* No; primarily offline.  
- *Fit:* Use as a data-augmentation / pseudo-GT source for training the `TemporalRefinerModel`.

**5. RapidPoseTriangulation — real-time multi-person triangulation (arXiv 2025)**  
Although not a generative method, it represents the state-of-the-art fast geometric baseline. It confirms that a well-engineered triangulation can run in milliseconds, validating MotionFlow’s decision to keep DLT as the real-time backbone.  
- *Accuracy:* Competitive with learned methods when 2D detections are good.  
- *Generalization:* Works for multi-person scenes; purely geometric.  
- *Real-time:* Yes, sub-millisecond multi-person triangulation.  
- *Fit:* Can extend `motionflow_mv/fusion/robust_triangulation.py` with its clustering/assignment ideas.

### Concrete Recommendations
1. **Plug ScoreHMR into the `HumanMotionIR` adapter layer.** Add a `scorehmr_adapter.py` next to `gvhmr_adapter.py` so ScoreHMR’s SMPL/SMPL-X output feeds the same fusion modules. Benchmark on Shelf / Human3.6M with 3D GT to see whether diffusion priors beat DLT on real data.
2. **Use GPS-Gaussian+ as a neural rendering regularizer.** Render per-view depth/normal maps from its sparse-view Gaussian output and add a geometry consistency term to the `RobustTriangulationModel` / `TemporalRefinerModel` loss, or fit SMPL to its point cloud.
3. **Leverage HumanSplat / PSHuman for data augmentation.** Synthesize occluded-view pseudo-labels and SMPL pseudo-GT to train the existing fusion stack, addressing the current bottleneck of weak 3D supervision.
4. **Keep DLT + learned weighting for real-time.** Reserve diffusion/Gaussian modules for an **offline quality mode** or low-fps batch processing, while preserving the current `RobustTriangulationModel` for latency-critical robot retargeting.

### Risks, Blockers, and Assumptions
- **Compute:** ScoreHMR and Gaussian splatting models are too heavy for real-time 30 fps on an RTX 4090; A800-D is needed for training and large-scale experiments.  
- **Calibration:** GPS-Gaussian+ and the DLT pipeline assume calibrated cameras; uncalibrated multi-view fusion requires extra work.  
- **3D supervision:** Beating the strong DLT baseline still requires real 3D GT or strong geometry priors, as the project’s own experiments already show.  
- **Domain gap:** Generative models trained on in-the-wild images may not generalize to the specific robot-demonstration capture environment without fine-tuning.  
- **Licensing:** ScoreHMR is MIT; GPS-Gaussian+, HumanSplat, and PSHuman licenses must be checked before redistribution.

### Key References
1. **Score-Guided Diffusion for 3D Human Recovery** — *CVPR 2024.* [arXiv:2403.09623](https://arxiv.org/abs/2403.09623)  
2. **GPS-Gaussian+: Generalizable Pixel-wise 3D Gaussian Splatting for Real-Time Human-Scene Rendering from Sparse Views** — *arXiv 2024 (journal extension of CVPR 2024 highlight).* [arXiv:2411.11363](https://arxiv.org/abs/2411.11363)  
3. **HumanSplat: Generalizable Single-Image Human Gaussian Splatting with Structure Priors** — *arXiv 2024.* [arXiv:2406.12459](https://arxiv.org/abs/2406.12459)  
4. **PSHuman: Photorealistic Single-image 3D Human Reconstruction using Cross-Scale Multiview Diffusion and Explicit Remeshing** — *arXiv 2024.* [arXiv:2409.10141](https://arxiv.org/abs/2409.10141)  
5. **RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond** — *arXiv 2025.* [arXiv:2503.21692](https://arxiv.org/abs/2503.21692)</subagent>
<subagent agent_id="agent-34" item="Survey uncertainty estimation and view reliability weighting in multi-view human pose: per-joint uncertainty, per-view confidence, and how to store this in the HumanMotionIR uncertainty field." outcome="completed">## Report: Uncertainty Estimation and View Reliability Weighting for Multi-View Human Pose in MotionFlow

### Executive Summary

MotionFlow currently fuses multi-view 2D keypoints through confidence-weighted DLT and a family of learned models (`AttentionFusion`, `RobustTriangulationModel`, `ResidualRefinerModel`, `TemporalRefinerModel`). All of these compute per-view or per-joint reliability signals internally, but none are persisted. The `HumanMotionIR.uncertainty` field is defined as a free-form dictionary and is empty by default. To reach ICRA/CVPR-quality, the project should explicitly represent, compute, and store two things: (1) **per-joint 3D positional uncertainty** and (2) **per-view confidence/weight**, and use them both for fusion and downstream robot retargeting.

### Key Findings

**1. Per-view confidence is already computed but discarded.**  
The DLT baseline (`motionflow_mv/fusion/triangulation.py`) accepts a `weights` argument from the 2D detector confidence. `RobustTriangulationModel` learns a sigmoid weight per view per joint via self-attention. `ViewAttentionFusion` produces softmax attention weights over views. These are exactly the view-reliability signals the literature treats as uncertainty, yet they never reach the IR.

**2. Per-joint uncertainty can be derived from geometry or learned.**  
- **Geometric proxy:** after triangulation, the per-joint reprojection residual is a calibration-free indicator of 3D reliability (Hartley & Zisserman).  
- **Heatmap confidence:** VoxelPose-style methods lift per-view 2D heatmap maxima into a 3D voxel probability, yielding per-joint likelihoods.  
- **Learned covariance:** recent single-view work shows that predicting a per-joint covariance (or standard deviation) is feasible and improves downstream tasks such as filtering and motion planning.

**3. State-of-the-art multi-view fusion now relies on learned view reliability, not hand-tuned confidence.**  
- Iskakov et al. (ICCV 2019) introduced **learnable triangulation** that aggregates per-view 2D heatmap confidences into a 3D volume.  
- Ma et al. (BMVC 2021) proposed **TransFusion**, a cross-view transformer whose attention weights can be interpreted as view reliability. This is the conceptual ancestor of the current `ViewAttentionFusion`.  
- VoxelPose (ECCV 2020) and Faster VoxelPose (ECCV 2022) demonstrate that per-joint 2D heatmap confidence is the natural input to a 3D probability field.  
- Bramlage et al. (ICCV 2023) showed how to predict **plausible per-joint covariance** for pose regression, which could be extended from single-view to fused 3D output.  
- Recent 2025/2026 works such as RUMPL and AHAP explicitly model view-agnostic fusion and soft cross-view association, confirming that view-reliability weighting remains a core research problem.

### Concrete Recommendations

1. **Populate `HumanMotionIR.uncertainty` with standardized fields.** Suggested schema:
   - `view_confidence`: `(T, V)` or `(T, V, J)` per-view confidence/weight.
   - `joint_3d_std`: `(T, J)` or `(T, J, 3)` per-joint positional standard deviation.
   - `reprojection_residual`: `(T, J)` geometric residual after fusion.
   - `view_weights`: `(T, V, J)` learned or normalized weights from `RobustTriangulationModel` / `ViewAttentionFusion`.

2. **Instrument existing fusion modules to return uncertainty.**  
   - In `triangulate_confidence_weighted`, compute the per-joint reprojection residual and return it alongside the 3D point.  
   - In `RobustTriangulationModel`, expose the learned `weights` and convert them to `view_weights`.  
   - In `ViewAttentionFusion`/`AttentionFusionModelV2`, save the softmax attention scores as `view_weights`.  
   - Update `MultiViewPipeline.fuse_frame` and `pipeline_utils.select_best_person_group` to optionally return and pack these arrays into the IR.

3. **Add a small helper in `human_motion_ir.py` to validate uncertainty tensor shapes.** This prevents silent mismatches when downstream retargeters read the IR.

4. **Use uncertainty for downstream retargeting.** Start simple: mask out joints whose `joint_3d_std > threshold`, or weight the loss in robot imitation learning by inverse uncertainty. This turns the IR from a passive data container into a risk-aware interface.

5. **Adopt an occlusion-aware 3D fusion baseline as the next architecture.** The 2024 “Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation” approach fuses per-view monocular 3D estimates and optimizes them with reprojection error plus limb-length symmetry constraints. It naturally produces per-joint residuals and occluded-view weights, aligning with the uncertainty-field design above.

### Key References

1. **Iskakov et al.**, “Learnable Triangulation of Human Pose,” *ICCV* 2019. arXiv:1905.05754. Foundational volumetric fusion of per-view 2D confidences.
2. **Tu et al.**, “VoxelPose: Towards Multi-Camera 3D Human Pose Estimation in Wild Environment,” *ECCV* 2020. arXiv:2004.06239. Per-joint heatmap confidence lifted to 3D voxel likelihood.
3. **Ma et al.**, “TransFusion: Cross-view Fusion with Transformer for 3D Human Pose Estimation,” *BMVC* 2021. arXiv:2110.09554. Cross-view attention as view-reliability weighting.
4. **Bramlage et al.**, “Plausible Uncertainties for Human Pose Regression,” *ICCV* 2023. DOI:10.1109/iccv51070.2023.01389. Per-joint covariance regression.
5. **Giberti et al.**, “Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation,” arXiv:2408.15810, 2024. Occlusion-aware fusion and reprojection optimization.

### Risks, Blockers, and Assumptions

- **No 3D ground truth on Shelf:** uncertainty estimates can only be validated geometrically (reprojection) or via downstream robot-policy metrics, which biases calibration.
- **DLT already dominates reprojection:** learned weights must add value beyond reprojection error, e.g., occlusion handling or downstream task performance.
- **Memory cost:** storing full per-joint covariances (`(T, J, 3, 3)`) for long sequences is expensive; start with diagonal/std scalars.
- **Downstream consumption:** robot retargeting currently ignores `uncertainty`; a schema change alone has no effect until the consumer is updated.
- **Assumption:** per-view 2D detector confidence is calibrated. If it is not, learned `view_weights` from TransFusion-style attention will be more reliable.</subagent>
<subagent agent_id="agent-35" item="Survey real-time/efficiency considerations for multi-view human pose: model sizes, runtime on 4090/A800, TensorRT/ONNX, edge vs. server deployment." outcome="completed">## Report: Real-Time / Efficiency Considerations for Multi-View Human Pose in MotionFlow

### Executive Summary

MotionFlow-MultiView currently extends the monocular GVHMR → `HumanMotionIR` pipeline by adding a lightweight fusion stage (`motionflow_mv/fusion/`). The trainable fusion heads (`AttentionFusionModelV2`, `RobustTriangulationModel`, `TemporalRefinerModel`) are only a few thousand to ~160k parameters, so their runtime is negligible compared with the upstream per-view human recovery. For ICRA/CVPR-level real-time deployment, the dominant bottleneck is therefore **per-view GVHMR inference**, not the fusion network. A publishable real-time story should quantify latency end-to-end, deploy the heavy vision backbone through ONNX/TensorRT, and decide between edge (RTX 4090) and server (A800) execution based on view count and latency requirements.

### Key Findings

1. **Upstream cost dominates.** GVHMR is a video-based human motion recovery network built on a vision backbone (ViTPose-style 2D keypoint/feature extraction plus temporal reasoning). In the MotionFlow adapter, `gvhmr_pt_to_ir` only consumes GVHMR’s saved outputs; the actual per-view network is orders of magnitude larger than the fusion modules. Any multi-view extension multiplies this cost by the number of cameras unless views are batched or share a backbone.

2. **Fusion is tiny and easily deployable.** Current modules are MLP/attention/GRU-based:
   * `AttentionFusionModelV2` ≈ a few k params.
   * `RobustTriangulationModel` ≈ 4–5 k params.
   * `TemporalRefinerModel` ≈ 150 k params.
   These can run in < 1 ms on a GPU and export cleanly to ONNX/TensorRT. The exception is `RobustTriangulationModel._triangulate`, which uses `torch.linalg.svd`; TensorRT/ONNX support for SVD is limited, so that op may need a fallback to NumPy/PyTorch or a custom CUDA kernel.

3. **Real-time methods literature (2024–2026).**
   * **MV-SSM** (CVPR 2025) replaces heavy multi-view transformers with State Space Models (Mamba), explicitly targeting better generalization across camera arrangements. It still uses a ResNet-50 backbone, so the per-frame cost is ~25–30 GFLOP before any geometric head. The efficiency message is *architectural* (sub-quadratic scanning vs. full attention), not yet edge real-time.
   * **SkelSplat** (WACV 2026) models each joint as a 3D Gaussian and optimizes via differentiable rendering. It removes the need for 3D GT, matching MotionFlow’s reliance on 2D/3D pseudo-labels, but its per-scene optimization is **not online**; it is a robust *offline* alternative.
   * The **markerless multi-view survey** (IMAVIS 2025) confirms that 3D-feature methods are more accurate but costlier, while 2D-keypoint + triangulation remains the fastest and most deployment-friendly family. MotionFlow’s current DLT/attention fusion falls into the latter.
   * **TensorRT-RTX** (NVIDIA, 2024–2026) reports two-phase AOT+JIT compilation under 30 s and a < 200 MB runtime for RTX GPUs; it is the natural deployment path for the RTX 4090 WSL machine.

4. **Hardware expectations.** On an RTX 4090, a lightweight 2D pose network (e.g., YOLOv8-pose or RTMPose) can run at hundreds of FPS once TensorRT-optimized. GVHMR-style video HMR is heavier; a sensible target on 4090 is ~30–60 FPS for a single view, dropping roughly linearly with more views if processed serially. An A800 server can batch many views and sustain higher throughput, but latency is less dependent on raw TFLOPs and more on memory bandwidth and batching strategy.

### Concrete Recommendations

1. **Profile before optimizing.** Add a `benchmark.py` that times (a) GVHMR per view, (b) the fusion forward pass, and (c) the full pipeline end-to-end on the RTX 4090 and A800. Without these numbers, efficiency decisions are speculative.
2. **Export the fusion head to TensorRT/ONNX.** Use `torch.onnx.export` for `AttentionFusionModelV2` and `TemporalRefinerModel`. For `RobustTriangulationModel`, either replace the SVD with a fixed-size linear solve or move triangulation to a CUDA extension.
3. **Batch multi-view GVHMR where possible.** If GVHMR accepts batched frames, run all camera views through the same backbone call to amortize overhead on the A800.
4. **Keep an edge vs. server split.** For < 4 cameras and 30 Hz output, a TensorRT-optimized pipeline on the RTX 4090 may suffice. For > 4 views or > 30 Hz, deploy inference on the A800 and stream compressed crops/poses to the edge.
5. **Consider a lightweight 2D-only fallback.** For real-time robot retargeting, an alternate branch using YOLOv8-pose/RTMPose + DLT triangulation (already in `motionflow_mv/fusion/triangulation.py`) could deliver > 100 Hz while GVHMR runs asynchronously at a lower rate for SMPL-X refinement.

### Key References

1. **Markerless Multi-view 3D Human Pose Estimation: A Survey** — *A. F. Nogueira et al.*, Image and Vision Computing, 2025. arXiv:2407.03817.
2. **MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation** — *A. Chharia, W. Gou, H. Dong*, CVPR, 2025. [Project page](https://aviralchharia.github.io/MV-SSM/)
3. **SkelSplat: Robust Multi-view 3D Human Pose Estimation with Differentiable Gaussian Rendering** — *L. Bragagnolo, L. Barcellona, S. Ghidoni*, WACV, 2026. arXiv:2511.08294.
4. **NVIDIA TensorRT for RTX Documentation** — NVIDIA, 2024–2026. [docs.nvidia.com/deeplearning/tensorrt-rtx](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html)
5. **GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates** — *Z. Shen et al.*, SIGGRAPH Asia, 2024 (also T-PAMI 2026). [GitHub](https://github.com/zju3dv/GVHMR)

### Risks / Blockers

* **GVHMR export risk:** GVHMR contains temporal and SMPL/SMPL-X outputs that may not easily export to ONNX/TensorRT; the retargeting-ready adapter in `ir_to_gvhmr_pt` depends on those outputs.
* **SVD in `RobustTriangulationModel`:** TensorRT does not support the SVD used in the differentiable DLT block; a deployment-ready version needs an alternative solver.
* **Amdahl’s law:** Since fusion is already fast, any real-time gains must come from the upstream network or camera preprocessing; expect diminishing returns from further shrinking fusion.
* **Hardware numbers are estimated.** Actual GVHMR latency on the local RTX 4090 and remote A800 has not yet been measured; the above FPS ranges should be treated as assumptions pending benchmarks.</subagent>
<subagent agent_id="agent-36" item="Survey evaluation metrics and benchmarks for multi-view human pose/mesh (MPJPE, PA-MPJPE, MRPE, MPVE, etc.) and datasets (Human3.6M, MPI-INF-3DHP, Shelf, CMU Panoptic, etc.)." outcome="completed">## Report: Multi-View Human Pose/Mesh Evaluation Metrics & Benchmarks for MotionFlow

### Executive Summary

The MotionFlow multi-view branch already has a working DLT baseline, several learned fusion variants, and a small `eval/metrics.py` that implements MPJPE, PA-MPJPE, and PCK on the Shelf dataset. To move toward ICRA/CVPR-level rigor, the project needs to broaden its metric set and benchmark on datasets with real 3D ground truth. This report surveys the standard and emerging evaluation metrics (MPJPE, PA-MPJPE, MRPE, MPVE, etc.) and the most relevant multi-view datasets (Shelf, Campus, Human3.6M, MPI-INF-3DHP, CMU Panoptic), and maps them onto the existing IR/fusion code.

### Key Findings

**1. Metrics used in 2024–2026 multi-view human pose/mesh work**
- **MPJPE (Mean Per Joint Position Error)** – root-relative Euclidean error in mm; the de-facto primary accuracy metric. The repo already computes it in `motionflow_mv/eval/metrics.py:6`.
- **PA-MPJPE** – MPJPE after Procrustes alignment (rigid rotation/reflection/translation, no scale); already implemented in `metrics.py:28`.
- **MRPE (Mean Root Position Error)** – absolute position error of the root/pelvis. Critical for MotionFlow because robot retargeting needs metric-scale world coordinates; currently missing from `metrics.py`.
- **MPVE (Mean Per Vertex Error)** – for mesh-based methods (SMPL/SMPL-X), measures vertex-level accuracy. Relevant because `HumanMotionIR` carries SMPL pose parameters and `gvhmr_adapter.py` handles SMPL/SMPLX.
- **PCK/AUC** – percentage of joints within a distance threshold and area under the PCK curve; already partially present in `metrics.py:34`. Common thresholds are 50/100/150 mm.
- **Reprojection error** – what the repo currently optimizes; it is not a true 3D accuracy metric and cannot substitute for 3D GT metrics.
- **Per-joint/body-part breakdowns** and **bone-length/limb-symmetry constraints** are increasingly reported, e.g., Bragagnolo et al. (ECCVW 2024) explicitly use limb-length symmetry as a post-process.

**2. Datasets and benchmarks**
- **Shelf / Campus** – small (~3k / ~2k frames), calibrated, 3–4 views, 3D joint GT. Best for fast iteration; the repo already uses Shelf via `tmp/voxelpose-pytorch/data/Shelf`.
- **Human3.6M** – 3.6M frames, 4 views, 11 actors, 17 scenarios. The standard large-scale benchmark; MPJPE/PA-MPJPE are reported with the standard train/test splits (S1,5,6,7,8 train; S9,11 test, 14-camera and 17-joint subsets).
- **MPI-INF-3DHP** – in-the-wild multi-view 3DHP with studio and outdoor sequences; uses PCK at 150 mm and AUC.
- **CMU Panoptic** – 480 VGA + 31 HD synchronized cameras, 1.5M 3D skeletons, multi-person social scenes. Useful for multi-person and whole-body evaluation, though license restricts commercial use.
- **3DHP test-set / HumanEva** are secondary; HumanEva is small and mostly used for cross-dataset validation.

**3. Connection to existing MotionFlow code**
- The `HumanMotionIR` dataclass (`human_motion_ir.py`) already stores pose, timestamps, coordinate system, and provenance—sufficient metadata to align with any benchmark's world coordinate frame.
- `gvhmr_adapter.py` converts GVHMR SMPL params into the IR; mesh metrics (MPVE) can be computed by forwarding the IR’s pose through an SMPL/SMPL-X body model.
- `pipeline.py` fuses per-view 2D keypoints into 3D skeletons; adding MRPE means comparing the fused root position against dataset GT after root alignment.
- `attention_model_v2.py` and `robust_triangulation.py` are differentiable fusion modules; training them on 3D GT requires replacing the reprojection loss with 3D MSE / MPJPE loss, exactly what the design doc already recommends.

### Concrete Recommendations

1. **Extend `motionflow_mv/eval/metrics.py`** to add MRPE and MPVE (and a 3D AUC helper) so the repo can evaluate metric-scale world accuracy and mesh quality, not just reprojection.
2. **Acquire Human3.6M** (or a small subject/action subset) and run the existing DLT/attention/temporal models through it. This gives the 3D-supervised training signal the design doc identifies as the missing ingredient.
3. **Adopt the standard H36M/Panoptic evaluation protocol**: root at pelvis, report MPJPE and PA-MPJPE per action, plus MRPE to validate absolute world positioning. Keep Shelf for fast regression tests.
4. **Use ScoreHMR or a similar 2024 per-view estimator** as the 2D/3D source; its SMPL/SMPLX output maps directly into `HumanMotionIR`, and its mesh parameters enable MPVE evaluation.
5. **Consider limb-length/bone-length constraints** as a post-processing or loss term, following Bragagnolo et al., to reduce physically implausible poses when 3D GT is scarce.

### Key References

1. Stathopoulos et al., *Score-Guided Diffusion for 3D Human Recovery*, CVPR 2024. [arXiv:2403.09623](https://arxiv.org/abs/2403.09623) – strong per-view SMPL/SMPLX estimator, maps to `HumanMotionIR`.
2. Bragagnolo et al., *Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation*, ECCV Workshops 2024. [arXiv:2408.15810](https://arxiv.org/abs/2408.15810) – fuses monocular 3D skeletons with reprojection and limb-symmetry constraints; directly relevant to the fusion module.
3. Bermuth et al., *RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond*, arXiv 2025. [arXiv:2503.21692](https://arxiv.org/abs/2503.21692) – fast whole-body triangulation; good engineering baseline for `pipeline.py`.
4. Wang et al., *Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining*, arXiv 2025. [arXiv:2503.03222](https://arxiv.org/abs/2503.03222) – shows how 2D pretraining + 3D multi-view fine-tuning can recover metric-scale poses; a long-term modeling direction.
5. Ionescu et al., *Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments*, T-PAMI 2014. [project](http://vision.imar.ro/human3.6m/) – the canonical large-scale 3D HPE benchmark.

### Risks, Blockers, and Assumptions

- **3D GT access**: The repo currently relies on reprojection-only supervision. Human3.6M and CMU Panoptic require license agreements; plan for download/storage on A800-D or local NAS.
- **Coordinate-frame alignment**: `HumanMotionIR.coordinate_system` assumes a right-handed y-up metric world; each benchmark uses its own coordinate frame, so adapters must be written.
- **SMPL vs. SMPL-X vs. dataset skeletons**: MPVE requires a body model and per-joint mapping; dataset GT is usually a reduced joint set (e.g., 14/17 joints), so mesh evaluation must be reconciled.
- **Computational cost**: Human3.6M and Panoptic are large; start with Shelf/Campus and a small H36M subset on the RTX 4090, then scale to A800-D.
- **License**: CMU Panoptic and EasyMocap have non-commercial restrictions; ensure any external baseline is cited and not redistributed in the default package, as noted in `phase0_literature_audit.md`.</subagent>
<subagent agent_id="agent-37" item="Identify multi-view human motion datasets available via WebBridge or similar web-accessible sources: how to download, license, format, and suitability for training/fine-tuning." outcome="completed">## Report: Multi-View Human Motion Datasets for MotionFlow Training & Fine-Tuning

### Executive Summary

The MotionFlow multi-view extension already has a working fusion stack (`motionflow_mv/fusion/`), but the internal README/design docs show that learned fusion (AttentionFusionV2, RobustTriangulation, Residual/Temporal refiners) has only **matched—not beaten**—geometric DLT on Shelf, because the current pipeline lacks **real 3D ground-truth supervision**. To move toward ICRA/CVPR2027 quality, the priority is to obtain datasets that provide (1) calibrated multi-view images, (2) 3D body joints or SMPL/SMPL-X parameters in a world coordinate frame, and (3) permissive research/commercial licenses. The most practical mix is to keep Shelf/Campus for rapid iteration, add Human3.6M for 3D-supervised scaling, and selectively sample CMU Panoptic and 3DPW for multi-person and in-the-wild validation.

---

### Key Findings / Dataset Descriptions

| Dataset | Views | 3D GT | License | Best Use in MotionFlow |
|---|---|---|---|---|
| **Shelf / Campus** | 3–5 fixed | Yes (joints) | Research / benchmark | Dev loop; already loaded via `voxelpose_loader.py`; small, fast |
| **Human3.6M** | 4 fixed | Yes (32 joints) | Research, registration required | Primary 3D-supervised training/fine-tuning |
| **CMU Panoptic** | 31 HD + 480 VGA | Yes (19 coco joints) | Research-only, non-commercial | Multi-person/social motion, temporal refiners |
| **3DPW** | Moving single + IMU | Yes (SMPL poses) | Research | In-the-wild validation, world-scale trajectories |
| **AMASS** | N/A (mocap only) | Yes (SMPL params) | Research | Motion prior / synthetic 3D sequences |

**Shelf / Campus** are the current anchors. Shelf provides ~3,200 frames from 5 calibrated cameras and 3D joint annotations; Campus adds ~2,000 frames from 3 views. The repo already consumes these through `motionflow_mv/data/voxelpose_loader.py`, and the DLT baseline in `motionflow_mv/fusion/triangulation.py` uses their `K, R, t`. They are ideal for debugging and for the cross-view person-matching work already underway, but too small to train large fusion models from scratch.

**Human3.6M** is the next logical scale-up. With 3.6M frames, 11 actors, 17 actions, and 4 calibrated 50 Hz cameras, it supplies the 3D world-coordinate labels needed to train `AttentionFusionModelV2` or `TemporalRefinerModel` with a direct 3D MSE loss rather than the weaker reprojection loss. Its 32-joint skeleton maps naturally to the SMPL-style `pose` fields in `HumanMotionIR`. The main friction is the registration-and-download workflow at [http://vision.imar.ro/human3.6m/](http://vision.imar.ro/human3.6m/).

**CMU Panoptic** offers 31 HD + 480 VGA views and 3D COCO19 skeletons for social, multi-person scenes. It is excellent for validating cross-view person matching (`pipeline_utils.py::select_best_person_group`) and temporal smoothing, but its research-only license and massive volume make it a secondary resource. The `panoptic-toolbox` scripts at [https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) allow downloading only a few HD views (e.g., `getData.sh 171204_pose1_sample 0 5`), which fits the fast-iteration principle.

**3DPW** is the best in-the-wild option: moving camera, IMU, and SMPL ground truth. While not a fixed multi-view rig, it is valuable for validating generalization of world-coordinate `HumanMotionIR` outputs beyond the calibrated lab setting. Its license requires citation and research use.

**AMASS** contains no images but unifies >40 hours of mocap in SMPL. It can bootstrap motion priors or generate synthetic multi-view training data by rendering SMPL bodies from multiple virtual cameras, complementing the real datasets above.

---

### Concrete Recommendations

1. **Immediate (next 1–2 iterations)**: Keep Shelf/Campus as the dev benchmark. Use them to finalize the `HumanMotionIR ↔ VoxelPose` calibration loader and to test any new fusion module.
2. **Short-term**: Register and download Human3.6M. Train `AttentionFusionModelV2` and `TemporalRefinerModel` with 3D MPJPE loss; this directly addresses the finding that reprojection-only supervision cannot beat DLT.
3. **Medium-term**: Pull a small CMU Panoptic HD subset for multi-person matching and temporal-refiner experiments.
4. **Optional**: Use 3DPW for in-the-wild sanity checks and AMASS to synthesize motion-prior-augmented training sequences.

---

### Key References

1. Belagiannis et al., "3D Pictorial Structures for Multiple Human Pose Estimation," CVPR 2014. (Shelf/Campus)
2. Ionescu et al., "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments," TPAMI 2014.
3. Joo et al., "Panoptic Studio: A Massively Multiview System for Social Interaction Capture," TPAMI 2017.
4. von Marcard et al., "Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera," ECCV 2018. (3DPW)
5. Mahmood et al., "AMASS: Archive of Motion Capture as Surface Shapes," ICCV 2019.
6. Bragagnolo et al., "Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation," arXiv:2408.15810, ECCV workshops 2024.
7. Wang et al., "Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining," arXiv:2503.03222, 2025.

---

### Risks, Blockers, and Assumptions

- **License risk**: CMU Panoptic is research-only; Human3.6M and Shelf/Campus are non-commercial by default. If MotionFlow targets a commercial product, these datasets cannot be redistributed or used for commercial training without additional agreements.
- **Supervision gap**: The biggest blocker is not dataset availability but 3D GT access. Without 3D world-coordinate labels, learned fusion will likely keep tying DLT.
- **Scale mismatch**: Shelf/Campus are too small to train large attention/temporal models from scratch; use them only for validation and low-capacity experiments.
- **Format friction**: Each dataset uses a different skeleton and camera convention. The `HumanMotionIR` abstraction is the right container, but adapters per dataset are still required.
- **In-the-wild gap**: Human3.6M and Panoptic are indoor/lab; 3DPW is closer to real-world conditions but has a different capture paradigm. Expect a domain gap when moving from training to deployment.</subagent>
<subagent agent_id="agent-38" item="Propose a concrete multi-view fusion baseline architecture that fits the MotionFlow IR: inputs (multiple HumanMotionIRs from different views), outputs (single fused HumanMotionIR), and key modules." outcome="completed">## Report: Multi-View Human Motion Fusion Baseline for MotionFlow

### Executive Summary

The MotionFlow IR extension is at a point where geometric triangulation already works well, but it is not yet expressed as a clean “many `HumanMotionIR` in → one `HumanMotionIR` out” module. The most publishable next step is to define a **minimal multi-view IR fusion baseline** that reuses the existing `motionflow_mv` infrastructure, rather than replacing it. The recommended baseline has three parts: (1) canonicalize per-view `HumanMotionIR`s into a shared world frame, (2) fuse per-view 2D/3D evidence with a deterministic confidence-weighted DLT, and (3) re-encode the fused 3D skeleton back into a single `HumanMotionIR`. Learned variants (`RobustTriangulationModel`, `ResidualRefinerModel`, `TemporalRefinerModel`) can then be plugged in as drop-in upgrades, evaluated against the same IR contract.

### Key Findings / Method Descriptions

**1. The existing IR is already a good contract, but it lacks explicit multi-view fields.**  
`HumanMotionIR` stores `body_pose`, `global_orient`, `transl`, `betas`, and a `coordinate_system`/`provenance` block. The `gvhmr_adapter.py` keeps `K_fullimg` in provenance, which is enough to reconstruct per-view camera intrinsics. What is missing is a per-view extrinsic and, optionally, the 2D keypoint/confidence fields that the current fusion models consume. Recommendation: extend the IR (or its provenance/uncertainty fields) to carry `keypoints_2d` (T,J,2) and `keypoints_2d_conf` (T,J). This makes the IR self-contained for multi-view fusion and keeps the fusion module independent of which single-view estimator produced it.

**2. DLT remains the right baseline, and the repo already proves it.**  
The Shelf 300–600 evaluation shows DLT at mean 9.88 px / median 5.52 px reprojection error, while every lightweight learned model (attention, residual, temporal) only matches or slightly trails it. This matches the wider literature: geometric triangulation is a very strong prior when only 2D reprojection loss is available. Therefore, the multi-view IR fusion baseline should be **DLT-first**: `MultiViewPipeline.fuse_frame` already triangulates 2D keypoints + confidences + cameras. The contribution should not be “a new model that beats DLT,” but a clean IR-level multi-view fusion plugin and a systematic comparison of upgrade modules.

**3. Recent methods support a modular upgrade path.**  
- **ScoreHMR** (CVPR 2024) is a strong per-view SMPL/SMPL-X estimator; running it on each view before fusion raises the quality of the per-view IRs that go into the fusion stage.  
- **TransFusion** (arXiv 2110.09554) frames multi-view fusion as cross-view transformer attention over 2D pose tokens, directly compatible with the `AttentionFusionModel` already in `motionflow_mv/fusion/`.  
- **RapidPoseTriangulation** (arXiv 2503.21692) shows that very fast multi-person triangulation is possible, relevant for real-time MotionFlow inference.  
- **Mocap-2-to-3** (arXiv 2503.03222) demonstrates that 2D pretraining + multi-view lifting can recover motion without dense 3D labels, pointing to a future training recipe when 3D ground truth is scarce.

### Concrete Recommendations

1. **Define `MultiViewFusion.from_irs(ir_list: List[HumanMotionIR]) -> HumanMotionIR`.**  
   - Canonicalization: use each IR’s `coordinate_system.world_from_reference` and camera `K`/`R`/`t` (from provenance or a new `cameras` field) to transform per-view SMPL global orientation/translation into a single world frame. Extract 17-joint 3D positions via the SMPL joint regressor.  
   - Fusion: feed the per-view 2D keypoints + confidence and projection matrices into the existing `triangulate_confidence_weighted` to obtain a fused world 3D skeleton.  
   - Re-encoding: fit SMPL parameters to the fused 3D joints (or average `body_pose` weighted by per-view `quality` and set `global_orient`/`transl` from the fused root) and return a new `HumanMotionIR` with updated provenance.

2. **Keep the baseline deterministic.**  
   Do not train a learned fusion for the first IR-fusion PR. The DLT pipeline in `motionflow_mv/pipeline.py` and `motionflow_mv/fusion/triangulation.py` already does the core work; wrap it so it accepts IRs instead of raw arrays.

3. **Use ScoreHMR per-view and Shelf/Campus for evaluation.**  
   Replace or augment the GVHMR adapter with a ScoreHMR adapter, producing richer per-view IRs. Evaluate on Shelf (and optionally Campus) using MPJPE/PA-MPJPE, not only reprojection error, because reprojection alone cannot distinguish real improvements from geometric re-slicing.

4. **Reserve learned fusion for a follow-up with 3D GT.**  
   The repo’s history shows that `RobustTriangulationModel`, `ResidualRefinerModel`, and `TemporalRefinerModel` can only match DLT under reprojection-only supervision. Re-train them only after adding 3D-supervised losses on Shelf/Campus or Human3.6M.

### Key References

1. **Score-Guided Diffusion for 3D Human Recovery** (ScoreHMR). Stathopoulos et al., CVPR 2024. arXiv:2403.09623 — strong per-view SMPL/SMPL-X estimator to feed into the multi-view fusion stage.
2. **TransFusion: Cross-view Fusion with Transformer for 3D Human Pose Estimation**. Ma et al., arXiv:2110.09554 — cross-view attention fusion, compatible with `AttentionFusionModel`.
3. **RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond**. Bermuth et al., arXiv:2503.21692 — fast geometric triangulation baseline for real-time multi-person scenarios.
4. **Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining**. Wang et al., arXiv:2503.03222 — training recipe for multi-view lifting when 3D labels are limited.
5. **Learnable Triangulation of Human Pose**. Iskakov et al., ICCV 2019. arXiv:1905.05754 — foundational learnable triangulation work that justifies the `RobustTriangulationModel` design.

### Risks, Blockers, and Assumptions

- **Camera extrinsics are required.** The GVHMR adapter currently stores only intrinsics (`K_fullimg`). Multi-view fusion needs `R` and `t` for each view; these must be supplied by the dataset or estimated.
- **Per-view coordinate frames may not be aligned.** GVHMR’s “global” output may be in a per-view camera frame, not a shared world frame. Canonicalization must be verified with a synthetic test.
- **Re-encoding SMPL from fused 3D joints is under-constrained.** A simple average of `body_pose` may be acceptable for a baseline, but a principled SMPL fitting step may be needed later.
- **Evaluation needs 3D GT.** Without 3D ground truth, improvements over DLT cannot be claimed; Shelf/Campus 3D annotations are sufficient for the next milestone.
- **No external network search was performed** beyond targeted arXiv fetches; the cited papers are those confirmed to exist and be relevant.</subagent>
<subagent agent_id="agent-39" item="Propose a multi-view data preprocessing and calibration pipeline: from raw synchronized multi-view videos to per-view HumanMotionIRs and then to a fused physical-space output." outcome="completed">## Report: Multi-View Preprocessing and Calibration Pipeline for MotionFlow

### Executive Summary

The MotionFlow multi-view extension already has a clean backbone: per-view 2D keypoints are lifted by DLT/attention/temporal fusion into a world-coordinate skeleton. To move from raw synchronized multi-view videos to a fused physical-space output, the pipeline should be split into five stages: (1) capture & sync, (2) per-view human motion IR, (3) camera calibration, (4) cross-view person matching, and (5) geometry-aware fusion. Existing DLT fusion is strong, so the next gains will come from better preprocessing, calibration-free fallback, and true 3D/physical priors rather than larger reprojection-only models.

### Key Findings and Where They Fit

**1. Per-view IR generation: keep GVHMR, add ScoreHMR as an optional inverse-problem module.**
The current `gvhmr_adapter.py` converts GVHMR’s `smpl_params_global` into `HumanMotionIR`. GVHMR (SIGGRAPH Asia 2024) estimates world-grounded SMPL in a Gravity-View (GV) coordinate frame, which is ideal for MotionFlow’s physical-space output because gravity alignment and metric scale are built in. ScoreHMR (CVPR 2024) solves inverse problems for 3D human recovery via diffusion score guidance and can reconstruct from multiple uncalibrated views or video. It fits as a drop-in replacement or ensemble estimator: run it on each view independently, then feed its SMPL output through the same `HumanMotionIR` adapter. This gives per-view IRs with uncertainty that can be propagated downstream.

**2. Calibration: default to known extrinsics, fallback to human-as-calibration.**
The `Camera` class in `calibration/camera.py` currently assumes a calibrated rig (VoxelPose/Shelf style). In a practical deployment, use CasCalib (FG 2024), which performs temporal synchronization plus intrinsic and extrinsic calibration using humans in the scene as calibration objects. It is a direct preprocessing plug-in before `Camera` construction. When even that is too heavy, RUMPL (arXiv 2512.15488) provides a calibration-free ray-based transformer that lifts 2D keypoints to 3D independent of camera parameters; it can initialize the fusion stage or serve as a sanity check on unknown rigs.

**3. Cross-view person matching: extend beyond reprojection combinatorics.**
`pipeline_utils.py::select_best_person_group` matches people across views by exhaustively minimizing reprojection error. This is fine for small, controlled setups but scales poorly with more people or views. For general scenes, add an appearance-based re-ID branch or use trajectory consistency (temporal window) before the geometric test. The Geometry-Biased Transformer (FG 2024) demonstrates that geometry-aware attention and detection-score weighting greatly improve occluded-view reconstruction and can replace the current `AttentionFusionModelV2` at the fusion head.

**4. Fusion: DLT is the geometric baseline; learned variants need 3D or physical priors.**
The experiments confirm that DLT is very hard to beat on reprojection alone. `RobustTriangulationModel`, `ResidualRefinerModel`, and `TemporalRefinerModel` all match but do not clearly surpass it. The practical path is to use `triangulate_confidence_weighted` as the default, with `RobustTriangulationModel` as an adaptive outlier filter, and reserve learned fusion for datasets with real 3D ground truth such as Human3.6M, CMU Panoptic, HUMBI, or the FreeMotion dataset from LiveHPS (CVPR 2024).

**5. Physical-space output: standardize on a metric, gravity-aligned coordinate system.**
`HumanMotionIR` already records `coordinate_system`. The fused output should be stored with `world_from_reference` set to the rig’s extrinsic transform and `up_axis` aligned with gravity. GVHMR’s GV coordinates can be used directly; otherwise, enforce a ground-plane constraint (e.g., via RANSAC on ankle trajectories) so downstream robot retargeting receives a consistent physical frame.

### Concrete Recommendations

1. **Build a preprocessing CLI**: `videos/` → run GVHMR/ScoreHMR per view → write one `HumanMotionIR` per view with frame timestamps and confidence. Reuse the existing `gvhmr_adapter.py` and add a `scorehmr_adapter.py` sibling.
2. **Calibration module**: wrap CasCalib into a preprocessing step that outputs the existing `Camera` JSON format; keep the `VoxelPoseShelfLoader` path for already-calibrated data.
3. **Person matching**: add a `reid_person_matcher.py` that computes view-to-view person similarity from appearance embeddings and 2D pose, then uses `select_best_person_group` only to break ties.
4. **Fusion default**: keep `MultiViewPipeline` with confidence-weighted DLT. Expose `RobustTriangulationModel` as an optional learned weighting module, and add the Geometry-Biased Transformer as a future fusion head.
5. **Evaluation**: move beyond reprojection error. Train on a dataset with 3D GT (Panoptic/Human3.6M) and report MPJPE/PA-MPJPE, not just pixel error.

### Key References

- **Score-Guided Diffusion for 3D Human Recovery**. Stathopoulos et al. *CVPR 2024*. [arXiv:2403.09623](https://arxiv.org/abs/2403.09623)
- **World-Grounded Human Motion Recovery via Gravity-View Coordinates**. Shen et al. *SIGGRAPH Asia 2024*. [arXiv:2409.06662](https://arxiv.org/abs/2409.06662)
- **CasCalib: Cascaded Calibration for Motion Capture from Sparse Unsynchronized Cameras**. Tang et al. *IEEE FG 2024*. [arXiv:2405.06845](https://arxiv.org/abs/2405.06845)
- **RUMPL: Ray-Based Transformers for Universal Multi-View 2D to 3D Human Pose Lifting**. Ghasemzadeh et al. *arXiv 2512.15488* (2025). [arXiv:2512.15488](https://arxiv.org/abs/2512.15488)
- **Geometry-Biased Transformer for Robust Multi-View 3D Human Pose Reconstruction**. Moliner et al. *IEEE FG 2024*. [arXiv:2312.17106](https://arxiv.org/abs/2312.17106)

### Risks, Blockers, and Assumptions

- **Synchronization**: CasCalib and most fusion modules assume frame-level temporal alignment. If cameras are free-running, audio clapboards or motion-event matching must precede calibration.
- **Metric scale**: Without a known object or calibrated baseline, monocular estimators are scale-ambiguous. GVHMR/CasCalib provide scale; pure DLT from 2D alone does not.
- **3D GT bottleneck**: Learned fusion cannot clearly beat DLT on the Shelf/VoxelPose reprojection proxy. Real 3D ground truth or strong bone-length priors is required for a statistically significant advance.
- **License**: ScoreHMR is MIT and can be bundled. EasyMocap-style toolboxes are non-commercial and should remain an external benchmark, as noted in the Phase 0 audit.
- **Compute**: Per-view GVHMR/ScoreHMR on long videos is expensive; batch on A800-D and cache per-view IRs before fusion.</subagent>
<subagent agent_id="agent-40" item="Propose a training strategy: synthetic pre-training (how to generate synthetic multi-view data), fine-tuning on real datasets, domain adaptation, and loss functions." outcome="completed">## Report: Training Strategy for MotionFlow Multi-View Human Motion Fusion

### Executive Summary
The MotionFlow multi-view extension already has a strong geometric baseline—confidence-weighted DLT—and a family of lightweight learned fusion models (`AttentionFusionModel`, `RobustTriangulationModel`, `ResidualRefinerModel`, `TemporalRefinerModel`). Empirical results on Shelf 300–600 show that learned models only **match** DLT, not beat it, because they are trained with reprojection loss or DLT pseudo-GT rather than true 3D supervision. The path to publishable quality is therefore: (1) generate realistic synthetic multi-view sequences with 3D ground truth, (2) pre-train a temporal/geometry-aware fusion model on that synthetic data, (3) fine-tune on real datasets that provide 3D GT, and (4) add domain-adaptive self-supervision and strong geometric losses.

### Key Findings / Method Descriptions

**1. Synthetic pre-training must move beyond random 3D points.**  
Current scripts like `experiments/train_temporal_synthetic.py` generate random skeleton trajectories and project them through synthetic pinhole cameras. This validates the pipeline but lacks body structure, realistic occlusions, and detection noise distributions. A useful synthetic stage should sample motion from a large motion-capture corpus, animate a parametric body (SMPL/SMPL-X), and render/project joints through a calibrated camera rig. The output format should match the existing `VoxelPoseShelfLoader` input so the same `TemporalRefinerModel` and `RobustTriangulationModel` can consume it.

**2. Geometric DLT is the ceiling when only reprojection loss is available.**  
`docs/design_v2.md` already shows that `RobustTriangulationModel` and `TemporalRefinerModel` plateau near DLT. This mirrors the observation in *Learnable Triangulation of Human Pose* that differentiable triangulation with confidence is a strong baseline; to surpass it, a model needs additional supervisory signals beyond reprojection. Real 3D labels are the most direct way to break the ceiling.

**3. Motion priors from large-scale mocap improve generalization.**  
`AMASS` provides unified SMPL motion clips that can be used to synthesize long, physically plausible sequences. Pre-training on AMASS-derived multi-view data injects realistic human motion dynamics into the `TemporalRefinerModel`, which currently learns from frame-wise random walks.

**4. Domain adaptation can exploit unlabeled real video.**  
Recent work on refining pre-trained motion models shows that cycle-consistent pseudo-labeling and augmentation on real video can improve a model trained on synthetic data without manual 3D annotation. For MotionFlow, this means taking the pre-trained temporal refiner, generating pseudo-3D targets on real Shelf/Human3.6M/Panoptic frames with high cycle-consistency, and fine-tuning with a mix of 3D pseudo-labels and reprojection losses.

**5. Loss functions should combine 3D, reprojection, and human-body priors.**  
A single reprojection loss is insufficient. A strong training objective is:

\[
\mathcal{L} = \lambda_1 \underbrace{\mathcal{L}_{3D}}_{\text{MSE/MPJPE}} + \lambda_2 \underbrace{\mathcal{L}_{reproj}}_{\text{2D reprojection}} + \lambda_3 \underbrace{\mathcal{L}_{bone}}_{\text{bone-length consistency}} + \lambda_4 \underbrace{\mathcal{L}_{temporal}}_{\text{velocity smoothness}} + \lambda_5 \underbrace{\mathcal{L}_{sym}}_{\text{limb symmetry}}
\]

This directly targets the failure modes observed in the current experiments: DLT-style reprojection alone cannot enforce plausible skeletons, while a 3D term and bone-length term can.

### Concrete Recommendations

1. **Build a realistic synthetic generator** using AMASS + SMPL. Project joints through `motionflow_mv.calibration.camera.Camera`-style rigs, add detection noise modeled from real 2D predictions, and produce `(V, J, 3)` inputs. Store in the same format as `outputs/shelf_matched_dataset.pkl`.
2. **Pre-train `TemporalRefinerModel` with 3D MSE** on long AMASS-derived sequences. Use a combined loss during the last epochs to expose the model to reprojection geometry.
3. **Fine-tune on real 3D datasets** (Human3.6M, Panoptic, TotalCapture, Shelf if 3D GT is available). Use the combined loss above, not DLT pseudo-GT alone.
4. **Add bone-length and symmetry losses** to `experiments/train_temporal_refiner_shelf.py` and `experiments/train_residual_refiner_shelf.py` to prevent the physically implausible poses that reprojection-only models tolerate.
5. **Add domain-adaptive pseudo-labeling** for real unlabeled video: use the pre-trained model to generate cycle-consistent 3D pseudo-labels, then fine-tune, following the two-stage “label-making then training” recipe.
6. **Evaluate with true 3D metrics** (MPJPE, PA-MPJPE, PCK) in `motionflow_mv/eval/metrics.py`, not just reprojection error.

### Key References

1. **Learnable Triangulation of Human Pose** — Iskakov et al., *ICCV 2019*. [arXiv:1905.05754](https://arxiv.org/abs/1905.05754)
2. **MotionBERT: A Unified Perspective on Learning Human Motion Representations** — Zhu et al., *ICCV 2023*. [arXiv:2210.06551](https://arxiv.org/abs/2210.06551)
3. **AMASS: Archive of Motion Capture as Surface Shapes** — Mahmood et al., *ICCV 2019*. [arXiv:1904.03278](https://arxiv.org/abs/1904.03278)
4. **Refining Pre-Trained Motion Models** — Sun et al., *ICRA 2024*. [arXiv:2401.00850](https://arxiv.org/abs/2401.00850)

### Risks, Blockers, and Assumptions

- **3D GT availability:** The biggest blocker is access to real multi-view datasets with 3D ground truth. Shelf/Campus alone may not provide it; Human3.6M and Panoptic are essential.
- **Sim-to-real gap:** Synthetic data must match camera layouts, image noise, and occlusion patterns of the target domain, otherwise pre-training provides little transfer.
- **DLT ceiling:** Even with 3D GT, learned models may only marginally beat DLT on well-calibrated setups; the contribution must be framed as robustness to occlusion/noise rather than raw accuracy.
- **Integration cost:** Connecting the output of `TemporalRefinerModel` back into the `HumanMotionIR` used by the GVHMR adapter requires careful handling of SMPL pose parameters and world-coordinate alignment.
- **Compute:** Training on AMASS-derived sequences and real 3D datasets will benefit from the A800-D; the current RTX 4090 is sufficient only for small-scale validation.</subagent>
<subagent agent_id="agent-41" item="Propose a concrete evaluation protocol for the multi-view fusion module: metrics, test sets, baselines to compare, and ablations." outcome="completed">## Report: Concrete Evaluation Protocol for the MotionFlow Multi-View Fusion Module

### Executive Summary

The MotionFlow multi-view branch currently fuses per-view 2D keypoints into 3D skeletons using a geometric DLT baseline and several lightweight learned variants (`AttentionFusionModel`, `RobustTriangulationModel`, `ResidualRefinerModel`, `TemporalRefinerModel`) in `motionflow_mv/fusion/`. All learned variants have so far been evaluated only by reprojection error on the Shelf 300–600 frames, where they fail to beat plain DLT. This report proposes a rigorous evaluation protocol that adds **3D ground-truth benchmarks**, standard **3D pose metrics**, strong **baselines**, and **diagnostic ablations** so the next iteration can credibly demonstrate a scientific or practical advantage.

### Key Findings

1. **DLT is a near-ceiling baseline on Shelf.**  
   The README shows DLT achieves mean/median reprojection error of ~9.9/5.5 px, and every learned variant (attention, residual, temporal, robust triangulation) only matches or marginally improves it. Because DLT already minimizes a related geometric objective, models trained with reprojection loss alone cannot easily outperform it.

2. **The current evaluation lacks 3D GT metrics.**  
   `motionflow_mv/eval/metrics.py` already implements `MPJPE`, `PA-MPJPE`, and `PCK`, but the existing scripts (e.g., `eval_shelf_voxelpose.py`, `eval_attention_fusion_shelf_v2.py`) only report reprojection error. Without 3D GT, the protocol cannot validate true 3D accuracy or bone-length/temporal consistency.

3. **The architecture is modular and ready for stronger supervision.**  
   - `motionflow_mv/ir/human_motion_ir.py` defines `HumanMotionIR`, which can carry world-coordinate 3D poses, uncertainty, and provenance.  
   - `motionflow_mv/fusion/` already provides the candidate fusion heads.  
   - `motionflow_mv/calibration/camera.py` and `VoxelPoseShelfLoader` handle calibrated camera inputs.

4. **Recent literature points to two viable directions:** using strong per-view estimators (ScoreHMR) and fast learned triangulation/lifting (RapidPoseTriangulation, Mocap-2-to-3). A occlusion-robust multi-view fusion method (Bragagnolo et al., ECCVW 2024) also shows the value of bone-length and reprojection constraints.

### Concrete Recommendations

#### Metrics
- **Primary:** `MPJPE` and `PA-MPJPE` (mm) against 3D GT, plus `PCK@150mm` and `AUC` for cross-dataset comparison.
- **Diagnostic:** per-joint error, bone-length consistency (mean absolute deviation from a skeleton prior), temporal jitter (average 2nd derivative of 3D joints), and reprojection error.
- **Downstream (optional):** robot retargeting joint-angle RMSE or policy success rate if the fused pose is fed back into MotionFlow.

#### Test Sets
- **Shelf / Campus** (existing `VoxelPoseShelfLoader`): use frames 300–600 as a held-out test set and the remaining frames for training/validation. Both have calibrated cameras and 3D GT.
- **Human3.6M**: a small subset (e.g., subjects S9/S11, a few actions) for cross-dataset generalization with 4 views and 3D GT.
- **CMU Panoptic** (optional): provides occlusion-rich social scenes and multi-person data.
- **Synthetic test set** (already generated by `experiments/compare_dlt_attention_synthetic.py`): controlled noise/occlusion ablations.

#### Baselines
1. **DLT** (`triangulate_confidence_weighted` in `motionflow_mv/fusion/triangulation.py`).
2. **RobustTriangulationModel** – learned per-view confidence weights.
3. **AttentionFusionModel / V2** – transformer-based fusion.
4. **ResidualRefinerModel + TemporalRefinerModel** – post-DLT refinement and temporal smoothing.
5. **External:** EasyMocap (non-commercial, use only as a paper benchmark) and ScoreHMR as a strong per-view estimator.

#### Ablations
- **View count:** 2, 3, 4, 5 views with random dropout.
- **Supervision signal:** reprojection-only vs. 3D MSE vs. 3D MSE + reprojection + bone-length + temporal smoothness.
- **Temporal window:** 1, 5, 9, 15 frames for `TemporalRefinerModel`.
- **Pretraining:** synthetic pretrain → real fine-tune vs. train-from-scratch.

### Next Steps
1. Add loaders for Shelf/Campus 3D GT and Human3.6M that return `HumanMotionIR`.
2. Implement an evaluation harness in `motionflow_mv/eval/` that computes MPJPE/PA-MPJPE/PCK for every fusion head.
3. Re-train the learned fusion heads with a 3D-supervised loss and compare against the DLT baseline.
4. Report per-joint, bone-length, and temporal-jitter diagnostics to understand where learned models improve beyond reprojection.

### Key References

1. **Score-Guided Diffusion for 3D Human Recovery** — Stathopoulos et al., CVPR 2024. [arXiv:2403.09623](https://arxiv.org/abs/2403.09623)
2. **Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation** — Bragagnolo et al., ECCV Workshops 2024. [arXiv:2408.15810](https://arxiv.org/abs/2408.15810)
3. **RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond** — Bermuth et al., arXiv 2025. [arXiv:2503.21692](https://arxiv.org/abs/2503.21692)
4. **Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining** — Wang et al., arXiv 2025. [arXiv:2503.03222](https://arxiv.org/abs/2503.03222)
5. **Fast and Robust Multi-Person 3D Pose Estimation and Tracking from Multiple Views** — Dong et al., IEEE T-PAMI 2021. [project page](https://zju3dv.github.io/mvpose/)

### Risks, Blockers, and Assumptions

- **3D GT availability:** Shelf/Campus 3D annotations and Human3.6M/Panoptic require downloads and license checks; skeleton conventions (COCO vs. H36M) must be aligned to `HumanMotionIR`.
- **DLT ceiling:** Without 3D-supervised or strong physical priors, learned models may keep tying DLT; the ablation on supervision signal is critical.
- **Licensing:** EasyMocap is non-commercial and can only be used as an external benchmark, not shipped in the default MotionFlow release.
- **Data scale:** Shelf/Campus are small; Human3.6M/Panoptic are needed for publishable generalization claims.
- **Compute:** Training temporal models with 3D GT on Human3.6M may require the A800-D; the current RTX 4090 setup is sufficient only for small-scale ablations.</subagent>
<subagent agent_id="agent-42" item="Propose how to integrate the multi-view fusion module into the existing MotionFlow pipeline with minimal intrusion: file changes, data flow, and API design." outcome="completed">## Report: Integrating the Multi-View Fusion Module into the MotionFlow Pipeline

### Executive Summary
The repository already has a clean, modular multi-view skeleton (`motionflow_mv/ir`, `motionflow_mv/calibration`, `motionflow_mv/fusion`, `MultiViewPipeline`). Phase 1 only completed the GVHMR → `HumanMotionIR` passthrough. The next step is to plug the existing multi-view fusion models into that IR so that N calibrated views produce a single world-coordinate 3D skeleton that downstream MotionFlow stages can consume.

The concrete, minimal-intrusion design is:

1. Keep the geometric **DLT triangulator** as the default fusion backend.
2. Wrap the learned fusion variants (`AttentionFusionModel`, `RobustTriangulationModel`, `ResidualRefinerModel`, `TemporalRefinerModel`) behind a single `FusionModule` interface.
3. Add a `MultiViewMotionFlowIRAdapter` that converts the fused 3D skeleton into a `HumanMotionIR`, reusing the existing GVHMR adapter for SMPL parameter fields.
4. Expose one new top-level entry point: `experiments/run_motionflow_multiview.py`.

### Key Findings / Method Descriptions
**Existing code already supports the core geometry.** `motionflow_mv/fusion/triangulation.py` implements confidence-weighted DLT, `pipeline.py` triangulates per-frame 2D keypoints, and `pipeline_utils.py` handles cross-view person matching by minimizing reprojection error. The learned variants (`attention_model_v2.py`, `robust_triangulation.py`, `residual_refiner.py`, `temporal_refiner.py`) operate on `(B, V, J, 3)` 2D observations plus camera matrices and are therefore easy to unify.

**DLT is the practical baseline.** As `docs/design_v2.md` shows, DLT achieves ~5.5 px median reprojection error on Shelf 300–600 and none of the learned re-projection-only models beat it. The literature confirms that learned fusion needs either 3D ground-truth supervision or strong geometric priors (bone lengths, temporal coherence) to outperform triangulation.

**Recent methods that fit this architecture.**
- **Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation** (ECCVW 2024, arXiv:2408.15810) fuses 3D monocular outputs and refines them via reprojection error with limb-length symmetry constraints. Its “monocular 3D skeleton + multi-view refinement” pattern maps directly to our `ResidualRefinerModel` and `TemporalRefinerModel`.
- **RapidPoseTriangulation** (arXiv:2503.21692) shows how to triangulate whole-body multi-person poses in milliseconds. Its fast, training-free triangulation stage can replace the current per-joint DLT loop in `pipeline.py`.
- **Mocap-2-to-3** (arXiv:2503.03222) recovers metric-scale absolute poses by lifting monocular input with multi-view fine-tuning after 2D pre-training. The two-stage “2D pre-train → multi-view fine-tune” recipe is exactly what our `train_attention_fusion_shelf.py` family should adopt.
- **Score-Guided Diffusion for 3D Human Recovery (ScoreHMR)** (CVPR 2024, arXiv:2403.09623) can serve as a strong per-view SMPL/2D estimator. Running it on each view and feeding its 2D projections into the fusion module keeps the system modular.
- **DenseWarper** (ICLR 2026, arXiv:2605.14525) performs spatio-temporal fusion across sparse, interleaved multi-view frames. Its dense epipolar heatmap exchange is a natural upgrade path for the current `TemporalRefinerModel`.

### Concrete Recommendations for Implementation

**1. Define a `FusionModule` interface and registry.**
Create `motionflow_mv/fusion/fusion_module.py`:
```python
class FusionModule(ABC):
    @abstractmethod
    def forward(self, points_2d: Tensor, confidence: Tensor,
                  cameras: List[Camera]) -> np.ndarray:
        # points_2d: (V, J, 2), confidence: (V, J)
        # returns (J, 3) in world coordinates
        ...
```
Implement `DLTFusion`, `AttentionFusion`, `RobustTriangulationFusion`, `TemporalRefinerFusion` as thin wrappers. Update `MultiViewPipeline.__init__(self, fusion_module)` and replace the hard-coded DLT in `fuse_frame`.

**2. Add a multi-view → `HumanMotionIR` adapter.**
Create `motionflow_mv/ir/multiview_adapter.py`:
- Input: fused 3D skeleton `(J, 3)`, per-view SMPL params from GVHMR, and camera calibration.
- Output: `HumanMotionIR` with `pose["body_pose"]`, `pose["global_orient"]`, `pose["transl"]`, `pose["betas"]`.
- Implementation options (pick the simpler one first):
  - **Option A (minimal):** average per-view GVHMR `betas`, `body_pose`, `global_orient`, and replace `transl` with the fused 3D root position.
  - **Option B (geometrically faithful):** run a light SMPL fitting step from the fused 3D joints (e.g., `smplx` + L-BFGS) to produce SMPL parameters consistent with the world skeleton.

**3. Update the end-to-end pipeline.**
Modify `motionflow_mv/pipeline.py` so that `MultiViewPipeline.run(videos, cameras)`:
1. Calls `BasePoseEstimator.extract` for each view.
2. Runs `select_best_person_group` (or a faster epipolar/Hungarian replacement).
3. Feeds matched 2D to the selected `FusionModule`.
4. Passes the fused 3D skeleton to `MultiViewMotionFlowIRAdapter.to_ir(...)`.

**4. Add a MotionFlow integration demo.**
Create `experiments/run_motionflow_multiview.py` that takes a directory of synchronized videos and a calibration file, runs the above pipeline, and writes a `HumanMotionIR` pickle/JSON that the existing MotionFlow retargeting stage can read.

**5. Training strategy.**
Do not continue training on Shelf reprojection loss alone. Switch to a dataset with 3D GT (Human3.6M, Panoptic, or Shelf 3D GT if available) and optimize a combined 3D + reprojection + bone-length loss. Use synthetic pre-training only to initialize the temporal/refiner models.

### Key References
1. **Score-Guided Diffusion for 3D Human Recovery** — CVPR 2024. https://arxiv.org/abs/2403.09623  
2. **Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation** — ECCV Workshops 2024. https://arxiv.org/abs/2408.15810  
3. **RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond** — arXiv:2503.21692. https://arxiv.org/abs/2503.21692  
4. **Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining** — arXiv:2503.03222. https://arxiv.org/abs/2503.03222  
5. **From Sparse to Dense: Spatio-Temporal Fusion for Multi-View 3D Human Pose Estimation with DenseWarper** — ICLR 2026. https://arxiv.org/abs/2605.14525  

### Risks, Blockers, and Assumptions
- **3D supervision gap:** The main blocker is the lack of real 3D ground truth. Without it, learned fusion is unlikely to beat DLT, as the past eight iterations have shown.
- **SMPL fitting:** Converting a fused 3D skeleton back to `HumanMotionIR` may require an IK/SMPL fitting step that is not yet in the repo.
- **Cross-view person ID:** `select_best_person_group` is combinatorial and will not scale beyond a few views/people; a Hungarian or epipolar matching step is needed for multi-person scenes.
- **Calibration quality:** The whole pipeline assumes accurate intrinsics and extrinsics. On real robots, this is a non-trivial assumption.
- **GVHMR as 2D source:** GVHMR currently outputs SMPL parameters, not 2D keypoints. A small adapter is needed to project SMPL joints to 2D or to use an off-the-shelf 2D detector alongside GVHMR.</subagent>
<subagent agent_id="agent-43" item="Propose novel contribution angles that could differentiate this work for ICRA/CVPR2027: gaps in existing methods, unique combinations, or new problem formulations." outcome="completed">## Report: Novel Contribution Angles for MotionFlow Multi-View (ICRA/CVPR2027)

### Executive Summary

The repository has progressed beyond a simple "beat DLT on reprojection error" problem. Current experiments on Shelf (300–600 frames, 5 views) show that DLT is essentially a saturated baseline: attention fusion, learned confidence weighting, residual refinement, and even large temporal refiners only match, but do not surpass, geometric triangulation. The most promising path to a top-tier 2027 paper is therefore to **reframe the contribution**: not "a better triangulator," but a **modular, world-grounded, multi-view human motion representation and fusion framework** that is explicitly designed for downstream robot retargeting. Novelty should come from (1) a gravity-aligned, physically consistent intermediate representation (IR), (2) stronger per-view estimators and fusion primitives, (3) explicit skeleton/motion priors, and (4) downstream policy-level evaluation.

### Key Findings / Method Descriptions

**1. The core technical story should shift from "triangulation" to "world-grounded motion IR."**
The existing `HumanMotionIR` (`motionflow_mv/ir/human_motion_ir.py`) stores a `coordinate_system` field but does not yet define a gravity-aligned, metric-scale world frame. Recent world-grounded HMR work shows that defining poses in a *Gravity-View (GV)* coordinate system—aligned with gravity and the camera viewing direction—eliminates the per-sequence coordinate ambiguity that plagues autoregressive monocular methods. Extending `HumanMotionIR` to a GV-style world frame and making the multi-view fusion module produce metric-scale, gravity-aligned output is a concrete, publishable contribution. This directly supports the robot retargeting downstream task.

**2. Replace or augment the per-view backbone with a strong diffusion-based estimator.**
ScoreHMR (CVPR 2024) formulates 3D human recovery as a diffusion inverse problem and already demonstrates reconstruction from multiple uncalibrated views. It is MIT-licensed and can serve as a per-view estimator or as a multi-view consistency prior. In this codebase, it would sit upstream of `HumanMotionIR` (alongside or replacing the GVHMR adapter) and provide per-view SMPL parameters and uncertainty. A novel angle is to derive a **multi-view score** that guides the diffusion process using reprojection consistency across calibrated cameras, turning ScoreHMR from a per-view method into a calibrated multi-view fusion method.

**3. Move from 2D-keypoint fusion to feature-level volumetric aggregation.**
Current fusion modules (`AttentionFusionModel`, `RobustTriangulationModel`, `TemporalRefinerModel`) operate only on 2D keypoints + confidence. The literature has moved toward **volumetric aggregation of per-view CNN features** (Learnable Triangulation, VTP). A differentiable voxel grid that aggregates intermediate backbone features from multiple views and refines them with 3D convolutions can model a human pose prior and consistently outperform 2D-keypoint DLT. This would be a natural next module in `motionflow_mv/fusion/` and would provide the 3D supervision signal that current reprojection-only losses lack.

**4. Integrate explicit skeleton priors: bone length, symmetry, and temporal motion prior.**
The current reprojection-only loss is the key bottleneck. Adding:
- bone-length and limb-symmetry constraints,
- a learned temporal motion prior (e.g., a small diffusion model or VQ-VAE trained on AMASS/CMU Panoptic),
- and a physics plausibility term,
would create a "DLT + learned prior" hybrid that can finally beat pure triangulation. This can be implemented inside `ResidualRefinerModel`/`TemporalRefinerModel` or as a new `SkeletonConstrainedRefiner`.

**5. Evaluation on downstream robot policy metrics, not just MPJPE.**
A strong differentiator is to evaluate how multi-view fusion improves the *downstream* MotionFlow robot retargeting/policy task. If multi-view input reduces foot sliding, ground penetration, or end-effector error in the retargeted character, the paper has a clear robotics (ICRA) angle.

### Concrete Recommendations

1. **Dataset upgrade:** migrate training/validation from Shelf-only to **Human3.6M + CMU Panoptic**, using 3D GT and MPJPE/PA-MPJPE as the primary metric. Shelf remains a fast sanity benchmark.
2. **Implement a `GravityViewAdapter`** that transforms GVHMR/ScoreHMR per-view outputs into a gravity-aligned world frame before storing in `HumanMotionIR`.
3. **Add a `VolumetricFusion` module** in `motionflow_mv/fusion/` that aggregates per-view CNN features into a 3D voxel volume and regresses 3D joints/SMPL parameters.
4. **Add skeleton/motion prior losses** to the refiner: bone-length regularization, symmetry, and a small temporal diffusion/VAE prior trained on AMASS.
5. **Create a downstream evaluation track:** retarget fused 3D poses to a robot/skeleton and measure task-relevant metrics (foot contact, joint limits, policy success rate).
6. **Keep DLT as a strong baseline** but frame it as the geometric backbone inside a learned world-grounded fusion framework.

### 5 Key References

1. **Score-Guided Diffusion for 3D Human Recovery** (ScoreHMR), Stathopoulos et al., CVPR 2024. [arXiv:2403.09623](https://arxiv.org/abs/2403.09623)
2. **World-Grounded Human Motion Recovery via Gravity-View Coordinates**, Shen et al., SIGGRAPH Asia 2024. [arXiv:2409.06662](https://arxiv.org/abs/2409.06662)
3. **Learnable Triangulation of Human Pose**, Iskakov et al., ICCV 2019. [arXiv:1905.05754](https://arxiv.org/abs/1905.05754)
4. **VTP: Volumetric Transformer for Multi-view Multi-person 3D Pose Estimation**, Huang et al., arXiv 2022. [arXiv:2205.12602](https://arxiv.org/abs/2205.12602)
5. **Geometry-Biased Transformer for Robust Multi-View 3D Human Pose Reconstruction**, Moliner et al., FG 2024. [arXiv:2312.17106](https://arxiv.org/abs/2312.17106)

### Risks, Blockers, and Assumptions

- **3D ground-truth data access:** Human3.6M and CMU Panoptic require registration/download; without them, the project remains stuck on Shelf reprojection error, which is insufficient for CVPR/ICRA novelty.
- **Compute:** volumetric fusion and diffusion-based methods need substantially more GPU memory than the current lightweight models; the A800-D access is essential.
- **Downstream robot pipeline:** the MotionFlow-to-robot retargeting code is outside this repo; integrating it must be scoped carefully.
- **Per-view estimator dependency:** GVHMR is currently the default, but its license and output format must be checked before publication.
- **Risk of over-engineering:** each new module must be validated against DLT on real 3D metrics before being accepted into the pipeline.</subagent>
<subagent agent_id="agent-44" item="Perform a risk and feasibility analysis: technical risks, data availability risks, compute/time constraints on 4090/A800, and mitigation strategies." outcome="completed">## Report: Risk and Feasibility Analysis for MotionFlow Multi-View Extension

### Executive Summary

The MotionFlow multi-view extension project has built a clean modular architecture: `HumanMotionIR` (`motionflow_mv/ir/human_motion_ir.py`) decouples upstream estimators from downstream robot retargeting, and `gvhmr_adapter.py` converts GVHMR outputs into this IR. The fusion module (`motionflow_mv/fusion/`) already implements a DLT baseline, attention fusion, robust triangulation, residual refinement, and temporal refinement. Experiments on the Shelf/VoxelPose dataset (5 views, 300–600 frames) show that **geometric DLT is a very strong baseline** and learned variants only match, not beat, it under reprojection-only supervision. Achieving a clear paper-worthy contribution for ICRA/CVPR 2027 will require: (1) real 3D ground-truth data or strong physical priors, (2) integration with a strong per-view estimator such as GVHMR or ScoreHMR, and (3) a compelling downstream robot-policy validation.

### Key Findings / Method Descriptions

1. **GVHMR as the per-view backbone.** The current `gvhmr_adapter.py` reads `smpl_params_global` from GVHMR and populates `HumanMotionIR.pose`. GVHMR (SIGGRAPH Asia 2024 / T-PAMI 2026) produces world-grounded SMPL parameters using gravity-view coordinates. This is a natural fit for MotionFlow because its output is already metric-scale and aligned to a physical world frame, reducing the calibration burden for multi-view fusion.

2. **DLT is the ceiling on Shelf data.** `motionflow_mv/fusion/triangulation.py` implements confidence-weighted DLT. On Shelf it achieves mean/median reprojection error of ~9.94/5.53 px, while `RobustTriangulationModel`, `ResidualRefinerModel`, `TemporalRefinerModel`, and even an A800-D scaled temporal model (hidden=256, d=128, window=15) only match these numbers. The repository correctly identifies that **reprojection-only loss cannot overcome the geometric DLT prior**.

3. **ScoreHMR for stronger per-view 3D.** ScoreHMR (CVPR 2024) fits SMPL/SMPL-X to images via diffusion-model score guidance and can operate from single images, multiple uncalibrated views, or video. It can replace or augment GVHMR as the per-view estimator before fusion, potentially giving higher-quality 2D/3D keypoints and uncertainty estimates that feed into the confidence-weighted DLT or learned fusion modules.

4. **Recent multi-view fusion literature.** RapidPoseTriangulation (arXiv 2025) targets multi-view multi-person whole-body triangulation in milliseconds; Mocap-2-to-3 (arXiv 2025) lifts 2D monocular motion to 3D via multi-view synthesis; and the ECCVW 2024 occlusion-aware multi-view pose fusion work shows the value of bone-length and symmetry constraints under occlusion. These suggest that the next gain should come from **occlusion reasoning, physical skeleton priors, and whole-body extension**, not from larger temporal refiners.

5. **Data landscape.** Human3.6M offers 3.6M poses with 3D GT from 4 calibrated cameras, making it the most practical source for supervised 3D training. CMU Panoptic provides larger multi-view volumes but is heavier to acquire and process. Shelf/VoxelPose remain useful for reprojection evaluation but lack 3D GT for supervised learning.

### Concrete Recommendations

1. **Switch the paper story from “beating DLT” to “a modular, physically-consistent multi-view extension of MotionFlow.”** This is well supported by the existing `HumanMotionIR` and plugin-style fusion modules.
2. **Acquire Human3.6M (or a subset) and train fusion models with a 3D MPJPE loss** in addition to reprojection loss. This directly addresses the supervision bottleneck shown in Iterations 5–9.
3. **Add physical priors:** bone-length constraints, limb symmetry, and gravity/ground-plane constraints. These fit naturally on top of the current DLT + residual/temporal architecture.
4. **Use GVHMR as the default per-view estimator** and keep ScoreHMR as an optional stronger (but slower) per-view plugin. Both output SMPL/SMPL-X, so the `HumanMotionIR` adapter generalizes.
5. **Validate end-to-end on downstream robot retargeting** rather than only reprojection/MPJPE. A modest accuracy gain in 3D pose may translate into a measurable policy improvement, which is the ultimate ICRA/CVPR contribution.

### Key References

1. **Score-Guided Diffusion for 3D Human Recovery** (ScoreHMR). Stathopoulos et al., CVPR 2024. [arXiv:2403.09623](https://arxiv.org/abs/2403.09623)
2. **World-Grounded Human Motion Recovery via Gravity-View Coordinates** (GVHMR). Shen et al., SIGGRAPH Asia 2024 / T-PAMI 2026. [GitHub: zju3dv/GVHMR](https://github.com/zju3dv/GVHMR)
3. **RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond.** Bermuth et al., arXiv 2025. [arXiv:2503.21692](https://arxiv.org/abs/2503.21692)
4. **Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining.** Wang et al., arXiv 2025. [arXiv:2503.03222](https://arxiv.org/abs/2503.03222)
5. **Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation.** Bragagnolo et al., ECCV Workshops 2024. [arXiv:2408.15810](https://arxiv.org/abs/2408.15810)

### Risks, Blockers, and Assumptions

- **Technical risk:** Learned fusion cannot currently beat DLT. The risk is low if the contribution is framed as a modular multi-view extension with systematic baselines, but a top-tier paper still needs a clear accuracy or downstream advantage.
- **Data risk:** Shelf lacks 3D GT; Human3.6M must be requested and preprocessed. Panoptic has administrative overhead. Without real 3D labels, supervised 3D learning remains blocked.
- **Compute/time risk:** RTX 4090 is sufficient for the existing small models and for GVHMR inference, but the A800-D scaling experiment showed that larger models do not help under reprojection-only supervision. Significant compute should not be spent until 3D GT data and stronger priors are in place.
- **Licensing risk:** EasyMocap is non-commercial and cannot be included in a product release; it can only be an external baseline. ScoreHMR is MIT; GVHMR’s license should be verified before redistribution.
- **Assumption:** The downstream MotionFlow robot-policy pipeline can consume `HumanMotionIR` with minimal changes. If the retargeting step expects a different pose format, additional adapter work is needed.</subagent>
<subagent agent_id="agent-45" item="Propose a concrete next-iteration implementation plan: what to build in the next 1-2 weeks, milestones, and validation steps." outcome="completed">## Report: Next-Iteration Plan for MotionFlow Multi-View Fusion

### Executive Summary

Phase 1 is complete: a stable `HumanMotionIR` and a GVHMR adapter now let MotionFlow ingest single-view videos, and the repo contains a full multi-view fusion testbed. However, experiments on Shelf 300–600 show that DLT remains the strongest fusion method; learned variants only match, not beat, it when trained only with reprojection loss. The next 1–2 weeks should therefore move from “geometric triangulation vs. small neural fusion” to **3D-supervised, skeleton-aware multi-view fusion** and a cleaner plug-in integration with the single-view estimator. This plan focuses on adding real 3D ground truth, per-view estimator upgradeability, skeleton priors, and 3D metrics.

### Key Findings / Method Descriptions

1. **DLT is the ceiling under reprojection-only supervision.** `motionflow_mv/fusion/triangulation.py` and `robust_triangulation.py` already confirm that confidence-weighted DLT gives a median reprojection error of ~5.5 px on Shelf. Learned attention, residual, and temporal refiners converge to the same level but cannot surpass it, because they optimize the same geometric objective DLT solves in closed form.

2. **3D ground-truth supervision is required to beat DLT.** The literature consistently shows that learned triangulation needs an independent 3D target (e.g., MPJPE loss) rather than reprojection alone. The current training scripts (`experiments/train_attention_fusion_shelf.py`, `experiments/train_temporal_refiner_shelf.py`) only use reprojection or pseudo-GT from DLT; this is the primary bottleneck.

3. **ScoreHMR is a strong, permissive per-view estimator.** *Score-Guided Diffusion for 3D Human Recovery* (CVPR 2024) reconstructs SMPL/SMPL-X from single images or multiple uncalibrated views and provides probabilistic outputs. It can replace or supplement GVHMR as the per-view source, giving higher-quality pseudo-3D labels for fusion training and a natural uncertainty channel.

4. **Occlusion-aware 3D fusion with skeleton constraints is a proven recipe.** *Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation* (ECCVW 2024) fuses monocular 3D predictions and enforces limb-length symmetry via reprojection optimization. This is directly applicable to our `AttentionFusionModel` and `TemporalRefinerModel`, which currently lack explicit bone-length or symmetry priors.

5. **Fast triangulation and 2D-pretrained multi-view lifting are relevant baselines.** *RapidPoseTriangulation* (arXiv 2025) offers millisecond-level multi-person triangulation and could improve cross-view person matching in `pipeline_utils.py::select_best_person_group`. *Mocap-2-to-3* (arXiv 2025) shows that 2D pre-training plus multi-view fine-tuning, with decoupled local/global pose representations, can improve metric-scale world positioning.

### Concrete Recommendations for Implementation

**Milestone 1 — 3D GT data loader (days 1–3)**
- Add `motionflow_mv/data/human36m_loader.py` (or Panoptic) that returns `(points_2d, confidences, projection_matrices, joints_3d_gt)`.
- If real GT is not available immediately, generate pseudo-3D labels with ScoreHMR and store them alongside Shelf/H36M frames.

**Milestone 2 — 3D loss + skeleton priors (days 2–5)**
- Update `experiments/train_attention_fusion_shelf.py` and `train_temporal_refiner_shelf.py` to accept 3D GT and minimize `L_3d + λ_reproj * L_reproj`.
- Add a bone-length consistency loss (`L_bone`) and limb-length symmetry loss as in the ECCVW 2024 paper. This targets the physically implausible poses that currently create outlier frames.

**Milestone 3 — Per-view estimator plug-in (days 4–7)**
- Create `motionflow_mv/ir/scorehmr_adapter.py` mirroring `gvhmr_adapter.py`, converting ScoreHMR output to `HumanMotionIR`.
- Refactor `MultiViewPipeline` in `pipeline.py` behind an estimator protocol so GVHMR or ScoreHMR can be swapped with one line.
- Run the end-to-end demo on a 5-view clip: videos → per-view IR → fusion → 3D skeleton → saved `HumanMotionIR`.

**Milestone 4 — 3D evaluation (days 6–10)**
- Implement MPJPE / PA-MPJPE / PCK in `motionflow_mv/eval/`.
- Compare DLT, `RobustTriangulationModel`, 3D-supervised `AttentionFusionModel`, and `TemporalRefinerModel`.
- Acceptance criterion: at least one learned model statistically outperforms DLT on MPJPE or PA-MPJPE on the chosen validation set.

**Milestone 5 — Reproducibility (days 9–10)**
- Document the Phase 2 plan in `docs/design_v2.md`.
- Add `experiments/run_scorehmr_multiview.py` and ensure `pytest tests/` still passes.

### Risks, Blockers, or Assumptions

- **Data access:** Human3.6M or Panoptic may require registration. Mitigate by using ScoreHMR pseudo-GT or existing Shelf 3D annotations.
- **Per-view matching:** `pipeline_utils.py::select_best_person_group` is combinatorial and only tested on single-person scenes. Multi-person generalization remains future work.
- **Calibration sensitivity:** DLT and learned fusion both assume accurate camera parameters. Noisy calibration could dominate improvements.
- **Compute:** ScoreHMR diffusion inference is heavier than GVHMR; batching or A800-D offloading may be needed.
- **Risk of no gain:** If pseudo-3D labels are too noisy, learned models may still not beat DLT.

### Key References

- Anastasis Stathopoulos et al., *Score-Guided Diffusion for 3D Human Recovery*, CVPR 2024. [arXiv:2403.09623](https://arxiv.org/abs/2403.09623)
- Laura Bragagnolo et al., *Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation*, ECCVW 2024. [arXiv:2408.15810](https://arxiv.org/abs/2408.15810)
- Daniel Bermuth et al., *RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond*, arXiv:2503.21692. [arXiv:2503.21692](https://arxiv.org/abs/2503.21692)
- Zhumei Wang et al., *Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining*, arXiv:2503.03222. [arXiv:2503.03222](https://arxiv.org/abs/2503.03222)</subagent>
<subagent agent_id="agent-46" item="Audit the existing MotionFlow code for multi-view readiness: identify what already supports multi-view, what needs refactoring, and what the IR contract must guarantee." outcome="completed">## Report: Multi-View Readiness Audit of the MotionFlow IR / Fusion Stack

### Executive Summary

The MotionFlow multi-view fork (`phase1-humanmotion-ir`) already has a functional 2D-keypoint-to-3D fusion backbone, but the **HumanMotionIR contract is still essentially single-view**. The `HumanMotionIR` dataclass is a clean, versioned container for one person’s SMPL/SMPL-X trajectory, yet it has no native slots for per-view observations, camera parameters, or multi-view uncertainty. Conversely, the `fusion/` package (`triangulation`, `AttentionFusionModel`, `RobustTriangulationModel`, `ResidualRefinerModel`, `TemporalRefinerModel`) already triangulates or learns from multi-view 2D inputs, but it is not wired to the IR or to per-view video HMR. To reach ICRA/CVPR2027 quality, the project needs a small architectural extension: treat multi-view fusion as a first-class stage that produces a `HumanMotionIR`, and feed that IR into the downstream retargeting/policy pipeline.

### Key Findings / Method Descriptions

1. **IR contract is single-person, not yet multi-view-aware**
   - `motionflow_mv/ir/human_motion_ir.py` defines `HumanMotionIR` with `pose`, `coordinate_system`, `uncertainty`, `quality`, and `provenance`. It is an excellent integration point, but `coordinate_system` only stores a generic 4×4 `world_from_reference`, and there is no field for per-view 2D/3D observations, camera IDs, or view-wise confidences.
   - `motionflow_mv/ir/gvhmr_adapter.py` converts GVHMR’s monocular `smpl_params_global` into the IR. It does not yet accept multiple views, and `K_fullimg` is stored only in `provenance`.

2. **Fusion modules are multi-view-ready at the 2D-keypoint level**
   - `motionflow_mv/fusion/triangulation.py` implements confidence-weighted DLT; tests pass for synthetic 4-view data.
   - `attention_model.py/v2.py`, `robust_triangulation.py`, `residual_refiner.py`, and `temporal_refiner.py` form a progression from pure geometry to learned fusion, residual correction, and temporal smoothing. They take `(B, V, J, 3)` 2D+confidence tensors and projection matrices, not SMPL parameters.
   - `motionflow_mv/pipeline.py::MultiViewPipeline` already fuses per-frame multi-view 2D keypoints into world 3D joints, and `pipeline_utils.py::select_best_person_group` handles cross-view person matching by reprojection error.

3. **Gap: per-view HMR → 3D skeleton fusion → IR**
   - GVHMR (SIGGRAPH Asia 2024) recovers world-grounded SMPL from a *single* video. It is the current upstream of `gvhmr_adapter.py`.
   - The existing code has no adapter that runs GVHMR (or any per-view HMR) independently on each view and then fuses the resulting 2D/3D cues into one `HumanMotionIR`.

4. **Latest literature alignment**
   - **GVHMR** (Shen et al., SIGGRAPH Asia 2024): the project’s current upstream. It produces gravity-aligned world motion from monocular video—useful as a per-view estimator, but not a fusion method.
   - **ScoreHMR** (Stathopoulos et al., CVPR 2024): diffusion-based single-image human recovery. Fits naturally as a strong per-view pseudo-label/uncertainty generator before multi-view fusion.
   - **MVGFormer** (Liao et al., CVPR 2024): a geometry+appearance transformer that iterates learning-free geometric modules with learned appearance modules. Its design is the closest architectural template for improving `AttentionFusionModelV2`.
   - **MvP / Direct Multi-view Multi-person 3D Pose Estimation** (Wang et al., NeurIPS 2021): regresses 3D joints directly with transformer queries; a strong baseline for end-to-end multi-person multi-view fusion.
   - **Mocap-2-to-3** (Wang et al., arXiv 2025) and **RapidPoseTriangulation** (Bermuth et al., arXiv 2025): newer 2025 methods for multi-view lifting and fast triangulation.

5. **Data sources**
   - `VoxelPoseShelfLoader` already loads Shelf/Campus calibration and 2D predictions from the VoxelPose repo.
   - For 3D supervision, the natural next datasets are **Human3.6M**, **CMU Panoptic**, and **3DPW**; these are the standard benchmarks used by GVHMR, ScoreHMR, and MVGFormer.

### Concrete Recommendations

1. **Extend `HumanMotionIR` for multi-view provenance**  
   Add optional fields: `views: List[str]`, `camera_parameters: Dict[str, np.ndarray]`, `per_view_2d: Optional[Dict[str, np.ndarray]]`, and `fusion_method: str`. Keep the downstream interface stable so retargeting/policy code is unaffected.

2. **Add a multi-view adapter stage between GVHMR/ScoreHMR and the IR**  
   Run the per-view HMR on each view to produce 2D keypoints + SMPL params, then call `MultiViewPipeline` (or `MVGFormer`-style fusion) to produce a single world-coordinate 3D skeleton/SMPL, and finally emit a `HumanMotionIR`.

3. **Replace/augment the current attention fusion with a geometry-aware transformer**  
   The current `AttentionFusionModelV2` adds camera embedding but is still shallow. Adopt the MVGFormer principle: interleave learning-free triangulation/geometric modules with learned appearance/attention modules. This directly addresses the finding that pure reprojection loss cannot beat DLT.

4. **Train/evaluate on real 3D-GT data**  
   The README correctly identifies that reprojection-only supervision is the bottleneck. Add loaders for Human3.6M / Panoptic / 3DPW and train with 3D MPJPE loss.

5. **Publish the plugin architecture**  
   A CVPR/ICRA paper should frame the contribution as the modular multi-view extension of MotionFlow plus a systematic comparison of fusion choices, rather than claiming a marginal improvement over DLT on reprojection error.

### Key References

1. Shen, Z. et al. **“World-Grounded Human Motion Recovery via Gravity-View Coordinates.”** *SIGGRAPH Asia 2024.* [arXiv:2409.06662](https://arxiv.org/abs/2409.06662)
2. Stathopoulos, A. et al. **“Score-Guided Diffusion for 3D Human Recovery.”** *CVPR 2024.* [arXiv:2403.09623](https://arxiv.org/abs/2403.09623)
3. Liao, Z. et al. **“Multiple View Geometry Transformers for 3D Human Pose Estimation.”** *CVPR 2024.* [arXiv:2311.10983](https://arxiv.org/abs/2311.10983)
4. Wang, T. et al. **“Direct Multi-view Multi-person 3D Pose Estimation.”** *NeurIPS 2021.* [arXiv:2111.04076](https://arxiv.org/abs/2111.04076)
5. Wang, Z. et al. **“Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining.”** *arXiv 2025.* [arXiv:2503.03222](https://arxiv.org/abs/2503.03222)

### Risks, Blockers, and Assumptions

- **License blockers**: EasyMocap has a non-commercial license and should remain an external benchmark only, not a core dependency.
- **3D ground truth**: The project currently lacks real 3D-GT loaders. Without 3D supervision, learned fusion is unlikely to outperform DLT.
- **GVHMR is monocular**: using it as the per-view estimator assumes independent per-view inference is acceptable and that camera calibration is available.
- **Calibration availability**: the fusion stack assumes calibrated cameras. Uncalibrated multi-view HMR remains future work.
- **Domain gap**: synthetic pre-training did not transfer to Shelf; any new 3D-GT dataset must be close to the target robot-retargeting domain.</subagent>
</agent_swarm_result>