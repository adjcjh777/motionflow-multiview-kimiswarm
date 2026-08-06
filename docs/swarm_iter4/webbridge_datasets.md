# WebBridge Dataset Sources for Multi-View 3D Pose

> Scope: Survey and prioritize publicly accessible dataset sources for training and validating the MotionFlow-MultiView `ray_attention` fusion pipeline, focusing on Human3.6M, Shelf/Campus, 3DPW, AMASS, and related aggregators.

## 1. Survey

The `ray_attention` model (`motionflow_mv/fusion/ray_attention_model.py`) already triangulates metric-scale 3D joints from calibrated multi-view 2D keypoints using differentiable weighted DLT. On synthetic data it outperforms the old end-to-end attention head, but on real data it is limited mainly by the lack of large, 3D-supervised training corpora. The project needs a curated bridge to web-accessible datasets with calibrated multi-view footage and true 3D ground truth.

| Dataset | Views | 3D GT | Skeleton | Notes |
|---------|-------|-------|----------|-------|
| **Shelf / Campus** | 3–5 fixed | 3D joints (mm) | 14–17 joints | Already in `data/shelf_campus/`; fast dev loop |
| **Human3.6M** | 4 fixed, 50 Hz | 3D joints (mm, world coords) | 32 → 17 | 3.6M frames; academic registration required |
| **CMU Panoptic** | 31 HD + 480 VGA | 3D COCO19 | COCO19 | Multi-person social scenes |
| **3DPW** | Single moving + IMU | SMPL pose/shape | SMPL 24 joints | 60 in-the-wild sequences; not multi-view |
| **AMASS** | — | SMPL pose/shape | SMPL/SMPL-X | Large mocap aggregate for synthetic rigs |
| **ScoreHMR pseudo-labels** | N/A | SMPL/SMPL-X | SMPL/SMPL-X | Pseudo-3D labels for unlabeled video |

Search engines and aggregators include **Google Dataset Search**, **Papers With Code**, **Hugging Face Datasets**, and **Zenodo**. The codebase already has Shelf/Campus loaders and a synthetic SMPL generator; loaders for H36M, Panoptic, and 3DPW are still missing.

## 2. Recommendations

1. **Acquire Human3.6M first.** Register at the official site, download the D3 positions, and add `motionflow_mv/data/human36m_loader.py` returning the same `(points_2d, confidences, joints_3d, cameras)` tuple used by `train_ray_attention_real.py`. Map H36M's 32 joints to the 17-joint COCO-style skeleton and convert mm → m on load.

2. **Build a `DatasetRegistry` adapter layer.** Wrap H36M, Shelf/Campus, Panoptic, and synthetic AMASS generators behind a single class exposing the plugin contract. This avoids duplicating preprocessing across training scripts and enforces consistent skeletons, units, and camera conventions.

3. **Generate synthetic multi-view data from AMASS.** Forward SMPL poses through configurable virtual rigs to create unlimited 3D-supervised sequences with controlled occlusion, noise, and outliers. This bridges the gap while real datasets are pending approval.

4. **Use 3DPW only for in-the-wild SMPL validation.** 3DPW is single-camera plus IMU, so it cannot train calibrated multi-view fusion directly. Use it to test cross-domain generalization after H36M training.

5. **Maintain a YAML provenance manifest.** Record official URL, mirror, license, registration status, checksum, skeleton convention, and ingestion script for each source. Update it whenever a new candidate is found.

## 3. Risks

- **License restrictions.** H36M, 3DPW, and Panoptic are research-only or require explicit agreements; commercial use needs additional permission.
- **Registration latency.** H36M/3DPW approvals can take days or weeks, so run the synthetic AMASS pipeline in parallel.
- **Skeleton/unit mismatch.** H36M uses 32 joints, Panoptic uses COCO19, and the fusion plugins expect 17 joints. A canonical rig mapper must be unit-tested.
- **Calibration inconsistency.** Plugins trained on one rig may not transfer to another unless inputs are normalized (rays + camera centers, already done in `ray_attention`) and augmented with scale-aware jitter.
- **Stale mirrors.** Web search can return outdated or modified copies; always verify against the official page and record checksums.

## 4. Paper Fit

For CVPR/ICRA 2027 the strongest claim is a calibrated multi-view fusion module that systematically combines geometric triangulation with learned view weighting under a common `HumanMotionIR`. To support this we need:

- **Shelf/Campus** for fast, calibrated reprojection validation.
- **Human3.6M** as the primary 3D-supervised training set to show `ray_attention` beats pure DLT on real 3D error (MPJPE / PA-MPJPE).
- **3DPW / AMASS-synthetic** to demonstrate in-the-wild and motion-prior generalization.
- **CMU Panoptic** (optional) for multi-person and temporal extension.

The immediate deliverable is a working H36M loader and a YAML dataset manifest; the synthetic AMASS generator can fill the gap until H36M is approved.
