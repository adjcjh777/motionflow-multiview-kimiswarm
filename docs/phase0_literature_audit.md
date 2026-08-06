# Phase 0: Candidate Multi-View Fusion Method Audit

> Status: read-only audit. No implementation.

This document audits the candidate methods mentioned in `research/multiview-easymocap-robot-profiles.md` for paper use: ScoreHMR, EasyMocap, and the placeholder names MUC / DMMR. The goal is to decide which methods are viable for integration, which require adapters, and which are blocked by license.

## 1. ScoreHMR

- **Paper**: Score-Guided Diffusion for 3D Human Recovery (CVPR 2024)
- **arXiv**: http://arxiv.org/abs/2403.09623
- **Code**: https://github.com/statho/ScoreHMR
- **License**: MIT
- **Input**: single image or video
- **Output**: SMPL / SMPL-X parameters (pose, shape, global translation)
- **Relevance**: State-of-the-art single-view 3D human recovery. Could be used as a **strong per-view estimator** before multi-view fusion, or as a pseudo-label generator for 3D supervision. Not a multi-view fusion method itself.
- **Open questions**: Does it produce per-frame uncertainty? Can it be run independently on each view and fused downstream? What is the inference speed/frame?

## 2. EasyMocap

- **Code**: https://github.com/zju3dv/EasyMocap
- **License**: Custom academic/non-commercial license
  - Educational, research, and non-profit use only.
  - Derivative works must be open-source and non-commercial.
  - Commercial use requires contacting the authors.
- **Input**: multi-view calibrated videos/images
- **Output**: SMPL parameters, camera parameters, 3D keypoints
- **Relevance**: Direct multi-view markerless motion capture. A natural geometric/optimization baseline.
- **Blocker**: **License is not permissive for commercial deployment**. The research design doc already flagged that EasyMocap cannot be included in the default MotionFlow Docker release. It may still be usable as an optional external plugin or as a benchmark in a paper, provided the license is cited and the code is not redistributed.

## 3. MUC / DMMR (placeholders)

These acronyms did not resolve to unique public repositories or papers in arXiv/GitHub search. They likely refer to internal or draft methods. Before Phase 1, the authors must clarify what each acronym stands for (paper title + code URL).

## 4. Other relevant methods from recent literature

A quick arXiv search for "multi-view human pose/motion" returns many candidates:

| Method | Venue / arXiv | Note |
|--------|--------------|------|
| TransFusion | arXiv 2110.09554 | Cross-view transformer fusion for 3D pose |
| Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation | arXiv 2408.15810 | Directly relevant to our occlusion problem |
| RapidPoseTriangulation | arXiv 2503.21692 | Multi-view multi-person triangulation in milliseconds |
| Mocap-2-to-3 | arXiv 2503.03222 | Multi-view lifting with 2D pretraining |

These are candidate baselines but require further paper/code/license review before integration.

## 5. Audit conclusion

- **ScoreHMR**: permissive license, strong per-view estimator. Use as a per-view pseudo-label / feature generator, not a fusion module.
- **EasyMocap**: non-commercial license. Can be used as a paper benchmark or optional plugin, not in the default product image.
- **MUC / DMMR**: undefined. Need clarification from authors.
- **Immediate next step**: define `HumanMotionIR` and a fusion plugin protocol, then build a minimal **ScoreHMR-per-view + geometric multi-view fusion** pipeline that fits into the existing MotionFlow GMR/MJLab flow. This keeps the system modular and avoids license-blocked dependencies in the core release.

## 6. Recommendations for a CVPR/ICRA paper

1. Do not claim to beat DLT on reprojection alone without 3D GT or strong priors.
2. Frame the contribution as a **modular multi-view extension of an existing video-to-policy pipeline**.
3. Use EasyMocap only as an external baseline, clearly citing its non-commercial license.
4. Use ScoreHMR as the per-view frozen estimator and contribute the **plugin architecture + HumanMotionIR** that allows swapping fusion modules.
5. Evaluate on both HMR metrics and downstream robot-policy metrics, not just 2D reprojection.
