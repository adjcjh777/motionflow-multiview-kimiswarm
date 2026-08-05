I read the relevant codebase and docs, then drafted the requested report. Since this subagent is read-only and does not have a file-write tool, I could not save it to `docs/swarm_iter3/webbridge_dataset_search.md` — the report text is below for the parent agent to persist.

---

# Web-Scale Dataset Search for Multi-View Human Motion (WebBridge Dataset Search)

> **Scope:** Survey how web-accessible dataset search engines and repositories can expand the data palette for the `motionflow-multiview` fusion pipeline. This note focuses on *finding, licensing, and ingesting* multi-view human pose / SMPL datasets rather than on model architecture.

---

## 1. Problem Statement

The `motionflow-multiview` pipeline currently trains and validates almost exclusively on the **Shelf / Campus** datasets (`motionflow_mv/data/voxelpose_loader.py`). These datasets are small (~2–3 k frames, 3–5 fixed calibrated cameras) and have already shown the limits of reprojection-only supervision: learned plugins (`attention`, `residual_refiner`, `temporal_refiner`) match geometric DLT but do not clearly surpass it. To reach CVPR / ICRA 2027 quality, the project needs:

1. **Larger 3D-supervised data** for training learned fusion modules.
2. **Diverse capture conditions** (indoor lab, in-the-wild, multi-person, varying camera rigs) to validate generalization.
3. **Clear, traceable provenance** for every dataset used, including license terms and registration requirements.

"WebBridge dataset search" therefore denotes a deliberate, web-scale discovery process: using public dataset search engines, code repositories, and dataset aggregators to identify candidate data sources that can be licensed, downloaded, and mapped into the `HumanMotionIR` / `FusionModule` contract.

---

## 2. Key Related Work / Data Sources

### 2.1 Multi-view / SMPL datasets most relevant to MotionFlow

| Dataset | Views | 3D GT | Skeleton / Model | License | Best Use |
|--------|------|------|------------------|---------|----------|
| **Shelf / Campus** | 3–5 fixed | 3D joints (17/18) | Custom / COCO-like | Research / benchmark | Dev loop, fast regression tests |
| **Human3.6M** | 4 fixed, 50 Hz | 3D joints (32) | H36M 32 joints | Academic registration | Primary 3D-supervised training |
| **CMU Panoptic** | 31 HD + 480 VGA | 3D COCO19 joints | COCO19 | Research-only, non-commercial | Multi-person / social motion |
| **3DPW** | Single moving + IMU | SMPL pose / shape / trans | SMPL (24 joints) | Research agreement | In-the-wild validation |
| **AMASS** | — (mocap only) | SMPL pose / shape | SMPL/SMPL-X | Research | Motion prior / synthetic multi-view generation |
| **ScoreHMR pseudo-labels** | N/A | SMPL/SMPL-X via diffusion | SMPL/SMPL-X | MIT code; model depends on SMPL license | Pseudo-3D labels for unlabeled video |

### 2.2 Key methods that rely on these data sources

- **Iskakov et al., *Learnable Triangulation of Human Pose*, ICCV 2019.** Foundational work showing that volumetric and confidence-weighted triangulation benefit from large multi-view training data. Justifies the `RobustTriangulationModel` / `AttentionFusionModel` direction in the repo.
- **Bragagnolo et al., *Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation*, ECCV Workshops 2024 (arXiv:2408.15810).** Fuses monocular 3D skeletons with reprojection and limb-symmetry constraints on Human3.6M. Demonstrates that 3D GT and occlusion-aware losses are required to beat pure triangulation.
- **Bermuth et al., *RapidPoseTriangulation*, arXiv:2503.21692 (2025).** Fast whole-body multi-person triangulation; useful as an engineering baseline and for real-time ICRA scenarios.
- **Wang et al., *Mocap-2-to-3*, arXiv:2503.03222 (2025).** Shows 2D pretraining + multi-view 3D fine-tuning can recover metric-scale motion with limited 3D labels—directly relevant if only small real datasets can be licensed.
- **Stathopoulos et al., *Score-Guided Diffusion for 3D Human Recovery* (ScoreHMR), CVPR 2024 (arXiv:2403.09623).** A strong per-view SMPL/SMPL-X estimator that can generate pseudo-3D labels for arbitrary video, effectively turning unlabeled multi-view footage into a training source.

### 2.3 Dataset search engines / aggregators

- **Google Dataset Search** (datasetsearch.research.google.com) – indexes millions of dataset pages; good first pass for "multi-view human pose" and "SMPL dataset".
- **Papers With Code** – links tasks (e.g., 3D human pose estimation) to datasets and leaderboard numbers; useful for benchmarking context.
- **Hugging Face Datasets / Kaggle / OpenML / Zenodo** – host smaller community datasets and synthetic data; valuable for quick prototyping and for finding SMPL motion sequences.

Important caveat: **"WebBridge" is not a dataset.** Prior swarm notes (see `docs/swarm_iter2/synthesis_scorehmr_datasets.md`) identify it as a browser-automation tool. Any web-scale search must therefore be paired with human verification of license, format, and calibration quality.

---

## 3. How It Relates to the Current Codebase

### 3.1 Data loader gap

- The only production loader is `motionflow_mv/data/voxelpose_loader.py` → `VoxelPoseShelfLoader`. It exposes `cameras`, `get_camera(cid)`, and `get_frame_predictions(cid, frame_idx)`.
- To ingest Human3.6M / Panoptic / 3DPW, new loaders are needed that produce the same `(points_2d, confidences, cameras, joints_3d_gt)` tuple consumed by `FusionModule.fuse()` and the training scripts.

### 3.2 IR / fusion contract

- `HumanMotionIR` (`motionflow_mv/ir/human_motion_ir.py`) already carries `coordinate_system`, `per_view_2d`, `per_view_confidence`, and `camera_parameters`. This is the right abstraction for multi-view dataset ingestion.
- `fuse_multiple_irs()` in `motionflow_mv/ir/multiview_adapter.py` expects per-view 2D observations and confidences. Datasets that provide 3D GT can therefore be plugged in as long as an adapter projects 3D joints to each view and stores confidence.

### 3.3 Training scripts

- `experiments/train_attention_fusion_shelf_v2.py` and siblings currently train on Shelf pseudo-targets derived from DLT. They directly benefit from real 3D labels (Human3.6M, Panoptic) because the current reprojection-only objective caps learned fusion at DLT performance.

---

## 4. Concrete Recommendations

1. **Maintain a dataset search tracker.** Create a spreadsheet or YAML manifest with columns: dataset name, search source, official URL, license, registration required, views, 3D GT type, skeleton, size, and ingestion status. Update it whenever a new candidate is found via Google Dataset Search / Papers With Code / Hugging Face.

2. **Priority acquisition order:**
   - **Immediate:** Keep Shelf / Campus for fast iteration.
   - **Short-term:** Register and download **Human3.6M**. It is the smallest-friction large 3D-supervised source for training `AttentionFusionModelV2` / `TemporalRefinerModel` with a true 3D MSE loss.
   - **Medium-term:** Pull a small **CMU Panoptic** HD subset for multi-person matching and temporal experiments.
   - **Validation:** Use **3DPW** for in-the-wild SMPL evaluation and **AMASS** for synthetic multi-view augmentation.

3. **Build a `motionflow_mv/data/dataset_registry.py`.** Provide lightweight adapters that map each dataset to the common `(T, V, J, 2)` points-2d, `(T, V, J)` confidence, `[Camera]` list, and optional `(T, J, 3)` world-ground-truth format. This avoids duplicating preprocessing logic across `train_*_shelf.py` scripts.

4. **Use ScoreHMR to expand pseudo-labeled data.** Since ScoreHMR is MIT-licensed and produces SMPL/SMPL-X parameters from single images or uncalibrated multi-view inputs, it can generate pseudo-3D labels for web-gathered multi-view videos where no GT exists. Cache per-view `HumanMotionIR`s before fusion.

5. **Generate synthetic multi-view training data from AMASS.** Forward SMPL poses through a configurable virtual camera rig to produce unlimited 3D-supervised `(points_2d, cameras, joints_3d_gt)` examples. This is the fastest way to break the current DLT ceiling when real datasets are still pending registration.

6. **Standardize evaluation metrics.** Beyond reprojection error, report **MPJPE**, **PA-MPJPE**, and **PCK@150mm** on Human3.6M / Panoptic; report reprojection error on Shelf / Campus. This aligns the project with CVPR / ICRA reviewing expectations.

---

## 5. Open Questions / Risks

- **License and commercial use.** Human3.6M, CMU Panoptic, and Shelf/Campus are research-only or require explicit agreements. If MotionFlow targets a commercial product, these datasets cannot be redistributed or used for commercial training without additional permission.
- **Web search reliability.** Dataset links found via search engines can be stale or point to mirrors with inconsistent checksums. Always verify against the official project page and record SHA256 / citation details.
- **Skeleton and coordinate mismatch.** Human3.6M uses 32 joints, Panoptic uses COCO19, and SMPL uses 24 joints. Each dataset needs a rig mapper to the 17-joint fusion skeleton used by the current `FusionModule` implementations.
- **Registration latency.** Human3.6M and 3DPW require manual approval; plan a parallel synthetic-pipeline path so development is not blocked.
- **Quality of web-found datasets.** Not every result from a dataset search engine is calibrated, synchronized, or correctly licensed. A “download and ingest” script must include sanity checks (projection of 3D GT to 2D, bone-length consistency, frame sync).
- **Domain gap.** Training on indoor lab data (Human3.6M, Panoptic) may not transfer to in-the-wild deployment. Include 3DPW and AMASS-synthetic outdoor-style sequences in the validation mix.

---

## 6. Key References

1. Ionescu et al., "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments," *TPAMI*, 2014.
2. Joo et al., "Panoptic Studio: A Massively Multiview System for Social Interaction Capture," *TPAMI*, 2017.
3. von Marcard et al., "Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera," *ECCV*, 2018 (3DPW).
4. Mahmood et al., "AMASS: Archive of Motion Capture as Surface Shapes," *ICCV*, 2019.
5. Iskakov et al., "Learnable Triangulation of Human Pose," *ICCV*, 2019.
6. Stathopoulos et al., "Score-Guided Diffusion for 3D Human Recovery," *CVPR*, 2024. [arXiv:2403.09623](https://arxiv.org/abs/2403.09623)
7. Bragagnolo et al., "Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation," *ECCV Workshops*, 2024. [arXiv:2408.15810](https://arxiv.org/abs/2408.15810)
8. Bermuth et al., "RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond," *arXiv*, 2025. [arXiv:2503.21692](https://arxiv.org/abs/2503.21692)
9. Wang et al., "Mocap-2-to-3: Multi-view Lifting for Monocular Motion Recovery with 2D Pretraining," *arXiv*, 2025. [arXiv:2503.03222](https://arxiv.org/abs/2503.03222)

---

**Summary:** The current codebase is anchored on Shelf/Campus and reprojection-only supervision. To move toward CVPR/ICRA 2027, the next step is a disciplined web-scale dataset search and ingestion plan: register for Human3.6M, sample CMU Panoptic, validate on 3DPW, and augment with AMASS/ScoreHMR pseudo-labels—while carefully tracking licenses and format mappings.