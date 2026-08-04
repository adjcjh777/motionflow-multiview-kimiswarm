# MotionFlow Multi-View Extension — Design v2

## 1. Design Goals and Scope

**Goal:** extend the existing MotionFlow pipeline from monocular video to calibrated multi-view capture, producing a single, metric-scale, world-coordinate `HumanMotionIR` from N synchronized views.

**Scope for v2:**
- Keep the current per-view 2D/3D estimator interface (GVHMR, ScoreHMR, or any `BasePoseEstimator`).
- Add a clean multi-view adapter that consumes per-view `HumanMotionIR`s and emits one fused `HumanMotionIR`.
- Retain DLT as the default fusion backend; add learned fusion heads as plugins.
- Switch training to real 3D ground truth where available, ending the reprojection-only loop.
- Add uncertainty and provenance fields so downstream robot retargeting can reason about view reliability.

**Non-goals for v2:**
- Uncalibrated multi-view fusion (remains future work).
- Multi-person scenes (the current `select_best_person_group` is single-person only).
- Replacing the upstream single-view MotionFlow estimator architecture.

---

## 2. Proposed Architecture (v2)

```
┌─────────────────────────────────────────────────────────────┐
│  Input: N calibrated, synchronized videos of the same action │
└────────────────────────────────────────────────────────────┘
                                 │
                                 v
┌─────────────────────────────────────────────────────────────┐
│  Per-view estimator plugin                                   │
│  • GVHMR adapter  (existing, SIGGRAPH Asia 2024)             │
│  • ScoreHMR adapter  (CVPR 2024, MIT)                        │
│  • BasePoseEstimator interface for 2D detectors              │
└────────────────────────────────┬────────────────────────────┘
                                 │
                                 v
              per-view HumanMotionIR (V views)
                                 │
                                 v
┌─────────────────────────────────────────────────────────────┐
│  Multi-view adapter                                          │
│  • canonicalize coordinate frames                             │
│  • extract per-view 2D keypoints + confidence                 │
│  • load calibrated Camera objects                           │
└────────────────────────────────┬────────────────────────────┘
                                 │
                                 v
┌─────────────────────────────────────────────────────────────┐
│  Fusion module (plugin)                                      │
│  • DLTFusion          (deterministic confidence-weighted)    │
│  • RobustTriangulationFusion  (learned per-view weights)     │
│  • AttentionFusionV2  (geometry-aware transformer)             │
│  • TemporalRefinerFusion  (Bi-GRU window)                    │
│  • ResidualRefinerFusion  (post-DLT residual)                │
└────────────────────────────────┬────────────────────────────┘
                                 │
                                 v
              fused world-coordinate 3D skeleton (T, J, 3)
                                 │
                                 v
┌─────────────────────────────────────────────────────────────┐
│  IR writer                                                   │
│  • fit / average SMPL params                                  │
│  • populate HumanMotionIR.pose, uncertainty, quality, provenance
└────────────────────────────────────────────────────────────┘
                                 │
                                 v
              single fused HumanMotionIR
```

### 2.1 Module Breakdown

| Module | File | Responsibility |
|---|---|---|
| `HumanMotionIR` | `motionflow_mv/ir/human_motion_ir.py` | Stable pose/uncertainty/quality/provenance container |
| `GVHMRAdapter` | `motionflow_mv/ir/gvhmr_adapter.py` | Converts single-view GVHMR output to IR |
| `ScoreHMRAdapter` (new) | `motionflow_mv/ir/scorehmr_adapter.py` | Converts ScoreHMR SMPL/SMPL-X output to IR |
| `MultiViewAdapter` (new) | `motionflow_mv/ir/multiview_adapter.py` | Canonicalizes per-view IRs, extracts 2D/3D evidence |
| `Camera` | `motionflow_mv/calibration/camera.py` | Pinhole camera model (K, R, t) |
| `FusionModule` (new interface) | `motionflow_mv/fusion/fusion_module.py` | Plugin contract: `(B, V, J, 3) x cameras → (B, J, 3)` |
| `DLTFusion` | wraps `triangulation.py` | Confidence-weighted DLT baseline |
| `RobustTriangulationModel` | `robust_triangulation.py` | Learned per-view weights + differentiable DLT |
| `AttentionFusionModelV2` | `attention_model_v2.py` | Transformer fusion with camera embedding |
| `ResidualRefinerModel` | `residual_refiner.py` | Post-DLT per-frame residual correction |
| `TemporalRefinerModel` | `temporal_refiner.py` | Bi-GRU temporal smoothing over a window |
| `MultiViewPipeline` | `motionflow_mv/pipeline.py` | End-to-end orchestration |
| eval metrics | `motionflow_mv/eval/metrics.py` | MPJPE, PA-MPJPE, PCK, MRPE, reprojection |

### 2.2 Data Flow

1. Each video is processed by a per-view estimator plugin, producing a `HumanMotionIR` per view.
2. `MultiViewAdapter` canonicalizes all per-view IRs into the same world frame using `coordinate_system.world_from_reference` and camera extrinsics.
3. The adapter projects the per-view SMPL joints (or accepts 2D keypoints) into `(T, V, J, 3)` tensors of `(x, y, confidence)`.
4. The selected `FusionModule` produces a world-coordinate 3D skeleton `(T, J, 3)`.
5. The IR writer converts the skeleton back to SMPL-compatible parameters (`body_pose`, `global_orient`, `transl`, `betas`) and populates `uncertainty`, `quality`, and `provenance`.

---

## 3. Integrating Fusion with `HumanMotionIR`

The current `HumanMotionIR` dataclass is intentionally minimal. v2 extends it with **optional** multi-view fields so that downstream retargeting code remains unchanged.

### 3.1 Optional IR Extensions

Add the following optional fields to `HumanMotionIR` (all defaulting to `None`/`{}`):

```python
@dataclass
class HumanMotionIR:
    # ... existing fields ...
    views: List[str] = field(default_factory=list)
    camera_parameters: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    per_view_2d: Optional[Dict[str, np.ndarray]] = None          # T x J x 2
    per_view_confidence: Optional[Dict[str, np.ndarray]] = None  # T x J
    fusion_method: str = "dlt"
    uncertainty: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
```

`uncertainty` will contain:
- `view_weights`: `(T, V, J)` learned/normalized view weights
- `reprojection_residual`: `(T, J)` geometric residual after fusion
- `joint_3d_std`: `(T, J)` or `(T, J, 3)` per-joint positional standard deviation

### 3.2 Multi-View Adapter

`motionflow_mv/ir/multiview_adapter.py` provides:

```python
def fuse_multiple_irs(
    irs: List[HumanMotionIR],
    cameras: List[Camera],
    fusion_module: FusionModule,
) -> HumanMotionIR:
    ...
```

Steps:
1. Verify all IRs share the same `human_model`, `fps`, and frame count.
2. Build per-view 2D evidence: either directly from `per_view_2d` or by projecting the SMPL joints of each view using its camera.
3. Run the fusion module.
4. Average per-view `betas`, take the most confident per-view `body_pose` as initialization, and set `global_orient`/`transl` from the fused root.
5. Write a single fused IR with updated `coordinate_system`, `uncertainty`, and `provenance`.

For the first iteration, the SMPL parameter recombination can be a simple average (Option A in the swarm report). A principled SMPL fitting step (Option B) is reserved for later once 3D GT is available.

---

## 4. Dataset and Training Plan

### 4.1 Datasets

| Dataset | Role | 3D GT | License | Notes |
|---|---|---|---|---|
| **Shelf / Campus** | Fast dev + reprojection sanity | Yes (joints) | Research | Already loaded; too small for training large models |
| **Human3.6M** | Primary 3D-supervised training | Yes (32 joints) | Research, registration | S1,5,6,7,8 train; S9,11 test |
| **CMU Panoptic** | Multi-person + temporal validation | Yes (COCO19) | Research-only | Download a few HD views only |
| **3DPW** | In-the-wild validation | SMPL | Research | Moving camera, not fixed rig |
| **AMASS** | Synthetic pre-training | SMPL params | Research | Render through virtual multi-view rigs |

**WebBridge:** the swarm's dataset agent did not identify a WebBridge-specific data feed. We fall back to direct dataset registration/download and synthetic generation.

### 4.2 Training Stages

**Stage 1 — Synthetic pre-training**
- Generate realistic multi-view sequences from AMASS SMPL motion clips.
- Project joints through virtual calibrated rigs with realistic 2D detection noise and occlusion patterns.
- Train `TemporalRefinerModel` and `AttentionFusionModelV2` with a 3D MSE loss.
- Goal: initialize temporal/motion priors, not to match the target domain exactly.

**Stage 2 — Real fine-tuning**
- Fine-tune on Human3.6M (and optionally Shelf 3D GT) with the combined loss:

```
L = λ_3d * L_MPJPE + λ_reproj * L_reproj + λ_bone * L_bone + λ_temp * L_temporal + λ_sym * L_symmetry
```

- `L_bone`: bone-length consistency (Bragagnolo et al., ECCVW 2024).
- `L_temporal`: velocity smoothness over windows.
- `L_symmetry`: limb-length symmetry.
- Reserve CMU Panoptic/3DPW for cross-dataset validation.

**Fallback if real 3D GT is delayed:**
- Use ScoreHMR per-view predictions as pseudo-3D labels.
- Train with pseudo-3D loss + reprojection + bone-length terms.
- Clearly label these results as pseudo-supervised and validate against real 3D GT as soon as possible.

### 4.3 Loss Details

- `L_MPJPE`: per-joint Euclidean distance to 3D GT in millimeters.
- `L_reproj`: per-view 2D reprojection error after fusion.
- `L_bone`: mean absolute deviation of bone lengths from a learned or dataset prior.
- `L_temporal`: L2 norm of second-order finite differences of 3D joints.
- `L_symmetry`: L1 difference between left/right limb bone lengths.

---

## 5. Evaluation Protocol and Metrics

### 5.1 Primary Metrics

- **MPJPE** (mm): root-relative per-joint position error.
- **PA-MPJPE** (mm): after Procrustes alignment.
- **PCK@150mm** and **AUC**: percentage of joints within threshold.
- **MRPE** (mm): root/pelvis absolute position error. Critical because robot retargeting needs metric world coordinates.

### 5.2 Diagnostic Metrics

- **Reprojection error** (px): retained for sanity, but not used as the primary claim.
- **Bone-length consistency**: mean absolute deviation from a skeleton prior.
- **Temporal jitter**: average second derivative of 3D joints.
- **Per-joint / per-body-part breakdown**: to identify whether arms, legs, or torso benefit most from multi-view fusion.

### 5.3 Test Sets

- **Shelf/Campus**: reprojection sanity and fast ablations; use frames 300–600 as a held-out test set.
- **Human3.6M S9/S11**: primary 3D accuracy benchmark.
- **CMU Panoptic (small HD subset)**: multi-person/temporal stress test.
- **Synthetic test set**: controlled noise and occlusion ablations.

### 5.4 Baselines

1. Confidence-weighted DLT (`triangulation.py`)
2. `RobustTriangulationModel` (learned weights)
3. `AttentionFusionModelV2`
4. `ResidualRefinerModel`
5. `TemporalRefinerModel`
6. External: ScoreHMR per-view, EasyMocap (non-commercial, benchmark only)

---

## 6. Novel Contribution Angles for ICRA/CVPR 2027

Rather than claiming “a better triangulator than DLT on reprojection,” the paper story should be:

1. **A modular, world-grounded multi-view extension of MotionFlow.** Define the first end-to-end `HumanMotionIR`-compatible multi-view fusion stack, where DLT and learned fusion heads are interchangeable plugins.

2. **Systematic empirical study of fusion choices.** Provide the first public comparison of DLT, learned weighting, residual refinement, temporal refinement, and synthetic pre-training under a unified IR, on real 3D-GT benchmarks.

3. **Uncertainty-aware IR for downstream robotics.** Populate `HumanMotionIR.uncertainty` with per-view weights, reprojection residuals, and per-joint standard deviations, enabling risk-aware robot retargeting and motion planning.

4. **Geometry-aware learned fusion.** Adopt the MVGFormer principle (geometry + appearance) and state-space view scanning (MV-SSM, CVPR 2025) to improve over the current shallow `AttentionFusionModelV2`.

5. **Strong per-view priors.** Integrate ScoreHMR as a plug-in per-view estimator and demonstrate improved pseudo-3D labels for fusion training.

6. **Robot downstream validation (ICRA angle).** Evaluate how the fused 3D pose reduces foot sliding, ground penetration, or end-effector error in a robot retargeting/policy task, not just MPJPE.

---

## 7. Risk and Feasibility Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No real 3D GT available | Medium | High | Register Human3.6M; use ScoreHMR pseudo-3D labels as fallback |
| Learned fusion still ties DLT | Medium | High | Train with 3D + bone-length + temporal losses; frame paper as modular study, not raw accuracy claim |
| License issues (Panoptic, EasyMocap) | Low | Medium | Use only as external benchmarks; keep ScoreHMR (MIT) as default plugin |
| Calibration errors dominate | Medium | High | Validate on calibrated rigs first; add COLMAP/DUSt3R fallback later |
| Per-view estimator cost (ScoreHMR) | Medium | Medium | Batch on A800-D; keep GVHMR as fast default |
| Multi-person matching fails | Medium | Medium | Restrict v2 to single-person; prototype Hungarian/epipolar matching in v3 |
| Domain gap from synthetic pre-training | Medium | Medium | Use AMASS-derived realistic motion; match camera/noise distributions |

**Feasibility:** The repository already has the core modular skeleton. The main feasibility condition is 3D GT access. If Human3.6M is obtained within the next iteration, a 3D-supervised fusion head can be trained and evaluated on the RTX 4090 / A800-D. If not, ScoreHMR pseudo-labeling keeps progress moving.

---

## 8. Concrete Next-Iteration Plan (1–2 Weeks)

### Milestone 1 — IR extension and adapter (days 1–2)
- Add optional multi-view fields to `HumanMotionIR`.
- Implement `motionflow_mv/ir/multiview_adapter.py` with `fuse_multiple_irs(...)`.
- Ensure `pytest tests/` still passes.

### Milestone 2 — Fusion plugin interface (days 2–3)
- Create `FusionModule` ABC and wrap DLT, `RobustTriangulationModel`, `AttentionFusionModelV2`, `ResidualRefinerModel`, and `TemporalRefinerModel` as plugins.
- Update `MultiViewPipeline.__init__` to accept a `FusionModule`.
- Add a config-driven factory so the fusion backend can be swapped with one line.

### Milestone 3 — 3D GT loader and 3D loss (days 3–5)
- Add `motionflow_mv/data/human36m_loader.py` returning `(points_2d, confidences, proj_matrices, joints_3d_gt)`.
- Update `train_attention_fusion_shelf.py` / `train_temporal_refiner_shelf.py` to accept 3D GT and minimize `L_MPJPE + λ_reproj * L_reproj`.
- Add `L_bone` and `L_symmetry` helpers.

### Milestone 4 — Evaluation harness (days 5–7)
- Extend `motionflow_mv/eval/metrics.py` with MRPE and per-joint/bone-length diagnostics.
- Create `experiments/compare_fusion_h36m.py` that runs all fusion plugins and reports MPJPE/PA-MPJPE/PCK.
- Validate on Shelf 3D GT if available.

### Milestone 5 — Per-view ScoreHMR plugin (days 7–10)
- Implement `motionflow_mv/ir/scorehmr_adapter.py` mirroring `gvhmr_adapter.py`.
- Add `experiments/run_scorehmr_multiview.py` demo: N views → per-view IR → fusion → single `HumanMotionIR`.
- Document in `docs/design_v2.md`.

### Acceptance Criteria
- `pytest tests/` passes.
- At least one learned fusion model is trained with 3D loss and evaluated against DLT on a real 3D-GT validation set.
- A working end-to-end demo produces a fused `HumanMotionIR` from multi-view input.
- `docs/design_v2.md` and `docs/swarm_iter2/synthesis_phase1.md` are complete.

---

## References

- Stathopoulos et al., *Score-Guided Diffusion for 3D Human Recovery*, CVPR 2024.
- Matsubara et al., *HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery*, CVPR 2025.
- Liao et al., *Multiple View Geometry Transformers for 3D Human Pose Estimation*, CVPR 2024.
- Chharia et al., *MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation*, CVPR 2025.
- Bragagnolo et al., *Multi-view Pose Fusion for Occlusion-Aware 3D Human Pose Estimation*, ECCVW 2024.
- Bermuth et al., *RapidPoseTriangulation*, arXiv 2025.
- Wang et al., *Mocap-2-to-3*, arXiv 2025.
- Iskakov et al., *Learnable Triangulation of Human Pose*, ICCV 2019.
- Ionescu et al., *Human3.6M*, T-PAMI 2014.
- Joo et al., *Panoptic Studio*, T-PAMI 2017.
- Mahmood et al., *AMASS*, ICCV 2019.
