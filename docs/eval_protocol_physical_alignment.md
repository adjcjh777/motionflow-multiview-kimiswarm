# Physical-Space Alignment Evaluation Protocol

**Target:** ICRA / CVPR 2027 multi-view human pose estimation.  
**Scope:** Standardized evaluation of how well a predicted 3-D pose is *physically consistent* with the multi-view observations that produced it.  
**Last updated:** 2026-08-09

---

## 1. Motivation

Current evaluation (`MPJPE`, `PA-MPJPE`, `PCK`, `AUC`) measures 3-D error against a ground-truth skeleton, but it does not directly test whether a prediction is *geometrically explainable* by the input views. Two predictions can have the same MPJPE yet differ wildly in:

1. **Reprojection consistency** — how well the 3-D joints project back to the observed 2-D keypoints.
2. **Ray intersection / triangulation quality** — whether the predicted points lie near the optical rays from each view.
3. **Physical plausibility** — whether bone lengths, joint limits, and floor contacts are realistic.
4. **Cross-view stability** — whether the prediction remains consistent when subsets of views are active.

This protocol defines a compact, reproducible set of metrics and a driver script that quantify **physical-space alignment across views**.

---

## 2. Definitions

For a clip with `V` calibrated views and `J` joints, let:

- `P` be the predicted 3-D pose, shape `(J, 3)` in world coordinates (meters or millimeters).
- `X` be the input 2-D keypoints, shape `(V, J, 2)` in pixels.
- `K, R, t` be intrinsic / extrinsic camera parameters.
- `parents` be the kinematic parent list.
- `symmetry_pairs` be the left/right symmetric-joint pairs.
- `foot_indices` be the foot-joint indices.

**Physical-space alignment** is the combined satisfaction of:

1. *Image-to-3D reprojection*: `Π_v(P_j)` is close to `X_{v,j}` for every active view `v` and joint `j`.
2. *Ray consistency*: the 3-D point `P_j` lies close to the back-projected ray from each view.
3. *Physical plausibility*: bone lengths, joint angles, symmetry, and floor contact are within human limits.
4. *View-subset stability*: the above holds when only `k ≤ V` views are active.

---

## 3. Metrics

### 3.1 Reprojection error (REP)

Project the predicted 3-D joints back into each view and compute the Euclidean distance to the input 2-D keypoints.

```
REP_v  = (1/J) Σ_j || Π_v(P_j) - X_{v,j} ||_2
REP    = (1/V) Σ_v REP_v
```

Unit: pixels.  Lower is better.

*Reference implementation:* `motionflow_mv/losses/reprojection.py::reprojection_loss`.

### 3.2 Back-projection / ray distance (RDP)

For each view `v`, joint `j`, and predicted point `P_j`, compute the 3-D distance from `P_j` to the ray originating at camera center `C_v` and passing through the back-projection of `X_{v,j}`.

```
r_{v,j} = backproject(K_v, R_v, t_v, X_{v,j})
RDP_v  = (1/J) Σ_j min_λ || P_j - (C_v + λ · r_{v,j}) ||_2
RDP    = (1/V) Σ_v RDP_v
```

Unit: same as `P` (mm when metrics use mm).  Lower is better.

This metric is invariant to the depth scale ambiguity of a single view and directly measures whether the predicted 3-D point lies on the observed ray bundle.

### 3.3 Triangulation residual (TR)

Run a minimal algebraic or iterative triangulation (e.g., DLT) on the 2-D observations and compute the distance to the model prediction:

```
P_tri = triangulate(X, K, R, t)
TR    = (1/J) Σ_j || P_j - P_tri_j ||_2
```

Unit: same as `P`.  Lower is better.

A large `TR` indicates the prediction disagrees with the geometric lower-bound produced by the cameras themselves.

### 3.4 Physical plausibility terms

Reuse the loss terms in `motionflow_mv/losses/skeleton_physical_loss_v40.py`, but report them as **metrics** on the predicted pose (no ground truth required unless noted).

| Metric | Formula | Source |
|---|---|---|
| Bone-length variance (BLV) | Std-dev of predicted bone lengths across the dataset | `skeleton_physical_loss_v40` bone vectors |
| Joint-limit violation (JLV) | Fraction of interior angles exceeding `max_flexion_deg` | `joint_limit_loss` |
| Symmetry error (SE) | Mean left/right bone-length difference | `symmetry_loss` |
| Floor penetration (FP) | Mean depth of feet below estimated floor | `floor_loss` |
| Floor velocity (FV) | Mean foot velocity when near floor | `floor_loss` |
| Self-collision count (SC) | Number of capsule-capsule intersections | `physical_collision_penalty_v31` |

These can be compared against ground-truth statistics when GT is available, or reported as absolute plausibility scores.

### 3.5 Cross-view stability (CVS)

For every subset size `k ∈ {2, 3, 4, …, V}` (or a sampled set), run inference using only `k` active views and measure the per-joint standard deviation of the predicted 3-D positions across different view subsets:

```
P(k, s) = prediction using active-view subset s of size k
CVS(k)  = mean_j std_s( P(k, s)_j )
```

Unit: same as `P`.  Lower is better.

A model that is physically well-aligned should produce stable 3-D estimates regardless of which subset of views is used.

### 3.6 Composite score (optional)

A single physical-space alignment score can be formed by normalizing and weighting the above metrics:

```
PSA = w1·REP + w2·RDP + w3·TR + w4·JLV + w5·CVS
```

Default weights: `w = (0.25, 0.25, 0.20, 0.15, 0.15)`.  Weights are dataset- and unit-dependent and should be calibrated to the paper's primary metric (MPJPE).

---

## 4. Perturbation axes

Apply the following perturbations to measure robustness of physical-space alignment.

| Axis | Levels | Reference |
|---|---|---|
| Clean baseline | — | — |
| 2-D keypoint noise | σ = 1, 2, 5 px | `motionflow_mv/calibration/perturb.py` (indirectly on 2-D) |
| View dropout | k = 2, 3, 4 active views | `motionflow_mv/fusion/variable_view_inference.py` |
| Camera rotation noise | 0.3°, 0.5°, 1.0° | `perturb_extrinsics` |
| Camera translation noise | 10 mm, 50 mm | `perturb_extrinsics` |
| Focal-length error | 1%, 3%, 5% | `perturb_intrinsics` |
| Principal-point error | 5 px, 10 px, 20 px | `perturb_intrinsics` |
| Combined | noise + dropout + rotation | Deterministic seed, three-way combo |

For each condition, report every metric in Section 3 plus standard `MPJPE`/`PA-MPJPE` against GT.

---

## 5. Protocol steps

1. **Load model and data.**
   - Use the same validation loader as the current benchmark (`motionflow_mv/eval/benchmark_protocol.py`).
   - Keep `clip_len`, `stride`, and `unit_scale` identical to the standard protocol.

2. **Run clean evaluation.**
   - Compute `REP`, `RDP`, `TR`, physical-plausibility metrics, and `CVS`.
   - Record `MPJPE`/`PA-MPJPE` for comparison.

3. **Run perturbation matrix.**
   - For each axis/level in Section 4, corrupt the input or cameras **before** inference.
   - Use a deterministic seed derived from the base seed so results are reproducible.
   - Skip GPU training; only inference on the validation split.

4. **Run active-view ablation.**
   - For each `k`, sample `n=5` view subsets (or enumerate if `V` is small).
   - Compute `CVS(k)` and per-subset MPJPE.

5. **Persist results.**
   - JSON manifest: `physical_alignment_results.json`.
   - Markdown table: `physical_alignment_results.md`.
   - CSV for plotting: `physical_alignment_results.csv`.

---

## 6. Reporting format

### 6.1 JSON manifest

```json
{
  "protocol": "physical-space-alignment-v1",
  "model": "OmniMultiViewFusionV5",
  "checkpoint": "outputs/v43_adaptive_per_node_residual.pth",
  "dataset": "mpiinf3dhp_val",
  "config": {
    "clip_len": 27,
    "stride": 1,
    "unit_scale": 1000.0,
    "seed": 42
  },
  "clean": {
    "rep_px": 2.34,
    "rdp_mm": 4.12,
    "tr_mm": 6.55,
    "blv_mm": 8.90,
    "jlv": 0.02,
    "se_mm": 3.21,
    "fp_mm": 0.11,
    "fv_mm": 1.34,
    "sc_count": 0.05,
    "cvs_mm": 7.12,
    "mpjpe_mm": 26.42,
    "pa_mpjpe_mm": 18.31
  },
  "perturbations": [
    {
      "axis": "keypoint_noise",
      "level": "2px",
      "rep_px": 3.89,
      "rdp_mm": 7.41,
      "mpjpe_mm": 31.20
    }
  ],
  "active_views": [
    {"k": 2, "cvs_mm": 12.34, "mpjpe_mm": 38.50},
    {"k": 3, "cvs_mm":  8.21, "mpjpe_mm": 29.10},
    {"k": 4, "cvs_mm":  7.12, "mpjpe_mm": 26.42}
  ]
}
```

### 6.2 Markdown table

| Condition | REP (px) | RDP (mm) | TR (mm) | JLV | CVS (mm) | MPJPE (mm) |
|---|---|---|---|---|---|---|
| Clean | 2.34 | 4.12 | 6.55 | 0.02 | 7.12 | 26.42 |
| 2 px noise | 3.89 | 7.41 | 9.10 | 0.03 | 8.50 | 31.20 |
| 2-view | — | — | — | — | 12.34 | 38.50 |
| 0.5° rotation | 2.78 | 5.60 | 7.90 | 0.02 | 7.80 | 28.10 |

---

## 7. Implementation plan

### 7.1 New file: `motionflow_mv/eval/physical_alignment.py`

Responsibilities:

- `compute_reprojection_error(pred_3d, points_2d, K, R, t, mask=None)`
- `compute_ray_distance(pred_3d, points_2d, K, R, t, mask=None)`
- `compute_triangulation_residual(pred_3d, points_2d, K, R, t, mask=None)`
- `compute_physical_plausibility(pred_3d, parents, symmetry_pairs, foot_indices)`
- `compute_cross_view_stability(model, dataloader, device, view_subsets)`
- `compute_physical_alignment_report(pred_3d, points_2d, K, R, t, parents, ...)` — returns the full report dictionary.

### 7.2 New driver: `experiments/eval_physical_alignment.py`

Responsibilities:

- Load a checkpoint (reusing existing checkpoint-loading utilities).
- Run the clean evaluation and every perturbation axis.
- Write the JSON / Markdown / CSV outputs.
- Support `--smoke` mode for CPU testing.

### 7.3 New test: `tests/test_physical_alignment.py`

- Build a tiny synthetic `(B, T, V, J, 3)` input and random cameras.
- Assert all metrics are finite and non-negative.
- Assert `REP` increases monotonically with added 2-D noise.
- Assert `CVS` is lower when more views are active.

### 7.4 Optional integration with `benchmark_protocol.py`

Add a flag `physical_alignment=True` to `BenchmarkProtocol.run()` that also calls the new report and merges it into `results.json`.

---

## 8. Acceptance criteria

1. `python experiments/eval_physical_alignment.py --smoke` runs on CPU and writes a valid `physical_alignment_results.json`.
2. `pytest tests/test_physical_alignment.py -v` passes.
3. The clean-report values are reproducible across runs with the same seed.
4. `REP` and `RDP` correlate with `MPJPE` on a held-out validation set (Spearman ρ > 0.5).
5. The protocol does not require any training; it is inference-only.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Ray distance is sensitive to camera normalization | Scale predictions to the same unit as `K` (`unit_scale=1000.0`) before back-projection. |
| Triangulation residual is noisy with few views | Require at least 2 active views and fall back to median triangulation. |
| Physical plausibility metrics need skeleton metadata | Load `parents`/`symmetry_pairs`/`foot_indices` from the dataset config, with a fallback to H36M convention. |
| Combined perturbation space is large | Fix a deterministic grid and derive per-condition seeds from the base seed. |
| Existing models were not trained for this metric | Treat the protocol as diagnostic; do not change training until the metric is validated. |

---

## 10. Deliverables

1. `docs/eval_protocol_physical_alignment.md` (this file).
2. `motionflow_mv/eval/physical_alignment.py` — metric implementations.
3. `experiments/eval_physical_alignment.py` — evaluation driver.
4. `tests/test_physical_alignment.py` — unit tests.
5. Paper table: `docs/tables/icra2027/physical_alignment.md` (populated once checkpoints are evaluated).

---

## 11. Summary

This protocol adds a **physical-space alignment** lens to the existing 3-D pose evaluation. It reports reprojection, ray, triangulation, plausibility, and cross-view-stability metrics under clean and perturbed conditions. The goal is to complement `MPJPE`/`PA-MPJPE` with diagnostics that are directly tied to the multi-view geometry and physical realism of the predicted poses.
