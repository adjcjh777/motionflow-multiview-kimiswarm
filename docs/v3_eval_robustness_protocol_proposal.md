# v3 Evaluation & Robustness Protocol Proposal

**Target:** ICRA / CVPR 2027 paper-ready evaluation for `OmniMultiViewFusionV3`  
**Scope:** Read-only research and protocol design; no changes to running training, Docker, or read-only data.  
**Last updated:** 2026-08-07

---

## 1. Current state

### 1.1 Model

`motionflow_mv/fusion/omniview_fusion_v3.py` implements `OmniMultiViewFusionV3`, which extends v2 with:

- `use_multiscale_fusion`: hierarchical temporal/joint/cross-view fusion (`_HierarchicalMultiscaleFusion`).
- `use_camera_conditioning`: per-view camera-parameter embeddings (`_CameraConditioning`).
- `use_epipolar_bias`: epipolar-distance attention bias (`_CameraConditionedEpipolarBias` + `EpipolarBiasedTransformerEncoderLayer`).

The v3 forward signature is identical to v2: `model(x, K=..., R=..., t=...)` returns `(pred_3d, weights, visibility, covariance, epipolar_loss)` plus optional `pp_delta`/`focal_scale`.  
`__main__` smoke tests for single-frame, multi-frame, and full-ablation modes already pass.

### 1.2 Training

`experiments/train_omniview_fusion_v3_mpiinf3dhp.py` provides a self-contained trainer using `TrainerV2` and the same `.npz` data loader as v2. It supports `--use_multiscale_fusion`, `--use_camera_conditioning`, and `--use_epipolar_bias` flags.

### 1.3 Existing robustness infrastructure

- `motionflow_mv/eval/benchmark_protocol.py` — `BenchmarkProtocol` + `BenchmarkConfig`; model-agnostic evaluation harness that writes `results.json`/`results.txt`.
- `motionflow_mv/eval/metrics.py` — MPJPE, PA-MPJPE, root-relative, velocity, bone-length, PCK@ thresholds, PCK-AUC, per-joint variants.
- `motionflow_mv/calibration/perturb.py` — `perturb_cameras`, `perturb_intrinsics`, `perturb_extrinsics`, `perturb_radial_distortion`.
- `motionflow_mv/data/occlusion_aug.py` — `random_occlude_views`, `random_occlude_joints`.
- `experiments/run_robustness_matrix.py` — fixed matrix (noise, view dropout, principal-point) for the `eval_full_metrics.py` model family.
- `experiments/prototypes/run_extended_robustness_matrix.py` — adds joint occlusion + combined perturbations.
- `experiments/prototypes/swarm_iter18/eval_robustness_matrix_v2.py` — adds extrinsic rotation/translation, focal, and distortion perturbations.

### 1.4 Existing docs

- `docs/omniview_fusion_v3_design.md` — architecture design.
- `docs/paper_experiments.md` — roadmap with ablation matrix (Runs A–H).
- `docs/extended_robustness_matrix.md` — extended matrix usage for older models.
- `docs/tables/icra2027/robustness.md` — example robustness table.

---

## 2. Gap analysis

What is **missing** for a v3-specific robustness protocol:

1. **No v3 entry in the evaluation registry.** `experiments/eval_full_metrics.py` only registers pre-Omni model families (residual, campe, bayesian_tri, etc.). `OmniMultiViewFusionV3` is not registered, so the existing `run_robustness_matrix.py` cannot load a v3 checkpoint.
2. **No dedicated v3 eval/robustness script.** Training exists, but there is no `experiments/eval_omniview_fusion_v3.py` or `experiments/run_robustness_matrix_v3.py`.
3. **No v3 smoke tests.** `tests/test_benchmark_protocol.py` and `tests/test_extended_robustness_matrix.py` cover the generic protocol and the older prototype, but no test exercises v3 evaluation end-to-end.
4. **Calibration perturbations are split across prototypes.** The most complete camera-perturbation logic is in `swarm_iter18/eval_robustness_matrix_v2.py` and is hard-coded to `RayAttentionFusionModelTemporalResidual`; it is not reusable for v3.
5. **No variable-view robustness harness for v3.** `motionflow_mv/fusion/variable_view_inference.py` exists, but no v3 script evaluates robustness to 2/3/4 active views.
6. **No standardized report format for v3.** We need a single JSON/Markdown/CSV output that records which v3 ablation flags were used.

---

## 3. Proposed v3 evaluation & robustness protocol

### 3.1 Core principle

Reuse existing, tested components wherever possible:

- Data loading: `TemporalClipDataset` pattern from `experiments/eval_full_metrics.py` and `experiments/train_omniview_fusion_v3_mpiinf3dhp.py`.
- Metrics: `motionflow_mv.eval.metrics.compute_all_metrics`.
- Perturbations: `motionflow_mv.calibration.perturb` + `motionflow_mv.data.occlusion_aug`.
- Reporting: `motionflow_mv.eval.benchmark_protocol.BenchmarkProtocol`.

Add a thin v3-specific driver that instantiates `OmniMultiViewFusionV3`, applies the perturbation matrix, and writes reproducible manifests.

### 3.2 Robustness axes

Based on the architecture and existing prototypes, the v3 matrix should cover:

| Axis | Levels | Implementation reference |
|---|---|---|
| Clean baseline | — | — |
| 2-D keypoint noise | 0.5 px, 1.0 px, 2.0 px | `run_robustness_matrix.py::apply_noise` |
| Per-joint occlusion | 10%, 20%, 30% | `motionflow_mv.data.occlusion_aug.random_occlude_joints` |
| Whole-view dropout | 10%, 30%, 50% | `motionflow_mv.data.occlusion_aug.random_occlude_views` |
| Extrinsic rotation noise | 0.5°, 1.0° | `motionflow_mv.calibration.perturb.perturb_extrinsics` |
| Extrinsic translation noise | 5 mm, 10 mm | same as above |
| Focal length error | 1%, 2% | `perturb_intrinsics` |
| Principal-point error | 3 px, 5 px | `perturb_intrinsics` |
| Radial distortion | k1 = 0.01, 0.05 | `perturb_radial_distortion` (used in `swarm_iter18`) |
| Combined | noise + occlusion + dropout | pairwise / three-way combos |

### 3.3 Variable-view axis

Because v3 is fixed-view at training time but may face missing views at test time, add a dedicated axis using `VariableViewInferenceWrapper`:

| Active views | Method |
|---|---|
| 2 views | mask inactive views to zero confidence |
| 3 views | same |
| 4 views (full) | same |

This directly tests the visibility-gating and epipolar-bias behavior under realistic deployment scenarios.

---

## 4. Concrete implementation steps

### Step 1: Create `experiments/eval_omniview_fusion_v3.py`

A new evaluation driver specifically for `OmniMultiViewFusionV3`.

Responsibilities:

- Load a canonical `.npz` sequence (same format as v2/v3 trainers).
- Build `OmniMultiViewFusionV3` from CLI args: `--use_multiscale_fusion`, `--use_camera_conditioning`, `--use_epipolar_bias`, `--d`, `--residual_hidden`, etc.
- Load checkpoint with `strict=False` (v3 can warm-start from v2).
- Run clean evaluation and produce a `results.json` via `BenchmarkProtocol`.
- Support `--output_json` for downstream tooling.

Reuse:

- `TemporalClipDataset` and `collate_fn` from `experiments/eval_full_metrics.py` (or copy locally to avoid coupling).
- `compute_all_metrics` / `summarize_metrics` from `motionflow_mv.eval.metrics`.

### Step 2: Create `experiments/run_robustness_matrix_v3.py`

A v3 robustness matrix script modeled on `run_robustness_matrix.py` and `prototypes/run_extended_robustness_matrix.py`.

Responsibilities:

- Define the condition matrix in Section 3.2.
- Apply perturbations in deterministic order: noise → joint occlusion → view dropout → camera perturbation.
- Evaluate each condition and write:
  - `robustness_matrix.json` — full per-condition metrics.
  - `robustness_matrix.md` — Markdown table.
  - `robustness_matrix.csv` — CSV for plotting.
- Record model ablation flags and checkpoint path in the JSON manifest.

Key reuse:

- `motionflow_mv.calibration.perturb.perturb_cameras`
- `motionflow_mv.calibration.perturb.perturb_radial_distortion`
- `motionflow_mv.data.occlusion_aug.random_occlude_joints`
- `motionflow_mv.data.occlusion_aug.random_occlude_views`
- `motionflow_mv.eval.metrics.compute_all_metrics`

### Step 3: Add variable-view robustness helper

Create `motionflow_mv/eval/variable_view_robustness.py` (or add to existing `variable_view_inference.py`) with:

```python
def evaluate_with_active_views(model, loader, device, active_views: int | list[int]):
    """Wrap a fixed-view v3 model and evaluate with a subset of views."""
```

This function should:

- Use `prepare_variable_view_input` to mask inactive views.
- Run inference and accumulate metrics.

Then expose it in `run_robustness_matrix_v3.py` under conditions `active_views_2`, `active_views_3`, etc.

### Step 4: Add smoke tests

Create `tests/test_omniview_fusion_v3_eval.py`:

- Build a tiny synthetic `.npz` with 4 views, 17 joints, 30 frames.
- Instantiate `OmniMultiViewFusionV3` with all three flags enabled.
- Run `eval_omniview_fusion_v3.py` in CPU mode and assert `results.json` exists with expected keys.
- Run `run_robustness_matrix_v3.py` in CPU smoke mode (no checkpoint required) and assert all conditions are evaluated.

### Step 5: Update docs and paper tables

- Write `docs/v3_eval_robustness_protocol.md` (or keep this proposal as the working doc).
- Populate `docs/tables/icra2027/robustness_matrix.md` with v3 results once checkpoints are available.
- Update `docs/paper_experiments.md` Section 5 to reference the new v3 protocol.

### Step 6: Validate against existing v2 baseline

Before finalizing the protocol:

1. Train or use an existing v3 checkpoint (or v2 checkpoint loaded into v3 with ablations disabled).
2. Run `run_robustness_matrix_v3.py` for v3-full and the v2-equivalent ablation.
3. Compare the Markdown table to `docs/tables/icra2027/robustness.md` to ensure the protocol is calibrated.

---

## 5. Recommended ablation matrix

Use `run_robustness_matrix_v3.py` for each run:

| Run | Model | `use_multiscale_fusion` | `use_camera_conditioning` | `use_epipolar_bias` | Purpose |
|---|---|---|---|---|---|
| E (v2 baseline) | v3 | ✗ | ✗ | ✗ | Reproduce v2 numbers in v3 shell |
| B | v3 | ✗ | ✓ | ✓ | Measure multi-scale fusion impact |
| C | v3 | ✓ | ✗ | ✓ | Measure camera conditioning impact |
| D | v3 | ✓ | ✓ | ✗ | Measure epipolar bias impact |
| A (v3 full) | v3 | ✓ | ✓ | ✓ | Final model |

This matches `docs/paper_experiments.md` Runs A–E and lets the protocol slot directly into the paper.

---

## 6. Files to create / modify

### Create

- `experiments/eval_omniview_fusion_v3.py`
- `experiments/run_robustness_matrix_v3.py`
- `motionflow_mv/eval/variable_view_robustness.py`
- `tests/test_omniview_fusion_v3_eval.py`
- `docs/v3_eval_robustness_protocol.md` (this proposal or final version)

### Modify

- `docs/paper_experiments.md` — update Section 5 robustness tests and Section 8 next steps.
- `docs/tables/icra2027/robustness_matrix.md` — populate with v3 numbers.
- `motionflow_mv/eval/benchmark_protocol.py` — optional: add a helper to load v3-compatible manifests (currently not required).

---

## 7. Acceptance criteria

1. `python experiments/eval_omniview_fusion_v3.py --smoke` runs on CPU without a checkpoint and writes a valid `results.json`.
2. `python experiments/run_robustness_matrix_v3.py --smoke` runs on CPU and evaluates every condition in the matrix.
3. `pytest tests/test_omniview_fusion_v3_eval.py -v` passes.
4. The protocol can load a real v3 checkpoint from `experiments/train_omniview_fusion_v3_mpiinf3dhp.py` and produce a Markdown table comparable to `docs/tables/icra2027/robustness.md`.
5. The protocol reports all v3 ablation flags in the JSON manifest so results are reproducible.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| v3 checkpoint not yet ready | Implement smoke-mode drivers that run with random weights first; real checkpoints slot in later. |
| Multi-scale fusion changes temporal behavior | Keep `clip_len` identical between v2 and v3 evaluations; report per-frame metrics after flattening. |
| Camera perturbation order is undefined | Fix perturbation order in `run_robustness_matrix_v3.py` and derive per-condition seeds from the base seed. |
| Variable-view wrapper changes attention patterns | Document that wrapper uses zero-confidence masking; compare with explicit retraining when feasible. |
| Existing `eval_full_metrics.py` model registry clash | Keep v3 driver separate; do not add Omni to the legacy registry to avoid breaking older scripts. |

---

## 9. Next actions (priority order)

1. **Implement `experiments/eval_omniview_fusion_v3.py`** — single clean-evaluation driver with smoke mode.
2. **Implement `experiments/run_robustness_matrix_v3.py`** — full robustness matrix reusing `calibration.perturb` and `data.occlusion_aug`.
3. **Add variable-view evaluation** via `motionflow_mv/eval/variable_view_robustness.py` and include active-view conditions.
4. **Write `tests/test_omniview_fusion_v3_eval.py`** to lock the smoke behavior.
5. **Run smoke tests**, then run against the first available v3 checkpoint and populate `docs/tables/icra2027/robustness_matrix.md`.
