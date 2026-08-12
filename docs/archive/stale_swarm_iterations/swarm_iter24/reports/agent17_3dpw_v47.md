# Agent-17: WebBridge 3DPW Loader Analysis for v47 → v48 Data Expansion

**Owner:** Agent-17 (ANALYZE)  
**Tracking issue:** #162 (v47 temporal aggregation)  
**Branch:** `v47-temporal`  
**Date:** 2026-08-09

---

## 1. Executive summary

The WebBridge 3DPW loader is already integrated into the mixed-dataset pipeline and works unchanged with the v46 sparse-view and upcoming v47 temporal aggregation paths.  However, **3DPW is currently used only as a *pseudo multi-view training source* (static 4-view re-projections)**.  For v48 domain generalization, 3DPW should be exploited as both:

1. a **true in-the-wild sparse-view evaluation target** (single moving camera — `actual` mode), and
2. a **source of cheap, variable-view synthetic training data** via on-the-fly re-projection with configurable pseudo rigs.

This report describes the current loader, what it does/does not support for v47, and proposes a minimal, backward-compatible v48 data-expansion plan centered on 3DPW actual-mode evaluation and variable pseudo-view generation.

---

## 2. Current 3DPW loader state

### 2.1 Files involved

| File | Role |
|------|------|
| `motionflow_mv/data/webbridge_mixed_dataset.py` | `WebBridgeCanonical17Dataset` / `WebBridgeMixedDataset`; pads views to `MAX_VIEWS=14` and maps 3DPW 24-joint SMPL skeleton to the canonical 17-joint layout. |
| `motionflow_mv/data/webbridge_loader.py` | `convert_3dpw` is a stub; actual conversion is done by `experiments/convert_3dpw_multiview.py`. |
| `experiments/convert_3dpw_multiview.py` | Converts raw 3DPW `.pkl` to canonical `.npz`. Supports `pseudo` (static virtual 4-view rig) and `actual` (single moving camera with per-frame extrinsics). |
| `configs/splits/webbridge_3dpw_train_val.yaml` | 3DPW-only manifest: 24 train, 12 val, 24 test pseudo files. |
| `configs/splits/webbridge_all_train_mixed.yaml` | Mixed manifest that already lists 3DPW pseudo files under `train_paths` / `val_paths`. |
| `configs/splits/webbridge_all_train_mixed_no_3dpw.yaml` | Baseline mixed manifest without 3DPW for ablation. |
| `tests/test_webbridge_3dpw_loader.py` | Smoke test for 3DPW `.npz` loading and 17-joint / 14-view padding. |

### 2.2 3DPW-specific skeleton mapping

In `webbridge_mixed_dataset.py`:

```python
"3dpw": np.array(
    [0, 2, 5, 8, 1, 4, 7, 6, 12, 15, 16, 18, 20, 17, 19, 21, 15],
    dtype=np.int64,
),
```

This maps the 24-joint SMPL skeleton to the canonical 17-joint H36M layout.  `head_top` is aliased to `head`, which is the only approximation.

### 2.3 On-disk inventory

As of 2026-08-09, `data/webbridge/3dpw/converted/` contains **123 pseudo `.npz` files**:

- `train/`: 24 files
- `validation/`: 12 files
- `test/`: 24 files

No `actual` (single moving camera) files are present on disk.

### 2.4 Loader test status

```text
$ python -m pytest tests/test_webbridge_3dpw_loader.py -v
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-0.0.0
2 passed in 4.91s
```

Both `return_view_mask=False` and `return_view_mask=True` paths pass.

---

## 3. What works for v47

1. **Temporal aggregation sees the same tensor layout as other domains.** 3DPW pseudo clips are loaded as `(T, 14, 17, 3)` and `(T, 17, 3)`, so `TemporalAggregationV47` can operate on them without code changes.
2. **View mask is already correct.** Pseudo 3DPW has only 4 real views; the remaining 10 padded views are masked out, exactly the sparse-view scenario v47 targets.
3. **Domain tag available.** `dataset_id == 5` lets v47 domain-aware losses or batch statistics distinguish 3DPW from H36M/MPI/AIST if desired.
4. **No loader-side changes are needed for v47 smoke/full runs** as long as the existing `webbridge_all_train_mixed.yaml` manifest is used.

---

## 4. Gaps for v47 and risks

| Gap | Why it matters for v47 | Risk level |
|-----|------------------------|------------|
| `actual` mode not converted | v47 temporal aggregation is meant to help **real** sparse capture (1–2 views). 3DPW `actual` is the only in-the-wild single-camera source we have. | High |
| Per-frame camera arrays are ignored by the loader | `actual` mode stores `camera_K_frames`, `camera_R_frames`, `camera_t_frames`, but `WebBridgeCanonical17Dataset` reads only the static `camera_K/R/t` slots. | High (for `actual`) |
| Pseudo view count is fixed at 4 | Cannot ablate 2/3/4/6 pseudo views without re-converting the whole dataset. | Medium |
| No runtime re-projection augmentation | v47 could benefit from temporally-consistent virtual-view synthesis, but the loader does not generate it on-the-fly. | Medium |
| Binary confidence values in pseudo mode | `confidences` are visibility masks from positive camera depth, not detector uncertainty. v47 reliability/temporal heads should not assume Gaussian confidence semantics. | Low |

---

## 5. Proposed v48 data expansion

The guiding principle for v48 is: **reuse 3DPW as a domain-generalization and sparse-view test bed, not just as extra pseudo multi-view training data.**

### 5.1 Phase A — Convert and evaluate 3DPW `actual` mode (high priority)

**Goal:** create a clean in-the-wild sparse-view evaluation set for v48.

- Run `experiments/convert_3dpw_multiview.py` with `--mode actual` on the same 60 train/val/test `.pkl` files.
- Store outputs under `data/webbridge/3dpw/actual/`, mirroring the pseudo directory layout.
- Add `configs/splits/webbridge_3dpw_actual_eval.yaml` listing only `actual` files.

**Evaluation:**

- Extend `experiments/eval_variable_views.py` to load `actual` files.
- Report `MPJPE@1` (single moving camera) and `MPJPE@2` (temporal aggregation + synthetic second view) for v47 vs v48.
- This becomes the primary v48 domain-generalization metric.

### 5.2 Phase B — Variable pseudo-view generation on-the-fly (medium priority)

**Goal:** train with 2/3/4/6 virtual views without maintaining multiple copies of the dataset.

- In `experiments/convert_3dpw_multiview.py`, keep the default `n_views=4` but allow a list `--n_views 2 3 4 6` in batch mode.
- Store the sampled `n_views` in the `.npz` so the loader mask is correct.
- Alternatively, generate only the 4-view pseudo files and perform runtime re-projection in a new augmentation module (`motionflow_mv/data/view_synthesis_aug_v48.py`) when more/fewer views are needed.

**Recommended v48 default:** generate 4-view pseudo files (same as today) and synthesize variable views at training time to avoid disk bloat.

### 5.3 Phase C — 3DPW as a domain-generalization target (high priority)

**Goal:** measure how well v48 trained on H36M/MPI/AIST transfers to in-the-wild 3DPW.

- Create a **target-only eval manifest**: `configs/splits/webbridge_3dpw_target_eval.yaml` with train/val/test splits from the same 60 files but tagged as target domain.
- Add a **leave-one-domain-out** smoke script that trains on H36M+MPI+AIST and tests on 3DPW pseudo and actual.
- Surface this in `experiments/eval_cross_dataset_generalization.py` if it still exists, or in a new lightweight v48 eval script.

### 5.4 Phase D — CMU Panoptic and future in-the-wild data (future work)

`convert_panoptic` is still a stub and raw Panoptic data is not on disk.  Keep this as a post-v48 stretch goal.  Do not block v48 on it.

---

## 6. Concrete v48 files to touch

| File / New file | Change | Owner type |
|-----------------|--------|------------|
| `experiments/convert_3dpw_multiview.py` | Add batch `--mode actual` convenience; support list `--n_views` for variable pseudo rigs. | IMPLEMENT |
| `configs/splits/webbridge_3dpw_actual_eval.yaml` | New manifest for 60 `actual` files. | IMPLEMENT |
| `configs/splits/webbridge_3dpw_variable_views_train.yaml` | Optional: point to 4-view pseudo files and store per-file `n_views` metadata. | IMPLEMENT |
| `motionflow_mv/data/webbridge_3dpw_actual_loader.py` | Thin wrapper over `WebBridgeCanonical17Dataset` that returns per-frame cameras from `camera_K_frames/R_frames/t_frames`. | IMPLEMENT |
| `experiments/eval_variable_views.py` | Add `--mode actual` and `MPJPE@1` / `MPJPE@2` reporting. | EVAL |
| `motionflow_mv/data/view_synthesis_aug_v48.py` | On-the-fly re-projection from 3D GT with sampled virtual cameras (optional). | IMPLEMENT |
| `scripts/run_v48_3dpw_actual_eval.sh` | Smoke/eval script for local 4090. | IMPLEMENT |

---

## 7. Loader changes required

For Phase A, the existing `WebBridgeCanonical17Dataset` **already loads `actual` files** because it only reads the static `camera_K/R/t` slots.  However, to use the true moving camera, a small wrapper is needed:

```python
# motionflow_mv/data/webbridge_3dpw_actual_loader.py (proposed)
class WebBridge3DPWActualDataset(WebBridgeCanonical17Dataset):
    """Loads a 3DPW 'actual' .npz and returns per-frame camera parameters."""

    def __init__(self, npz_path: str, clip_len: int, ...):
        super().__init__(npz_path, dataset_name="3dpw", clip_len=clip_len, ...)
        # Override static cameras with per-frame arrays stored by convert_3dpw_multiview.py
        self.camera_K = self.camera_K  # placeholder, kept for compatibility
        self.camera_K_frames = torch.from_numpy(np.load(npz_path)["camera_K_frames"]).float()
        self.camera_R_frames = torch.from_numpy(np.load(npz_path)["camera_R_frames"]).float()
        self.camera_t_frames = torch.from_numpy(np.load(npz_path)["camera_t_frames"]).float()
```

For v48 training with variable pseudo views, no loader change is needed if the augmentation is applied in the training loop; the loader just needs the correct `view_mask`.

---

## 8. Tests and validation

- [ ] Run `pytest tests/test_webbridge_3dpw_loader.py -v` after any loader change.  
- [ ] Convert a single 3DPW `actual` file and verify the `.npz` contains `camera_K_frames`, `camera_R_frames`, `camera_t_frames`.  
- [ ] Load the `actual` file with the proposed `WebBridge3DPWActualDataset` and confirm per-frame camera shapes `(T, 1, 3, 3)`, `(T, 1, 3, 3)`, `(T, 1, 3)`.  
- [ ] Run a v47 smoke that includes 3DPW pseudo files to confirm temporal aggregation does not regress on the existing 4-view pseudo data.  
- [ ] Add a v48 test that drops 3DPW pseudo views to 2 and verifies `MPJPE@2` is computed correctly.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| `actual` mode has only one view, so v47 temporal aggregation must carry the full pose burden. | Use it as an evaluation target, not as the only training signal. |
| Per-frame moving cameras complicate the existing geometry-fusion path. | Keep `actual` evaluation separate from training; use a thin wrapper, not a rewrite of `webbridge_mixed_dataset.py`. |
| Variable pseudo-view generation could create invalid view masks. | Generate `view_mask` from the sampled `n_views` in the same place the loader already computes it. |
| 3DPW skeleton approximation (`head_top` → `head`) biases head error. | Report joint-wise MPJPE and note the approximation in v48 results. |

---

## 10. Open questions

1. Do we want to regenerate all 60 3DPW files in `actual` mode now, or only the test/validation subsets for v48 eval?  
2. Should the v48 variable pseudo-view augmentation live in the loader or in the training script?  (Loader = simpler; training script = more flexible.)  
3. Is the existing `experiments/eval_cross_dataset_generalization.py` still maintained, or should v48 domain-transfer eval use `eval_variable_views.py`?  
4. Should 3DPW `actual` clips be included in the v47 full A800 run as an auxiliary target domain, or kept strictly for v48 evaluation?
