# WebBridge Training Manifest Expansion Proposal

**Target:** ICRA / CVPR 2027 multi-view human pose estimation  
**Scope:** Expand the WebBridge mixed-dataset training manifest to increase data volume, domain diversity, and downstream generalization without changing model code.  
**Date:** 2026-08-09

---

## 1. Current State

### 1.1 Loader support

`motionflow_mv/data/webbridge_loader.py` already implements canonical `.npz` converters for:

- **Human3.6M** (`convert_human36m`) — 4 views, 17 joints, meters
- **MPI-INF-3DHP** (`convert_mpiinf3dhp`) — up to 14 views, 28 joints → 17 joints
- **AIST++** (`convert_aistpp`) — 9 views, 17 joints, meters
- **Shelf / Campus** (`convert_shelf_campus`) — 4 views, 17 joints
- **3DPW** (`convert_3dpw`) — stub only; pseudo-converted files exist under `data/webbridge/3dpw/converted/`
- **CMU Panoptic** (`convert_panoptic`) — stub only

`motionflow_mv/data/webbridge_mixed_dataset.py` unifies all sources into a common 17-joint skeleton, pads cameras to `MAX_VIEWS=14`, and returns `(x, y, K, R, t, dataset_id)`.

### 1.2 Existing manifests

| Manifest | Train files | Val files | Contents |
|----------|------------|-----------|----------|
| `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml` | 29 | 16 | S1 H36M + S1,S3-S8 MPI (subject 9 val) |
| `configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml` | 89 | 16 | S1,S5-S8 H36M + S1,S3-S8 MPI |
| `configs/splits/webbridge_all_train.yaml` | mixed | mixed | H36M, MPI, 3DPW, AIST++ (per-subset) |
| `configs/splits/webbridge_all_train_mixed.yaml` | mixed | mixed | Same as above in 17-joint mixed format |

### 1.3 On-disk inventory (as of 2026-08-09)

| Source | Available `.npz` files | Notes |
|--------|------------------------|-------|
| H36M (`h36m_meters`) | 105 | Subjects 01, 05-08 train; 02, 03, 04, 09-16 present on disk |
| MPI-INF-3DHP (`mpi_inf_3dhp`) | 16 (v14 meter) | Subjects 01-08, 2 sequences each |
| AIST++ (`aistpp_canonical`) | 1,408 | 9 cameras, 17 joints, already canonical |
| 3DPW (`3dpw/converted`) | 123 | 24-joint pseudo-labels, train/test/validation splits |
| Shelf/Campus (`shelf_campus`) | 4 | train/val for each |

---

## 2. Proposed Expansion

### 2.1 Phase 1: H36M full-subject train split (low effort, high value)

Currently the expanded manifest only uses subjects 01, 05, 06, 07, 08 for training. All remaining subjects (02, 03, 04, 09-16) are already converted to canonical `.npz` files.

**Proposal:**

- Add H36M subjects **02, 03, 04, 10, 11, 12, 13, 14, 15, 16** to the training manifest.
- Keep **subject 09** as the canonical validation subject.
- This increases the H36M training corpus from 75 files to **~165 files**.

**Validation rule:**

```text
H36M subject 09  -> val
All other H36M subjects -> train
```

### 2.2 Phase 2: MPI-INF-3DHP all-subject training (medium effort)

Currently subject 02 is reserved for validation and subjects 01, 03-08 are used for training. Subject 02 has two sequences (`seq_01`, `seq_02`).

**Proposal:**

- Keep subject 02 as the primary MPI validation subject.
- Optionally include subject 02 sequence 01 in training and reserve sequence 02 for validation, doubling the MPI training set.

### 2.3 Phase 3: AIST++ integration (medium effort)

AIST++ canonical files are 17-joint, 9-view, and meter-scaled — ideal for the mixed loader. They are not yet referenced in the standard mixed manifests.

**Proposal:**

- Add `data/webbridge/aistpp_canonical/*_multiview.npz` to the mixed manifest.
- Use the existing `_split_aist()` logic in `scripts/propose_webbridge_multiview_usage.py` (85/15 deterministic split by clip prefix) to avoid channel-level leakage.
- Expected addition: ~1,200 training files, ~200 validation files.

### 2.4 Phase 4: 3DPW inclusion (higher effort)

3DPW is single-camera + IMU, so the existing pseudo-converted files are not true multi-view. Before adding to the mixed loader:

- Verify pseudo-multi-view generation quality.
- Either map the 24-joint SMPL skeleton to the canonical 17-joint layout or train a separate 24-joint branch.

**Recommendation:** Add only after Phase 1-3 are validated and a 24→17 joint map is defined.

### 2.5 Phase 5: CMU Panoptic converter (future work)

The loader stub exists but no raw data is present. This is a longer-term data-acquisition task.

---

## 3. Suggested New Manifest

Introduce a single new manifest:

```
configs/splits/webbridge_h36m_mpi_aist_mixed_train_val_expanded.yaml
```

with:

- `train_paths`: all H36M except subject 09, all MPI-INF-3DHP sequences (with optional S2 seq 01), and AIST++ 85/15 train split.
- `val_paths`: H36M subject 09, MPI-INF-3DHP subject 02, and AIST++ 15% val split.
- `train_names` / `val_names`: dataset labels for `domain_balanced_sampler` and domain-loss weighting.

Optionally, update:

- `configs/splits/webbridge_all_train_mixed.yaml` to include the new H36M subjects and AIST++.

---

## 4. Implementation Steps

1. Run the existing proposal script with the expanded inventory:
   ```bash
   python scripts/propose_webbridge_multiview_usage.py \
       --root data/webbridge \
       --out-yaml configs/splits/webbridge_proposed_mixed.yaml \
       --out-report docs/swarm_iter_next/webbridge_multiview_usage_proposal.md
   ```
2. Inspect the generated YAML and report.
3. Copy / rename the generated YAML to `configs/splits/webbridge_h36m_mpi_aist_mixed_train_val_expanded.yaml`.
4. Run a CPU smoke test:
   ```bash
   python experiments/smoke_webbridge_mixed.py \
       --manifest configs/splits/webbridge_h36m_mpi_aist_mixed_train_val_expanded.yaml
   ```
5. If the smoke test passes, update `configs/benchmark_icra_cvpr_2027.yaml` and relevant benchmark configs to reference the new manifest.
6. Train a small smoke run (local RTX 4090) before scheduling on A800-D.

---

## 5. Validation & Risks

| Risk | Mitigation |
|------|------------|
| Domain imbalance (H36M dominates) | Use `domain_balanced_sampler.py` or per-domain loss weights (v41) |
| Variable view counts | `webbridge_mixed_dataset.py` already pads to 14 views |
| Unit mismatch | Only include meter-scaled files (`_m.npz` or `aistpp_canonical`) |
| AIST++ clip leakage across channels | Split by clip prefix, not by individual channel file |
| 3DPW skeleton mismatch | Defer until a 24→17 joint map is validated |
| Longer training time | Start with a smoke subset; scale up only after validation |

---

## 6. Definition of Done

- [ ] `configs/splits/webbridge_h36m_mpi_aist_mixed_train_val_expanded.yaml` generated and reviewed.
- [ ] New manifest passes a CPU smoke test with `webbridge_mixed_dataset.py`.
- [ ] Per-dataset file counts and domain balance are documented in this file or a follow-up report.
- [ ] (Optional) Local RTX 4090 smoke run reaches a stable first-epoch val MPJPE comparable to the existing manifest.
- [ ] No changes to A800-D or remote hosts until local validation is complete.

---

## 7. References

- `motionflow_mv/data/webbridge_loader.py`
- `motionflow_mv/data/webbridge_mixed_dataset.py`
- `scripts/propose_webbridge_multiview_usage.py`
- `configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml`
- `docs/webbridge_scaling_plan_proposal.md`
