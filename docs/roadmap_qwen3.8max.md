# qwen3.8max Next-Iteration Roadmap

> **Target:** CVPR 2027 (~13 weeks remaining)  
> **Current date:** 2026-08-11  
> **Paper pivot:** sparse-view / cross-domain robustness (absolute MPJPE records are no longer trustworthy)

---

## Context (do not start before reading)

- H36M true GT is now in `data/h36m_true_gt/` and is non-circular. The standard protocol (S1,5,6,7,8 → S9/S11) is reliable.
- **A800 ablations are running**:
  - GPU 4: `v25_true_gt_baseline_fix` — Epoch 1 val **46.53 mm**.
  - GPU 6: `v25_true_gt_geometry_regularization_a800` — Epoch 1 val **46.75 mm**.
- **Local RTX 4090 is idle** and reserved for quick smoke/diagnostics only.
- v80 H36M true-GT medium has finished: **39.98 mm** at epoch 4, then overfit to 133.71 mm by epoch 8. Best known v80 remains the A800 v2 checkpoint at **39.70 mm**.
- v25 H36M true-GT medium finished with test **43.93 mm**; the old validation log of 72.80 mm was inflated by a missing `view_mask`.
- v57 H36M true-GT medium is **DONE**: final **78.76 mm**; true best **75.16 mm** @ epoch 3 was not saved because the trainer monitored `loss` instead of `mpjpe`. The trainer is now fixed.
- `agent-67` has finished AIST++ smoke integration; full cross-dataset training remains future work.
- A800-D and the `motionflow` Docker container are **read-only**; do not write or start anything there.
- Latest handoff: `docs/handoff_qwen3.8max.md`  
  Leaderboards: `docs/results_true_gt_h36m.md`, `docs/results_true_gt_shelf_campus.md`

---

## Next Tasks to Run

### 1. Diagnose and mitigate learned-model overfitting on H36M true-GT (P0)

**Goal:** Understand the shared overfitting pattern across v25/v80/v57 on the true-GT protocol, then run targeted ablations/smoke experiments until a stable training recipe is found.

**Current state:**
- v25 medium finished (agent-51). Results:
  - Test MPJPE: **43.93 mm** (corrected).
  - Old validation log: **72.80 mm** @ epoch 2 (inflated due to missing `view_mask`).
- v80 medium finished locally. Results:
  - Epoch 4: **39.98 mm** (best).
  - Epoch 8: overfit to **133.71 mm**.
- v57 medium finished: final **78.76 mm**; true best **75.16 mm** @ epoch 3 (not saved).
- A800 v25 ablations are **running**:
  - `v25_true_gt_baseline_fix` (GPU 4) — Epoch 1 val **46.53 mm**.
  - `v25_true_gt_geometry_regularization_a800` (GPU 6) — Epoch 1 val **46.75 mm**.
- Training loss keeps decreasing while validation MPJPE grows, indicating severe overfitting.
- Iskakov leads at 23.35 mm; conf-DLT is 25.67 mm.

**Hypotheses to test (in order of expected ROI):**
1. All learned models were optimized for circular labels; on true GT they must learn real geometry and current recipes lack regularization.
2. Too few training steps per epoch (`train_samples=1024` → only 64 steps/epoch).
3. Learning rate too aggressive / warmup too short.
4. Missing weight decay, early stopping, and reduced augmentation.
5. Auxiliary loss imbalance pushing the model into degenerate minima.

**Concrete steps:**
1. Monitor the running A800 v25 ablations.
2. If Epoch 1–2 numbers stay ≤ ~50 mm and do not diverge, continue to mixed-dataset ablation.
3. If the regularised recipe works, apply it to **v80** first (current best learned baseline).
4. Document findings in `docs/v25_true_gt_failure_mode.md` (or `docs/true_gt_overfitting_diagnosis.md`).

**Success criteria:**
- A clear failure-mode diagnosis documented for the shared v25/v80/v57 overfitting pattern.
- At least one smoke ablation shows validation MPJPE stable or improving after epoch 2.
- A revised medium recipe (applicable to v25/v80/v57) is defined and queued.

**Blocked by:** A800 GPU 4/6 are occupied by v25 ablations.

---

### 2. Re-run v57 with fixed trainer (P0)

**Goal:** Confirm the true best epoch is now saved and beat the stale saved best.

**Decision now that v57 has finished:**
- v57 reached **75.16 mm** observed but saved ckpt was worse.
- Trainer now monitors `mpjpe`, so the true best epoch will be saved.
- Once A800 GPU is free, run `scripts/run_v57_true_gt_medium_a800.sh`.

**Success criteria:**
- Best saved checkpoint matches the best observed epoch.
- Result is stable and ≤ 75 mm.

**Blocked by:** Task 1 (v25 ablations occupying A800).

---

### 3. Re-run learned models on non-circular Shelf/Campus with longer/mixed training (P1)

**Goal:** Determine whether MotionFlow variants can close the gap to Iskakov (128.73 mm) on real detected-2D labels.

**Current state:**
- On `configs/splits/shelf_campus_detected_smoke.yaml`:
  - Iskakov: 128.73 mm
  - v80 long (25 epochs): 276.49 mm
  - v57 long (25 epochs): 306.45 mm
  - v25 smoke (3 epochs): 430.67 mm
- All learned models are far behind Iskakov.

**Concrete steps:**
1. Wait for the winning H36M true-GT recipe from Task 1/2.
2. Run the winner long on Shelf/Campus detected.
3. Run cross-dataset transfer: pretrain on H36M true-GT, fine-tune on Shelf/Campus.
4. Update `docs/results_true_gt_shelf_campus.md`.

**Blocked by:** Task 1/2 (stable H36M recipe).

---

### 4. Fix MPI-INF-3DHP detected-2D / camera / label alignment

**Goal:** Make the real detected-2D MPI-INF-3DHP data usable for standard-protocol benchmarking.

**Current state:**
- Real detected 2D already generated: 16 `_m.npz` files.
- `s_02_seq_02` was removed due to severe misalignment.
- DLT baseline on remaining files is still ~326–400 mm.
- RTMPose detector batch-dimension bug fixed in `scripts/generate_mpi_detected_2d.py`.

**Concrete steps:**
1. Re-generate MPI detected-2D with RTMPose.
2. Re-run `scripts/run_mpi_dlt_baseline.py` until DLT drops to ~20–30 mm.
3. Only then run DLT/v25/v80/v57 baselines on the standard MPI protocol.
4. Create `docs/results_true_gt_mpi.md`.

**Success criteria:**
- DLT baseline on real detected 2D is comparable to literature (~20–30 mm).
- v25/v80 smoke numbers recorded.

**Blocked by:** Need source MPI videos and RTMPose setup.

---

### 5. Repository hygiene (P2)

**Goal:** Reduce branch clutter and keep handoff docs current.

**Current state:**
- Many stale local/remote branches exist (`git branch -a`).
- Handoff docs were just refreshed.

**Concrete steps:**
1. Review and delete merged/stale local branches.
2. Run `git remote prune origin --dry-run` and confirm safe deletions.
3. Update `docs/git_status_summary.md`.

---

## Ordering and concurrency

1. **Task 1** is highest priority (v25 ablations already running on A800).
2. **Task 2** runs as soon as A800 GPU is free.
3. **Task 3** follows Task 2.
4. **Task 4** can proceed in parallel (CPU-only video/detector work) while GPU tasks run.
5. **Task 5** is housekeeping and can be done at any time.

---

## Anti-patterns to avoid

- Do not run more than one GPU training/diagnostic task at a time on the local RTX 4090.
- Do not write, start, or modify anything on A800-D or its Docker `motionflow` service.
- Do not use circular-label `.npz` (`data/h36m_hf/`, `data/webbridge/h36m*.npz`) for model selection.
- Do not launch a full v80/v57 medium before the v25 ablations identify a stable recipe.
- Do not duplicate agent-67's AIST++ work.

---

## Exit criteria for this roadmap

- [ ] Task 1: Shared v25/v80/v57 true-GT failure mode diagnosed and a revised training recipe is found.
- [ ] Task 2: v57 re-run with fixed trainer; best checkpoint correctly saved.
- [ ] Task 3: Shelf/Campus detected results improved or transfer experiments documented.
- [ ] Task 4: MPI-INF-3DHP real-detected-2D pipeline created and baselines run.
- [ ] Task 5: Stale branches pruned and handoff docs are current.

After these tasks, the project will have an honest, cross-dataset evaluation foundation (H36M + Shelf/Campus + MPI) and a stable learned baseline to support the sparse-view / cross-domain robustness paper story.
