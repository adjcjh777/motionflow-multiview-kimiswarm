# qwen3.8max Next-Iteration Roadmap

> **Target:** CVPR 2027 (~13 weeks remaining)
> **Current date:** 2026-08-11
> **Paper pivot:** sparse-view / cross-domain robustness (absolute MPJPE records are no longer trustworthy)

---

## Context (do not start before reading)

- H36M true GT is now in `data/h36m_true_gt/` and is non-circular. The standard protocol (S1,5,6,7,8 → S9/S11) is reliable.
- `agent-51` has finished the v25 medium run. It reached **72.80 mm** at epoch 2 but diverged to **207.62 mm** by epoch 8.
- v80 H36M true-GT medium has finished: **39.98 mm** at epoch 4, then overfit to 133.71 mm by epoch 8. Best known v80 is still the A800 v2 checkpoint at **39.70 mm**.
- v57 H36M true-GT medium status is **unconfirmed**; `nvidia-smi` shows no active GPU training process, so it is either not running, running on CPU, or already finished.
- `agent-67` has finished AIST++ smoke integration; full cross-dataset training remains future work.
- A800-D and the `motionflow` Docker container are **read-only**; do not write or start anything there.
- **GPU is currently idle** (`nvidia-smi` shows ~22–35% utilisation, ~2.0 GB memory, no active `python.exe` GPU training process). The previously reported v57 H36M true-GT medium run is not observed on the GPU; confirm process state before launching any new task.

Latest handoff: `docs/handoff_qwen3.8max.md`  
Leaderboards: `docs/results_true_gt_h36m.md`, `docs/results_true_gt_shelf_campus.md`

---

## Next Tasks to Run

### 1. Diagnose and mitigate learned-model overfitting on H36M true-GT (P0)

**Goal:** Understand the shared overfitting pattern across v25/v80/v57 on the true-GT protocol, then run targeted ablations/smoke experiments until a stable training recipe is found.

**Current state:**
- v25 medium finished (agent-51, local RTX 4090). Results:
  - Epoch 1: 83.19 mm
  - Epoch 2: **72.80 mm** (best)
  - Epoch 3–8: monotonic rise to **207.62 mm**
- v80 medium finished locally. Results:
  - Epoch 4: **39.98 mm** (best)
  - Epoch 8: overfit to **133.71 mm**
  - Best known v80 remains the A800 v2 checkpoint at **39.70 mm**.
- v57 medium status is **unconfirmed**; `nvidia-smi` shows no active GPU training process.
- Training loss keeps decreasing while validation MPJPE grows, indicating severe overfitting to the training distribution.
- Iskakov leads at 23.35 mm; DLT (conf-weighted) is 25.87 mm.
- A root-cause analysis already exists in `docs/v25_divergence_diagnosis.md` and should be extended to cover v80/v57.

**Hypotheses to test (in order of expected ROI):**
1. **All learned models were optimized for circular labels.** On the old circular H36M data the network could simply learn to invert the DLT triangulation; on true GT it must learn real geometry, which the current recipe is not regularized enough to do.
2. **Too few training steps per epoch.** `train_samples=1024` with `batch_size=16` gives only 64 gradient steps per epoch for a 2.73 M-parameter model, causing rapid memorisation of the small epoch.
3. **Learning rate too aggressive / warmup too short.** `lr=1e-3` with 1-epoch warmup may overfit quickly on the smaller true-GT training set.
4. **Missing regularization.** No weight decay, no early stopping, and aggressive outlier/variable-view augmentation on a small dataset all amplify overfitting.
5. **Auxiliary loss imbalance.** `v25_geom_loss_weight=0.1`, `reproj_loss_weight=0.1`, `aleatoric_reproj_loss_weight=0.1`, and `pa_loss_weight=0.5` may dominate the true pose loss and push the model into degenerate minima once the easy reprojection targets are fit.

**Concrete steps:**
1. Wait for the running v57 medium to finish; do not start another GPU task.
2. Confirm GPU is free (`nvidia-smi` shows no training python process).
3. Run 2–3 controlled smoke ablations (1–2 epochs each, ≤1 h total) on the best-performing variant (likely v80 unless v57 beats it) to isolate the biggest driver:
   - Increase `train_samples` to 4096 (or 8192).
   - Add weight decay (`1e-4`) and early stopping (patience 3).
   - Lower LR / longer warmup.
   - Reduce `outlier_view_prob` and outlier noise std.
4. Once a smoke recipe stops diverging, scale to the full 8-epoch medium schedule.
5. Document findings in `docs/v25_true_gt_failure_mode.md` (or rename to `docs/true_gt_overfitting_diagnosis.md`) and update `docs/results_true_gt_h36m.md`.

**Scripts:**
```bash
# Reproduce the failure (optional, already run)
# bash scripts/run_v25_h36m_true_gt_medium_local_4090.sh

# Example smoke ablations (run only when GPU is free)
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader --mixed_manifest configs/splits/h36m_true_gt_standard.yaml \
    --use_multiview_geometry_fusion_v25 --epochs 2 --batch_size 16 --train_samples 1024 \
    --lr 3e-4 --lr_warmup_epochs 2 --max_grad_norm 1.0 \
    --output outputs/v25_h36m_true_gt_lowlr_smoke.pth \
    > outputs/v25_h36m_true_gt_lowlr_smoke.log 2>&1
```

**Success criteria:**
- A clear failure-mode diagnosis documented for the shared v25/v80/v57 overfitting pattern.
- At least one smoke ablation shows validation MPJPE stable or improving after epoch 2.
- A revised medium recipe (applicable to v25/v80/v57) is defined and queued.

**Blocked by:** GPU state must be confirmed before launching any GPU task (`nvidia-smi` currently shows no active training process, but always re-check).

---

### 2. After v57: pick the best learned baseline and run the mitigated medium (P0)

**Goal:** Use the v57 result to decide which variant (v25/v80/v57) deserves the first full medium re-run with the overfitting fixes.

**Decision tree once v57 finishes:**
- **If v57 < v80 (best < 39.98 mm):**
  - v57 becomes the primary learned baseline.
  - Run the mitigated medium on v57 with the fixes from Task 1 (`train_samples 4096`, `weight_decay 1e-4`, `lr 5e-4`, early stopping, reduced outlier augmentation).
  - If it still trails Iskakov/DLT, ablate the v57-specific DC-PSC modules (`use_v57_domain_conditional_psc`, `v57_dcpsc_*`) to see whether they help or hurt on true GT.
- **If v57 ≥ v80 or v57 fails/diverges earlier:**
  - v80 remains the primary learned baseline.
  - Run the mitigated medium on v80 using the same fixes.
  - Optionally compare v80 against the A800 v2 checkpoint (39.70 mm) to verify reproducibility.
- **Regardless of winner:**
  - Update `docs/results_true_gt_h36m.md` with the new best result.
  - Update `docs/handoff_qwen3.8max.md` to reflect the new leader among learned models.

**Success criteria:**
- A single learned baseline is selected and queued with a concrete, overfitting-mitigated config.
- The selected baseline completes a full medium run without post-best-epoch divergence, or the divergence is materially delayed.
- The best learned result is within 10 mm of confidence-weighted DLT (≤ 35 mm) or a clear explanation is documented why it is not.

**Blocked by:** Task 1 smoke ablations and the currently running v57 medium.

---

### 3. Re-run learned models on non-circular Shelf/Campus with longer/mixed training (P1)

**Goal:** Determine whether MotionFlow variants can close the gap to Iskakov (128.73 mm) on real detected-2D labels, either through longer training or mixed-dataset pretraining.

**Current state:**
- On `configs/splits/shelf_campus_detected_smoke.yaml`:
  - Iskakov: 128.73 mm
  - v80 long (25 epochs): 276.49 mm
  - v57 long (25 epochs): 306.45 mm
  - v25 smoke (3 epochs): 430.67 mm
- All learned models are far behind Iskakov.

**Concrete steps:**
1. Confirm GPU is free and the winning H36M true-GT baseline from Task 2 is at least preliminarily stable.
2. Run the winning baseline long on Shelf/Campus detected using the same 25-epoch recipe as v80/v57.
3. Run a cross-dataset transfer experiment: pretrain v25/v80 on H36M true-GT, then fine-tune on Shelf/Campus detected.
4. Update `docs/results_true_gt_shelf_campus.md`.

**Scripts:**
```bash
# Winning baseline long on Shelf/Campus detected
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --config configs/splits/shelf_campus_detected_smoke.yaml \
    --epochs 25 --batch_size 8 --train_samples 512 --val_stride 2 \
    --log_path outputs/<winner>_shelf_campus_detected_long.log

# H36M true-GT -> Shelf/Campus transfer
# 1. Train <winner> on H36M true-GT (Task 2)
# 2. Fine-tune from the best checkpoint on shelf_campus_detected_smoke.yaml
```

**Success criteria:**
- Winning baseline long result ≤ 250 mm on Shelf/Campus detected (beating v80/v57 long runs).
- Transfer pretraining result ≤ 200 mm, or a clear signal that cross-dataset pretraining helps/hurts.

**Blocked by:** Task 2 (best baseline selected and mitigated) should produce a stable H36M true-GT recipe before investing 25 epochs on Shelf/Campus.

---

### 4. Establish the MPI-INF-3DHP real-detected-2D pipeline

**Goal:** Replace the GT-projected-2D + noise fallback with real detector outputs so MPI results are standard-protocol compliant.

**Current state:**
- `univ_annot3` is true 3D; DLT baseline on true GT is ~23.8 mm.
- `imageSequence/` is missing locally and on A800-D.
- Fallback is GT 2D + 2 px noise + fixed confidence 0.81, which is **not acceptable** for standard protocol.

**Concrete steps:**
1. Locate or request the MPI-INF-3DHP `imageSequence/` archives. Expected layout:
   ```
   data/webbridge/mpi_inf_3dhp/raw/S1/Seq1/imageSequence/
   data/webbridge/mpi_inf_3dhp/raw/S1/Seq2/imageSequence/
   ...
   ```
2. Run real 2D detection with the existing wrapper:
   ```bash
   python scripts/generate_mpi_detected_2d.py --detector auto --split trainval
   ```
3. Generate non-circular `.npz` from detected 2D + `univ_annot3`.
4. Run DLT/v25/v80 baselines on the standard MPI protocol (train S1/S3, test S2/Seq1).
5. Create `docs/results_true_gt_mpi.md` and update `docs/roadmap_cvpr2027.md` dataset table.

**Success criteria:**
- Real detected-2D `.npz` exist and pass `scripts/diagnose_circular_labels.py` (direct MJE ≫ 0).
- DLT baseline on real detected 2D is recorded and comparable to literature (~20–30 mm).
- v25/v80 smoke numbers recorded.

**Blocked by:** External data acquisition (MPI `imageSequence/` not yet present).

---

## Ordering and concurrency

1. **Task 1** is highest priority because v25/v80 diverged on the now-fixed H36M protocol. We must understand and fix the recipe before investing GPU time on longer runs.
2. **Task 2** runs as soon as the v57 result is available and the GPU is free; it selects the best learned baseline and re-runs it with the mitigated recipe.
3. **Task 3** follows Task 2; it extends the winning baseline to Shelf/Campus and tests the cross-dataset story.
4. **Task 4** (MPI real-detected-2D pipeline) is parallelizable with documentation work until the MPI `imageSequence/` data and a free GPU are available; once a GPU task is needed, it must queue behind Task 2.

---

## Anti-patterns to avoid

- Do not run more than one GPU training/diagnostic task at a time on the RTX 4090.
- Do not write, start, or modify anything on A800-D or its Docker `motionflow` service.
- Do not use circular-label `.npz` (`data/h36m_hf/`, `data/webbridge/h36m*.npz`) for model selection.
- Do not launch a full v80/v57 medium before understanding the shared overfitting pattern on true GT.
- Do not duplicate agent-67's AIST++ work.

---

## Exit criteria for this roadmap

- [ ] Task 1: Shared v25/v80/v57 true-GT failure mode diagnosed and a revised training recipe is found (validation MPJPE no longer diverges after the best epoch).
- [ ] Task 2: Best learned baseline (v25/v80/v57) selected and re-run with the mitigated recipe; H36M true-GT leaderboard updated.
- [ ] Task 3: Shelf/Campus detected results improved or transfer experiments documented.
- [ ] Task 4: MPI-INF-3DHP real-detected-2D pipeline created and baselines run.

After these tasks, the project will have an honest, cross-dataset evaluation foundation (H36M + Shelf/Campus + MPI) and a stable learned baseline to support the sparse-view / cross-domain robustness paper story.
