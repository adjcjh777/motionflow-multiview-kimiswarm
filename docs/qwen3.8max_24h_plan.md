# qwen3.8max 24-Hour Plan (2026-08-11 → 2026-08-12)

**Owner:** qwen3.8max  
**Workspace:** `D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`  
**Status at plan creation:** Local RTX 4090 is **BUSY** running two GPU processes:
- `experiments/eval_variable_views.py` (PID 55087) — v25 variable-views evaluation
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` (PID 55090) — v57 H36M true-GT medium training

**Constraints**
- A800-D `/mnt/nvme0n1/zhangzy/projects` and the `motionflow` Docker are **read-only** — no writes, starts, or modifications.
- Do **not** start a second GPU training job while the RTX 4090 is occupied.
- Prefer documentation, analysis, and script preparation until GPU frees up.

---

## 0. Immediate First Action (< 15 min)

Confirm the currently running GPU jobs and estimate when they will finish.

```bash
nvidia-smi
ps -ef | grep -E "train_omniview|eval_variable_views" | grep -v grep
tail -n 50 outputs/omniview_fusion_v57_h36m_true_gt_medium.log
tail -n 50 outputs/eval_variable_views_h36m_true_gt/v25_results.csv 2>/dev/null || true
```

**Expected outcome:** A note in this plan (Section 7) with the actual start time, estimated finish time, and whether the v57 run has produced a usable checkpoint.

---

## 1. CPU-Only Block (0 h – 6 h): Documentation, Analysis, Script Prep

These tasks do not touch the GPU and can run in parallel with the active v57/eval jobs.

### 1.1 Consolidate the v25/v80/v57 true-GT failure mode note

**Task:** Create or update `docs/true_gt_overfitting_diagnosis.md` that compares the three runs on the same axes:
- Best epoch, best val MPJPE, final val MPJPE
- Train-samples/epoch, batch size, learning rate, weight decay, early-stopping patience
- Whether the saved checkpoint is the best-epoch or final-epoch file

**Inputs:**
- `docs/v25_divergence_diagnosis.md`
- `docs/results_true_gt_h36m.md`
- `outputs/omniview_fusion_v25_h36m_true_gt_medium.log`
- `outputs/omniview_fusion_v80_h36m_true_gt_medium*.log`
- `outputs/omniview_fusion_v57_h36m_true_gt_medium.log` (once available)

**Expected outcome:** A single table and 3–4 bullet conclusions that explain *why* all learned models diverge and what the common fix is.

**Scheduling:** CPU-only; 1–2 h.

### 1.2 Prepare the v25/v80/v57 regularised re-run scripts

**Task:** Create three new shell scripts that apply the fixes from `docs/v25_divergence_diagnosis.md`. Do not execute yet.

- `scripts/run_v25_h36m_true_gt_medium_v2_local_4090.sh`
- `scripts/run_v80_h36m_true_gt_medium_v2_local_4090.sh`
- `scripts/run_v57_h36m_true_gt_medium_v2_local_4090.sh`

**Required hyperparameter changes (compared with the failed runs):**
- `--train_samples 4096` (or 8192 if GPU memory/schedule allows)
- `--weight_decay 1e-4`
- `--early_stopping_patience 3 --early_stopping_min_delta 0.001`
- `--lr 5e-4 --lr_warmup_epochs 2`
- `--outlier_view_prob 0.15`
- For v25/v57: `--v25_geom_loss_weight 0.05` (reduced from 0.1)

**Expected outcome:** Three ready-to-run scripts committed under `scripts/`, plus a short `docs/v25_v80_v57_regularised_runs.md` describing the rationale and the exact diff from the old scripts.

**Scheduling:** CPU-only; 1–2 h.

### 1.3 Audit and fix the saved-config inconsistency

**Task:** In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, find where `manifest` is written to `config.json` and make it reflect the actually used `--mixed_manifest` when `--use_mixed_loader` is set.

**Why:** The current saved config still lists `configs/splits/webbridge_all_train.yaml` even when the run used `configs/splits/h36m_true_gt_standard.yaml`. This is a reproducibility bug.

**Expected outcome:** A minimal patch plus a test that loads the latest v57 `config.json` and asserts `manifest == mixed_manifest` when `--use_mixed_loader` is active.

**Scheduling:** CPU-only; 30–60 min.

### 1.4 Generate / validate the missing MPI-INF-3DHP S2/Seq2 canonical `.npz`

**Task:** Investigate why `data/webbridge/mpi_inf_3dhp_detected_2d/s_02_seq_02_v14_multiview_m.npz` is missing and generate it if the raw canonical data exists.

```bash
# Check what is present
ls -la data/webbridge/mpi_inf_3dhp/raw/S2/Seq2/imageSequence/ 2>/dev/null | head
ls -la data/webbridge/mpi_inf_3dhp/*s_02_seq_02* 2>/dev/null || true
ls -la data/webbridge/mpi_inf_3dhp_detected_2d/*s_02* 2>/dev/null || true

# If canonical exists but detected 2D is missing, re-run only this sequence
python scripts/generate_mpi_detected_2d.py \
    --raw_dir data/webbridge/mpi_inf_3dhp/raw \
    --output_dir data/webbridge/mpi_inf_3dhp_detected_2d \
    --subjects 2 --seqs 2 --detector mediapipe
```

**Expected outcome:** Either the missing file is produced, or a documented blocker (e.g., raw video frames missing) is added to `docs/mpi_3dhp_blockers.md`.

**Scheduling:** CPU / light GPU (MediaPipe detection); can run in background on CPU if frames exist. Do not run if it conflicts with the active GPU training.

### 1.5 Prepare AIST++ medium run script

**Task:** Create `scripts/run_v25_aist_medium_local_4090.sh` and `scripts/run_v80_aist_medium_local_4090.sh` based on the existing smoke script, but with `--train_samples 4096`, 10 epochs, and the same regularisation as above.

**Expected outcome:** Two ready-to-run scripts and a note in `docs/qwen3.8max_24h_plan.md` (this file) on when they should be launched.

**Scheduling:** CPU-only; 30 min.

---

## 2. GPU Window #1 (6 h – 12 h): Launch First Regularised Re-run

**Pre-condition:** The currently running v57 medium (PID 55090) and v25 eval (PID 55087) have finished or been safely stopped. Re-check `nvidia-smi` and confirm no `python.exe` Compute process is using the GPU.

### 2.1 Launch v80 regularised re-run

**Task:** Run `scripts/run_v80_h36m_true_gt_medium_v2_local_4090.sh`.

**Why first:** v80 has the best converged number so far (39.98 mm). Validating the regularisation recipe on the strongest variant first gives the fastest signal.

```bash
nvidia-smi  # MUST show no active python training process
bash scripts/run_v80_h36m_true_gt_medium_v2_local_4090.sh
```

**Expected outcome:** A log file `outputs/omniview_fusion_v80_h36m_true_gt_medium_v2.log` with a val MPJPE curve that no longer explodes after epoch 2–4.

**GPU budget:** ~4–6 h for 8 epochs at 4096 samples/epoch on RTX 4090.

---

## 3. CPU/GPU Parallel Block (12 h – 18 h)

### 3.1 Monitor v80 run and produce live curve plots

**Task:** While v80 is training, parse its log every 30 min and update `docs/v80_h36m_true_gt_v2_live_curve.md` with:
- Epoch, train loss, val loss, val MPJPE, PA-MPJPE, EMA shadow val MPJPE (if logged)
- A short text interpretation

```bash
tail -n 100 outputs/omniview_fusion_v80_h36m_true_gt_medium_v2.log
```

**Expected outcome:** A living document that lets the next shift pick up without re-reading the full log.

**Scheduling:** CPU-only; 10 min every 30 min.

### 3.2 CPU-only paper story rewrite (Part 1)

**Task:** Update `docs/paper_draft_icra_cvpr_2027.md` Section 3 (Results) and Section 1 (Contributions) to reflect:
- The corrected true-GT numbers (Iskakov 23.35 mm, DLT 25.87 mm, v25/v80 still behind)
- The new pitch: sparse-view / cross-domain robustness, not absolute MPJPE dominance

**Expected outcome:** A rewritten 500-word section with a TODO marker for final numbers after the regularised runs finish.

**Scheduling:** CPU-only; 1–2 h.

### 3.3 CPU-only cross-dataset manifest design

**Task:** Design the manifest for a mixed H36M + AIST++ + Shelf/Campus (and optionally MPI detected-2D) training run.

- File: `configs/splits/cross_domain_true_gt_medium.yaml`
- Requirements:
  - Train: H36M true-GT S1,5,6,7,8 + AIST++ train + Shelf/Campus train
  - Val: H36M S9/S11 + AIST++ val + Shelf/Campus val
  - Domain labels: 0=H36M, 1=AIST++, 2=Shelf/Campus, 3=MPI (if available)

**Expected outcome:** A valid YAML manifest plus a short validation script that checks all referenced `.npz` files exist.

**Scheduling:** CPU-only; 1 h.

---

## 4. GPU Window #2 (18 h – 24 h): Launch Second Regularised Re-run or AIST++ Medium

**Pre-condition:** v80 v2 run has finished or is clearly converging.

### 4.1 Decide the second GPU job

Decision tree:

1. If v80 v2 improves over 39.98 mm and is still converging:
   - Launch v57 v2 regularised re-run (`scripts/run_v57_h36m_true_gt_medium_v2_local_4090.sh`).
2. If v80 v2 does not improve:
   - Launch v25 v2 regularised re-run to test whether the same regularisation fixes v25.
3. If both v25/v80 are running behind and AIST++ data is fully ready:
   - Launch AIST++ medium for v25 or v80.

**Task:** Run the chosen script with a `nohup` background command, capture the task ID, and report it.

```bash
nvidia-smi  # confirm free
nohup bash scripts/run_<chosen>_h36m_true_gt_medium_v2_local_4090.sh \
    > outputs/<chosen>_v2_nohup.log 2>&1 &
echo $!  # record this PID
```

**Expected outcome:** A second GPU run is underway before the 24-hour window closes; a note is added to `docs/handoff_qwen3.8max.md` describing which run is active and when it is expected to finish.

---

## 5. Mandatory Checkpoints and Handoffs

At the end of the 24-hour window, produce an updated `docs/handoff_qwen3.8max.md` with:

- Exact GPU status at handoff (`nvidia-smi` output)
- List of files changed / scripts created
- Which GPU runs are active, their PIDs, and expected completion time
- The v57 medium result (if finished)
- Whether the v80/v25/v57 regularised scripts were launched
- Any blockers discovered (e.g., missing MPI S2/Seq2 frames)

---

## 6. Risk and Contingency

| Risk | Mitigation |
|------|-----------|
| v57 run still occupies GPU at 6 h | Continue CPU-only tasks; delay GPU Window #1 until it finishes. |
| v80 v2 still overfits | Document it, then try mixed-dataset training or progressive unfreezing (Section 3 of `docs/v25_divergence_diagnosis.md`). |
| MPI detected-2D alignment | Fix camera/label coordinate-frame mismatch; skip MPI from cross-domain manifest until DLT baseline ~20–30 mm. |
| RTX 4090 instability / OOM | Reduce `--batch_size` to 8 or `--train_samples` to 2048 in the prepared scripts. |

---

## 7. GPU Status Log (updated by qwen3.8max)

| Time (UTC) | GPU Util | Mem Used | Processes | Actionable |
|---|---:|---:|---|---|
| 2026-08-11 13:26 | 41–98 % | 14084 MiB | v57 train (PID 55090), v25 eval (PID 55087) | Do not launch new GPU work. |

**Next GPU-status check:** within the first hour of this plan.
