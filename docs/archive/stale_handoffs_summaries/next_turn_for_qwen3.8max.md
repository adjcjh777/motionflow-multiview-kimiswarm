# Next Turn for qwen3.8max — 2026-08-11 14:02 UTC

> **Created**: 2026-08-11 13:33 UTC  
> **Updated**: 2026-08-11 14:02 UTC  
> **GPU status**: RTX 4090 BUSY running v25 ablation smoke (`v25_true_gt_baseline_fix`) and `eval_variable_views`. v57 H36M true-GT medium is DONE.  
> **Do NOT start any GPU training/eval until the current process finishes.**

---

## Top 3 Priorities

### 1. Monitor v25 ablation smoke results and decide v80 medium v2 recipe

**Goal**: Capture the final numbers from the two v25 ablation smokes, verify whether the regularised recipe stops val MPJPE from rising, and prepare a v80 medium v2 launch script.

**Current status (as of 2026-08-11 14:02 UTC)**:

- v57 is **DONE**: best observed val MPJPE **75.16 mm** @ epoch 3, but saved best checkpoint is **81.47 mm** (epoch 2); early-stopped @ epoch 5 (final 80.21 mm).
- v25 ablation `v25_true_gt_baseline_fix` is **RUNNING** on GPU.
- v25 ablation `v25_true_gt_geometry_regularization` is **QUEUED / RUNNING**.
- `eval_variable_views` for v80 is also running on GPU.

**Exact commands**:

```bash
# Check GPU status
nvidia-smi

# Tail the ablation logs
tail -f outputs/ablations/v25_true_gt_baseline_fix.log
tail -f outputs/ablations/v25_true_gt_geometry_regularization.log

# Extract per-epoch val MPJPE once logs have content
python - <<'PY'
import glob, re
for log in sorted(glob.glob('outputs/ablations/v25_true_gt_*.log')):
    print(f"\n=== {log} ===")
    with open(log) as f:
        for line in f:
            if re.search(r'val\s*MPJPE|epoch\s*\d+.*MPJPE', line, re.I):
                print(line.rstrip())
PY
```

**What to record**:

- Best epoch / val MPJPE for each ablation.
- Whether val MPJPE stops rising after the best epoch (the key success metric).
- Whether geometry regularization improves over the baseline fix.

**Decision after ablations**:

- If the regularised recipe stabilises val MPJPE: prepare `scripts/run_v80_h36m_true_gt_medium_v2_local_4090.sh` and run it once GPU is free.
- If it still diverges: try lower lr (3e-4), higher weight decay (5e-4), or mixed-dataset smoke before committing to a full medium.

---

### 2. Record v57 H36M true-GT final result

**Goal**: v57 is already captured in `docs/results_true_gt_h36m.md`; verify the row is accurate.

**Exact commands**:

```bash
tail -n 20 outputs/omniview_fusion_v57_h36m_true_gt_medium.log
grep -A5 -B2 'v57' docs/results_true_gt_h36m.md
```

**What to verify**:

- Best epoch / combined direct MPJPE / PA-MPJPE.
- S9 and S11 per-split direct MPJPE if available.
- Compare v57 vs v25 (72.80 mm), v80 (39.98 mm), DLT (25.87 mm), Iskakov (23.35 mm).
- **Conclusion**: v57 (75.16 mm observed / 81.47 mm ckpt) is worse than v80 (39.98 mm); v80 remains the primary learned baseline.

---

### 3. Diagnose v25/v80/v57 overfitting on H36M true-GT and run smoke ablations

**Goal**: Understand why MotionFlow variants overfit/diverge on true GT, then validate fixes with short smoke runs.

**Pre-read**:

- `docs/v25_divergence_diagnosis.md`
- `docs/v25_ablation_plan.md`
- `configs/ablations/`

**Exact commands**:

```bash
# Ablations are already running; monitor them rather than starting duplicates.
# Confirm GPU is still busy and which processes are active
nvidia-smi
wmic process where "name='python.exe'" get ProcessId,CommandLine

# Tail the live ablation logs
tail -f outputs/ablations/v25_true_gt_baseline_fix.log
tail -f outputs/ablations/v25_true_gt_geometry_regularization.log

# Extract val MPJPE per epoch for comparison once logs have content
python - <<'PY'
import glob, re
for log in sorted(glob.glob('outputs/ablations/v25_true_gt_*.log')):
    print(f"\n=== {log} ===")
    try:
        with open(log) as f:
            for line in f:
                if re.search(r'val\s*MPJPE|epoch\s*\d+.*MPJPE', line, re.I):
                    print(line.rstrip())
    except FileNotFoundError:
        print(f"Log not found: {log}")
PY
```

**Expected outcome**: A new doc `docs/true_gt_overfitting_diagnosis.md` (or update `docs/v25_true_gt_failure_mode.md`) with:

- v25, v80, v57 epoch curves plotted side-by-side.
- Which ablation stops val MPJPE from rising after the best epoch.
- Recommended next medium-run recipe.

---

### 3. Fix MPI detected-2D alignment or run AIST++ full-medium

**Goal**: Make MPI detected-2D usable for benchmarking (DLT ~20–30 mm) OR get a full AIST++ medium run for learned models.

#### Option A — MPI (P0 blocker; data ready, alignment blocked)

```bash
# Check the generated MPI detected-2D .npz files
ls data/webbridge/mpi_inf_3dhp_detected_2d/*_m.npz 2>/dev/null | wc -l

# Re-run DLT baseline to confirm alignment status
python scripts/run_mpi_dlt_baseline.py \
    --glob "data/webbridge/mpi_inf_3dhp_detected_2d/*_m.npz" \
    --output outputs/mpi_dlt_baseline_detected_2d_full.json \
    --device cpu

# If the mean MPJPE is not ~20–30 mm, investigate camera/label alignment
# before benchmarking learned models on MPI.
```

#### Option B — AIST++ full medium (fallback if MPI is blocked)

**Exact commands**:

```bash
# Only run after v57 and ablations are done and GPU is idle
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --config configs/splits/aistpp_train_val.yaml \
    --log_path outputs/omniview_fusion_v80_aistpp_full_medium.log
```

Pick whichever is unblocked first.

---

## What to Avoid

- **Do NOT start any new GPU training/eval job while ablations/eval are running.** GPU concurrency on a single RTX 4090 will OOM or slow the current run.
- **Do NOT duplicate the v25 ablation smoke** — `v25_true_gt_baseline_fix` and `v25_true_gt_geometry_regularization` are already running. Monitor, do not restart.
- **Do NOT write, start, or modify anything on A800-D / Docker.** The A800 and its `motionflow` Docker service are read-only.
- **Do NOT delete or overwrite v57 checkpoint/log** until results are recorded and verified.
- **Do NOT duplicate the MPI real-detected 2D generation** if it is already running; check `outputs/generate_mpi_detected_2d_from_avi.log` first.  
  *Current observation*: there appear to be **two** `generate_mpi_detected_2d_from_avi.py` processes running (one with `--start_frame 500 --end_frame 1000 --only_m --cpu_only`, one without). Verify whether this is intentional or a duplicate before starting another.
- **Do NOT mix ablation smoke configs** (run them one at a time and log separately).

---

## Quick Status Check

```bash
# GPU / python processes
nvidia-smi
tasklist //FI "imagename eq python.exe" //FO TABLE
wmic process where "name='python.exe'" get ProcessId,CommandLine

# v57 final result
tail -n 20 outputs/omniview_fusion_v57_h36m_true_gt_medium.log

# Ablation smoke status
tail -n 30 outputs/ablations/v25_true_gt_baseline_fix.log
tail -n 30 outputs/ablations/v25_true_gt_geometry_regularization.log

# MPI generation status
tail -n 30 outputs/generate_mpi_detected_2d_from_avi.log

# Count running MPI generation processes
tasklist //FI "imagename eq python.exe" //FO CSV | grep -c generate_mpi_detected_2d_from_avi
```
