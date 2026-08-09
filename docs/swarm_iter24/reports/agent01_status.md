# Agent-01 Status Report: v45-AGF Medium & v46-SVG Smoke

**Scope:** Read `outputs/v45_agf_medium_local_4090.log` and the v46 smoke log, summarize current state, and predict when smoke and full A800 results will land.  
**Branch:** `v47-temporal`  
**Report date:** 2026-08-09 ~13:05 CST  
**Tracking issue:** #162

---

## 1. Executive Summary

- **v45-AGF medium (local RTX 4090)** is still running, in epoch 3 of 5.  Epoch 1 gave a promising `val_MPJPE = 31.95 mm`, but epoch 2 regressed sharply to `96.87 mm`, indicating severe overfitting.
- **v46-SVG smoke (local RTX 4090)** started around 13:03 and is in epoch 2.  Loss is dropping quickly, but the run is overlapping with v45-AGF on the same GPU despite the script warning not to overlap.
- **v47-temporal smoke** remains blocked until the v46-SVG smoke finishes and the v47 module/config are ready.
- **A800 full runs** for v46/v47 are not yet scheduled; the A800 cluster is heavily loaded with v25/v31–v34 runs.

---

## 2. v45-AGF Medium Local Run

| Item | Value |
|------|-------|
| Log file | `outputs/v45_agf_medium_local_4090.log` |
| Script | `scripts/run_v45_agf_medium_local_4090.sh` |
| Epochs | 5 |
| `train_samples` | 500 |
| `batch_size` | 4 |
| `d` | 64 |
| Current progress | Epoch 3, step ~400 |
| Epoch 1 val | `val_loss=0.029085`, `val_MPJPE=31.95 mm` |
| Epoch 2 val | `val_loss=0.034811`, `val_MPJPE=96.87 mm` |
| Epoch 3 train loss | ~7.0, flat/no improvement |

### Interpretation

- The epoch-2 validation jump from 31.95 mm to 96.87 mm is a clear overfit/instability signal.
- Training loss in epoch 3 is stuck around 7.0, confirming the model is no longer learning useful signal.
- With `early_stopping_patience=3`, the run may continue until epoch 5 unless validation improves.

### Estimated finish time

Assuming ~3 600 steps/epoch and the current observed rate of ~1–2 steps/sec:

- **Remaining steps:** ~3 200 in epoch 3 + ~7 200 for epochs 4–5 ≈ 10 400 steps.
- **ETA:** **2–6 hours**, depending on whether v46 keeps competing for GPU.
- If the run is killed manually due to the epoch-2 regression, it could finish sooner.

---

## 3. v46-SVG Smoke Local Run

| Item | Value |
|------|-------|
| Log file | `outputs/v46_svg_smoke_local_4090.log` |
| Script | `scripts/run_v46_svg_smoke_local_4090.sh` |
| Config | `configs/benchmark_v46_svg_smoke.yaml` |
| Epochs | 2 |
| `train_samples` | 500 |
| `batch_size` | 4 |
| `d` | 64 |
| Current progress | Epoch 2, step ~150 |
| Model params | 897 977 |
| Recent loss | step 50: 20.89 → step 100: 17.13 → step 150: 14.11 |

### Interpretation

- Loss is decreasing smoothly, which is a healthy smoke signal.
- The run was supposed to start **after** v45-AGF medium finished (`scripts/wait_then_run_v46_svg_smoke_local_4090.sh`), but it is running concurrently.
- Assuming the same ~3 600 steps/epoch as v45, there are still ~6 900 steps left.

### Estimated finish time

- **ETA:** **1–2 hours** if it were the only GPU user; **2–4 hours** while v45-AGF is also running.
- Validation results (`val_MPJPE`) will be available only after the 2nd epoch ends.

---

## 4. v47-Temporal Smoke Readiness

- **Status:** Not started.
- **Blocker:** Waiting on v46-SVG smoke to complete so we have a baseline `MPJPE@k` to compare against.
- The design proposal (`docs/proposals/v47_combined_architecture.md`) and the action plan (`docs/swarm_iter24_action_plan.md`) both state that v47 builds on v46 and should not start before v46 smoke lands.
- Implementation files (`motionflow_mv/fusion/temporal_aggregation_v47.py` and `tests/test_temporal_aggregation_v47.py`) are already present in the working tree, but the v47 smoke config and run script are not yet created.

---

## 5. A800 Full-Run Outlook

Read-only A800 check (`ssh a800-D "tmux ls"`):

- No `v46_*` or `v47_*` tmux sessions are active.
- The cluster is heavily loaded with `v25`, `v31`, `v32`, `v33`, and `v34` sessions across GPUs 4–7.
- Queue entries for v45/v46 exist in `scripts/launch_v33_a800_queue.py`, but v46/v47 full runs are not yet scheduled.

Consequences:

- Even if the local v46 smoke passes today, the A800 full run will wait for an available GPU.
- Given current A800 saturation, **v46 full A800 results are likely 1–3 days away** after the smoke passes.
- **v47 full A800 results** will land after v46 full completes and the v47 queue entry is promoted — likely **2–5 days** after v46 full starts.

---

## 6. ETA Summary

| Milestone | Estimated Arrival | Confidence |
|-----------|-------------------|------------|
| v45-AGF medium finishes | 2–6 hours | Medium |
| v46-SVG smoke finishes | 1–2 hours (4+ if GPU contention persists) | Medium |
| v47 smoke can start | After v46 smoke + v47 smoke script ready | High |
| v46 full A800 starts | 1–3 days after smoke passes | Low–Medium |
| v46 full A800 1-epoch result | 1–4 days after it starts | Medium |
| v47 full A800 starts | 2–5 days after v46 full starts | Low |
| v47 full A800 1-epoch result | 1–4 days after it starts | Medium |

---

## 7. Blockers and Risks

1. **GPU overlap:** v45-AGF and v46-SVG are running concurrently on the RTX 4090, contrary to the intended workflow. This slows both and may affect smoke timing.
2. **v45 overfit:** The epoch-2 regression from 31.95 mm to 96.87 mm raises doubts about whether v45-AGF is a stable base for v46/v47. The v46 smoke will tell us if the same overfitting occurs.
3. **A800 saturation:** Many v31–v34 runs are ahead in the queue; v46/v47 full results will not be immediate.
4. **No v47 smoke config/script yet:** Until `configs/benchmark_v47_temporal_svg_smoke.yaml` and `scripts/run_v47_temporal_svg_smoke_local_4090.sh` are created, v47 cannot smoke even after v46 finishes.

---

## 8. Recommendations

- Monitor the v45 log for manual kill if it does not recover by early-epoch-3 validation.
- Let the v46-SVG smoke finish; if `val_MPJPE < 75 mm`, treat it as a green light for v47 implementation.
- Ensure the v47 smoke config/script are created before v46 smoke ends so no GPU time is wasted.
- Do not queue v47 full A800 until v46 full A800 has produced at least one epoch of `MPJPE@k` numbers.
