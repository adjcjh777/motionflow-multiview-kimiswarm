# Agent-02 ANALYZE — A800 v25 all-train baseline status

**Agent:** Agent-02  
**Task:** SSH read-only check of A800 `v25 all-train baseline` tmux/log; estimate first val time.  
**Tracking issue:** #160 (depends on #154)  
**Branch:** `v46-svg`  
**Snapshot time (A800):** 2026-08-09 04:52 UTC (SSH command `date`)  
**Run start time:** 2026-08-09 04:32:15 UTC (tmux session creation)  

## Findings

### Session / process

| Item | Value |
|------|-------|
| tmux session | `v25_geometry_fusion_all_train_baseline_gpu6` |
| GPU | 6 (A800-SXM4-80GB) |
| PID | 236570 (main), children 236570, 237030-237038 (DataLoader workers / spawned copies) |
| Command | `python3 -u experiments/train_omniview_fusion_v5_webbridge_multi.py ... --d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 200 --epochs 5 ...` |
| GPU util | 100 % |
| GPU memory | 79 779 MiB / 81 920 MiB used (~97 %) |
| Temperature | 37 °C |

Notes:
- The command line contains two argument blocks; the second (later) block wins, giving `d=128`, `batch_size=16`, `clip_len=13`, `train_samples=200`, `epochs=5`. This matches the saved `.config.json`.
- `tmux capture-pane` returned no visible content on repeated attempts, so the analysis below relies on the log file and `nvidia-smi`.

### Log progress

File: `outputs/omniview_fusion_v25_geometry_fusion_all_train_baseline_a800.log`

```text
Device: cuda
/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/motionflow_mv/fusion/hierarchical_multiview_v30.py:77: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
Model params: 3311923
  train step 50: loss=23.571892
  train step 100: loss=21.639591
  train step 150: loss=19.912279
```

- **Current step:** 150 (as of log update ~04:51 UTC)
- **Loss trend:** decreasing steadily (23.57 → 21.64 → 19.91)
- **No errors or NaNs observed.**

### First-validation estimate

The manifest `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml` has **31 train sequences**. With `train_samples=200` per sequence and `batch_size=16`:

```
total_train_clips = 31 × 200 = 6 200
steps_per_epoch   = ceil(6 200 / 16) ≈ 388
```

At the observed rate:

- Run started: 04:32:15
- Step 150 reached at ~04:51:00
- Elapsed for 150 steps  18 min 45 s
- Observed step rate ≈ 7.5 s/step

Remaining in epoch 1:

```
remaining_steps = 388 - 150 = 238
remaining_time  ≈ 238 × 7.5 s ≈ 1 785 s ≈ 30 min
```

Adding validation overhead (16 val sequences, `val_stride=10`, d=128 model, ~79 GB resident) gives an additional **5–15 min**.

**Estimated first val completion: ~05:25–05:50 UTC on 2026-08-09.**

Validation runs once per epoch (`TrainerV2.fit`), so this will be the first `val_MPJPE` reported for the A800 v25 all-train baseline.

## Relevance to v46-SVG

- The v25 all-train baseline is the intended starting point for v46 sparse-view generalization (#160). It already enables `use_variable_view_training` (min 2, max 14 views, curriculum), which overlaps with the v46 view-dropout goal.
- A stable baseline val (target ~17 mm from historical v25 full A800 runs) is needed before v46-SVG smoke/full runs are queued.

## Blockers / questions

- None for this read-only check. The run is healthy and GPU-bound.
- Minor note: `tmux capture-pane` returned empty output; logs are the reliable read-only source.

## References

- A800-D remote host: `a800-D`
- Remote repo: `/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20`
- Log: `outputs/omniview_fusion_v25_geometry_fusion_all_train_baseline_a800.log`
- Config: `outputs/omniview_fusion_v25_geometry_fusion_all_train_baseline_a800.config.json`
- Related: docs/swarm_iter23_action_plan.md, docs/proposals/v46_sparse_view_generalization.md, AGENTS.md
