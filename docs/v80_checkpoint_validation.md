# v80 H36M true-GT medium checkpoint validation

> Validation date: 2026-08-11  
> Validator: coder subagent  
> GPU status at check: **busy** (RTX 4090, two `python.exe` compute processes, ~79% util, 14.1 GB VRAM). No GPU eval was launched.

## Artifacts inspected

| File | Path | Notes |
|---|---|---|
| Run script | `scripts/run_v80_h36m_true_gt_medium.sh` | Local RTX 4090 medium run (8 epochs, 1024 samples/epoch, batch 16). |
| Saved config | `outputs/omniview_fusion_v80_h36m_true_gt_medium.config.json` | Written by the training script at the end of the run. |
| Training log | `outputs/omniview_fusion_v80_h36m_true_gt_medium.log` | Console copy of the run. |
| Best checkpoint | `outputs/omniview_fusion_v80_h36m_true_gt_medium.pth` | Saved when validation MPJPE was lowest (epoch 4). |
| Final checkpoint | `outputs/omniview_fusion_v80_h36m_true_gt_medium_final.pth` | State at the end of epoch 8. |
| Data split | `configs/splits/h36m_true_gt_standard.yaml` | Standard protocol: S1,5,6,7,8 train → S9/S11 val. |

## Config vs. run-script cross-check

All 91 command-line flags in `scripts/run_v80_h36m_true_gt_medium.sh` were parsed and compared against the saved `config.json`. **No discrepancies were found.**

Key v80 architecture flags are all enabled and match the intended recipe:

- `use_v80_view_reliability: true`
- `v80_vrbt_hidden: 64`, `v80_vrbt_n_layers: 2`, `v80_vrbt_weight_type: "per_view_joint"`
- `v80_vrbt_identity_init: true`, `v80_vrbt_min_weight: 0.05`
- Companion modules (v45 adaptive geometry, v46 sparse-view generalisation, v50 SEFH, v51 cross-domain reliability, v52 uncertainty-weighted triangulation) are all active with the values set in the script.

Training hyperparameters also match:

- `epochs: 8`, `batch_size: 16`, `train_samples: 1024`, `val_stride: 20`
- `d: 64`, `residual_hidden: 128`, `n_st_layers: 2`
- `lr: 0.001`, `lr_cosine: true`, `lr_warmup_epochs: 1`, `lr_min: 1e-6`
- `output: "outputs/omniview_fusion_v80_h36m_true_gt_medium.pth"`

## Data split verification

The manifest `configs/splits/h36m_true_gt_standard.yaml` points to the non-circular true-GT files in `data/h36m_true_gt/`. All referenced `.npz` files exist and are non-empty:

- Train: S1, S5, S6, S7, S8 (`*_multiview_m.npz`)
- Val: S9, S11 (`*_multiview_m.npz`)

## Checkpoint integrity (CPU inspection)

Both checkpoints were loaded on CPU with `torch.load(..., map_location="cpu", weights_only=False)`.

| Property | Best checkpoint | Final checkpoint |
|---|---|---|
| Path | `outputs/omniview_fusion_v80_h36m_true_gt_medium.pth` | `outputs/omniview_fusion_v80_h36m_true_gt_medium_final.pth` |
| File size | 12.8 MB | 12.8 MB |
| Stored epoch | 4 | 8 |
| Top-level keys | `epoch`, `model`, `optimizer`, `amp`, `history`, `scheduler`, `ema` | same |
| Model state-dict keys | 321 | 321 |
| NaN / Inf params | none | none |
| v80 head keys | `view_reliability_head_v80.*` present | `view_reliability_head_v80.*` present |

The best checkpoint keeps the model state from the epoch with the lowest validation MPJPE. The final checkpoint keeps the end-of-training state. Both are valid PyTorch checkpoints with intact optimizer, scheduler, AMP, and EMA buffers.

## Training summary (from log)

| Epoch | Val loss | Val MPJPE (mm) |
|---|---:|---:|
| 1 | 0.002875 | 88.78 |
| 2 | 0.001722 | 66.26 |
| 3 | 0.000933 | 44.41 |
| **4** | **0.000851** | **39.98** |
| 5 | 0.001649 | 56.68 |
| 6 | 0.003270 | 83.30 |
| 7 | 0.005446 | 110.36 |
| 8 | 0.007782 | 133.71 |

- **Best validation MPJPE:** 39.98 mm at epoch 4.
- **Final validation MPJPE:** 133.71 mm at epoch 8.
- **Pattern:** monotonic improvement through epoch 4, then overfitting. This matches the AGENTS.md handoff note and the recipe-sweep findings in `docs/results_v80_h36m_true_gt.md`.
- No NaN/inf or training crashes were observed.

## Conclusion

The v80 H36M true-GT medium checkpoint and its saved configuration are **correct and internally consistent**:

1. The saved `config.json` exactly matches the launch script.
2. The data split is the intended non-circular H36M true-GT protocol.
3. Both the best and final checkpoints load cleanly, contain the expected v80 model weights, and show no parameter corruption.
4. The training log records a best val MPJPE of 39.98 mm at epoch 4, followed by the same overfitting pattern seen in earlier v80 recipe sweeps.

Because the local RTX 4090 was busy, no additional GPU evaluation or re-validation inference was run. If a full forward/eval sanity check is needed, it should be queued for when the GPU is idle.
