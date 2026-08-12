# v57 Checkpoint Saving Bug

> **Status: FIXED — v57 rerun ready**  
> The trainer in `motionflow_mv/training/trainer_v2.py` now saves the best checkpoint based on validation MPJPE (`monitor="mpjpe"`), and the validation path passes `view_mask` correctly. A fresh local v57 smoke test completed successfully (best val MPJPE 75.67 mm). The A800 rerun script `scripts/run_v57_true_gt_medium_a800.sh` is ready to launch on a free GPU.

## Observation

`outputs/omniview_fusion_v57_h36m_true_gt_medium.log` reports:

```
Epoch 3: val_MPJPE=75.16mm
Epoch 4: val_MPJPE=76.60mm
Epoch 5: val_MPJPE=80.21mm
Early stopping at epoch 5 (no val_loss improvement for 3 epochs).
Best val MPJPE: 75.16mm -> outputs\omniview_fusion_v57_h36m_true_gt_medium.pth
```

However, the saved checkpoint claims `epoch=2` and its history list only contains epochs 1 and 2:

```python
ckpt = torch.load('outputs/omniview_fusion_v57_h36m_true_gt_medium.pth', map_location='cpu')
ckpt['epoch']  # 2
ckpt['history'][-1]['val']['mpjpe']  # 0.08147186950757326 -> 81.47 mm
```

## Implication

The "best" checkpoint that the training script prints is **not** the true best epoch. The main `.pth` file is from epoch 2 (81.47 mm), not epoch 3 (75.16 mm). Evaluation or downstream variable-view tests using this checkpoint will report ~81 mm rather than the observed best of ~75 mm.

## Likely Root Cause

The trainer likely saves the checkpoint when early stopping patience runs out, but the saved `epoch` / `history` is stale or the model state was last updated at epoch 2. Another possibility is that the EMA update / checkpoint save logic saves at the wrong time.

## Action Items

1. Inspect `motionflow_mv/training/trainer_v2.py` or the saving logic in `experiments/train_omniview_fusion_v5_webbridge_multi.py` to verify when `save_checkpoint` is called and what `epoch` is written.
2. Ensure the checkpoint saved as `.pth` corresponds to the actual best val MPJPE epoch, or save both `_best.pth` and `_final.pth` explicitly.
3. For now, use the numeric result from the log (75.16 mm @ epoch 3) for leaderboard purposes, but note the saved checkpoint is epoch 2.
