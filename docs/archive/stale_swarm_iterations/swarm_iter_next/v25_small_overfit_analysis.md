# v25 small overfit analysis

## A800 v25 small val_MPJPE curve

| Epoch | val_MPJPE (mm) | train_loss | val_loss |
|-------|----------------|------------|----------|
| 1     | **18.31**      | 6.315      | 0.027590 |
| 2     | 45.26          | 5.908      | 0.028952 |
| 3     | 66.56          | 6.496      | 0.032757 |

## Observation

The v25 small configuration reaches its best validation MPJPE at **epoch 1**. After epoch 1 the model overfits: val_MPJPE monotonically increases while train_loss decreases, and val_loss also increases.

This indicates:
- The small subset (2000 train samples on A800, 500 on local 4090) is too small to support 20 epochs of training.
- The learning rate / optimizer setup does not generalise past the first epoch on this subset.
- Early stopping with `patience=3` and `min_delta=0.001` is required to preserve the epoch 1 checkpoint.

## Actions taken

- Added `--early_stopping_patience 3 --early_stopping_min_delta 0.001` to `scripts/run_v25_small_local_4090.sh`.
- Restarted local 4090 v25 small baseline to capture the best epoch 1 checkpoint.

## Next steps

1. Confirm local 4090 v25 small matches A800 ~18 mm at epoch 1.
2. Compare v25 baseline against v25+v18 top-k, v25+v27 UDP, and v25+outlier adaptive under the same early stopping.
3. Use the best combination as the new baseline for full-scale A800 runs.
