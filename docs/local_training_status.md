# Local RTX 4090 Training Status

Last checked: 2026-08-09 03:45:10 UTC

## Current run

- **Experiment:** v25 + physical loss + domain weights
- **Launch script:** `scripts/run_v25_physical_domain_local_4090.sh`
- **Log file:** `outputs/v25_physical_domain_local_4090_full.log`

## Latest progress

- **Epoch 1 completed:** `train_loss=6.344643`, `val_loss=0.028768`, `val_MPJPE=27.71mm`
- **Epoch 2 in progress:** latest logged step is `train step 1200: loss=6.370257`
- **Log file size:** 101 lines (still being written)

## Notes

Training is continuing; the log is actively updated. The next expected entries are the remainder of epoch 2 training steps followed by the epoch 2 validation metrics.
