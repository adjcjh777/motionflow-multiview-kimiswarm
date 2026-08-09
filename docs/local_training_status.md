# Local RTX 4090 Training Status

Last checked: 2026-08-09 03:45:10 UTC

## Current run

- **Experiment:** v45-AGF smoke
- **Launch script:** `scripts/run_v45_agf_smoke_local_4090.sh`
- **Log file:** `outputs/v45_agf_smoke_local_4090.log`

## Latest progress

- **v25 + physical + domain:** Epoch 1 val_MPJPE 27.71 mm, epoch 2 val_MPJPE 53.63 mm — stopped due to overfit.
- **v45-AGF fast smoke:** Completed. `train_loss=14.34`, `val_loss=0.0333`, `val_MPJPE=116.78mm` (1 epoch, 50 samples; high but expected for tiny data; code path verified).
- **v45-AGF medium run:** Running; train step ~3600, loss ~5.85.
