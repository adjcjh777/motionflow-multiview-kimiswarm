# Local RTX 4090 Training Status

Last checked: 2026-08-09 03:45:10 UTC

## Current run

- **Experiment:** v45-AGF smoke
- **Launch script:** `scripts/run_v45_agf_smoke_local_4090.sh`
- **Log file:** `outputs/v45_agf_smoke_local_4090.log`

## Latest progress

- **v25 + physical + domain:** Epoch 1 val_MPJPE 27.71 mm, epoch 2 val_MPJPE 53.63 mm — stopped due to overfit.
- **v45-AGF smoke:** Original smoke (`train_samples=500, epochs=2`) was training healthily but too slow; stopped. Fast smoke (`train_samples=50, epochs=1`) now running to get a quick val_MPJPE.
