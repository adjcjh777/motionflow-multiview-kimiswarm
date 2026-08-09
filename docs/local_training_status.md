# Local RTX 4090 Training Status

Last checked: 2026-08-09 03:45:10 UTC

## Current run

- **Experiment:** v45-AGF smoke
- **Launch script:** `scripts/run_v45_agf_smoke_local_4090.sh`
- **Log file:** `outputs/v45_agf_smoke_local_4090.log`

## Latest progress

- **v25 + physical + domain:** Epoch 1 val_MPJPE 27.71 mm, epoch 2 val_MPJPE 53.63 mm — stopped due to overfit.
- **v45-AGF smoke:** Running; train step ~2250, loss decreasing from 20.28 to ~6.16 (still running, see log).

## Notes

The v45-AGF smoke was started after stopping the regressing v25+physical+domain run. Result will guide whether to launch the full A800 v45-AGF run.
