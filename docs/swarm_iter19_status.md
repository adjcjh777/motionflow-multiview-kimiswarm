# Swarm Iteration 19 – Status & Plan

**Date**: 2026-08-07  
**Goal**: Push MotionFlow-MultiView toward ICRA/CVPR 2027 with larger-scale experiments, more complex models, and rigorous validation.

## Active workstreams

| # | Workstream | Owner | Status |
|---|-----------|-------|--------|
| 1 | 4090 dense+graph v2 full run | local watchdog | End-to-end Epoch 11, val_MPJPE=25.23 mm; full eval running |
| 2 | A800 MPI dense+graph v2 | local tmux | GPU 4, restarted with lr=3e-4 after divergence |
| 3 | A800 MPI OmniMultiViewFusion v3 | local | GPU 5, Epoch 1 done, val_MPJPE=25.72 mm |
| 4 | A800 H36M dense+graph v2 | local tmux | GPU 6, Epoch 20, val_MPJPE=21.69 mm; clean eval 15.03 mm |
| 5 | WebBridge multi-dataset loader | subagent | **Done** – pushed to main (commit `4475018`) |
| 6 | OmniMultiViewFusion v3 design | subagent | **Done** – pushed to main (commit `c9f2d2d`) |
| 7 | Robustness/eval pipeline | local | v3 eval script + smoke test pushed; auto-eval cron scheduled |
| 8 | 4090/A800 monitor | local | scripts/auto_eval_when_ready.sh runs every 30 min |

## Latest 4090 results

- **No-graph ablation**: clean MPJPE ~25.3 mm (baseline).
- **Dense+graph v2 (freeze phase)**: val_MPJPE = 25.13 mm → 25.37 mm across 5 freeze epochs.
- **A800 MPI v2**: diverged after unfreezing; restarted with lr=3e-4.
- **A800 H36M v2**: Epoch 20 val_MPJPE=21.69 mm; quick clean eval MPJPE=15.03 mm, PA-MPJPE=6.75 mm.

## Notes

- **Data quality**: H36M val/test (S9/S11) were 1000x too large in h36m_meters; corrected by converting from h36m_corrected. scripts/convert_h36m_to_meters.sh updated to handle all subjects.

## Next decisions

1. After dense+graph v2 finishes, if clean MPJPE < 9 mm, run robustness + official test-set inference.
2. If not, iterate on v3 or tune auxiliary-loss weights / view dropout.
3. Merge A800 results once available and compare scaling curves.

## Links

- GitHub issue #74: [Tracking] OmniMultiViewFusion v2 dense+graph v2 full run
- GitHub issue #75: [Tracking] Swarm iteration 19
