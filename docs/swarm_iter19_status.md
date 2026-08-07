# Swarm Iteration 19 – Status & Plan

**Date**: 2026-08-07  
**Goal**: Push MotionFlow-MultiView toward ICRA/CVPR 2027 with larger-scale experiments, more complex models, and rigorous validation.

## Active workstreams

| # | Workstream | Owner | Status |
|---|-----------|-------|--------|
| 1 | 4090 dense+graph v2 full run | local watchdog | End-to-end Epoch 6, val_MPJPE=25.40 mm |
| 2 | A800 MPI dense+graph v2 | local tmux | GPU 4, freeze epoch 3, val_MPJPE=25.62 mm |
| 3 | A800 MPI OmniMultiViewFusion v3 | local | GPU 5, first epoch in progress |
| 4 | A800 H36M dense+graph v2 | local | GPU 6, just started |
| 5 | WebBridge multi-dataset loader | subagent | **Done** – pushed to main (commit `4475018`) |
| 6 | OmniMultiViewFusion v3 design | subagent | **Done** – pushed to main (commit `c9f2d2d`) |
| 7 | Robustness/eval pipeline | local | v3 eval script + smoke test pushed; waiting for checkpoint |
| 8 | 4090/A800 monitor | local | watching logs & GPUs, will run eval when checkpoint ready |

## Latest 4090 results

- **No-graph ablation**: clean MPJPE ~25.3 mm (baseline).
- **Dense+graph v2 (freeze phase)**: val_MPJPE = 25.13 mm → 25.37 mm across 5 freeze epochs.
- **A800 MPI v2**: val_MPJPE improved to 25.07 mm at freeze epoch 2, currently 25.62 mm at epoch 3.
- **A800 H36M v2**: launched to expand scale beyond MPI-INF-3DHP.

## Next decisions

1. After dense+graph v2 finishes, if clean MPJPE < 9 mm, run robustness + official test-set inference.
2. If not, iterate on v3 or tune auxiliary-loss weights / view dropout.
3. Merge A800 results once available and compare scaling curves.

## Links

- GitHub issue #74: [Tracking] OmniMultiViewFusion v2 dense+graph v2 full run
- GitHub issue #75: [Tracking] Swarm iteration 19
