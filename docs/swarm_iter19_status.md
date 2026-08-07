# Swarm Iteration 19 – Status & Plan

**Date**: 2026-08-07  
**Goal**: Push MotionFlow-MultiView toward ICRA/CVPR 2027 with larger-scale experiments, more complex models, and rigorous validation.

## Active workstreams

| # | Workstream | Owner | Status |
|---|-----------|-------|--------|
| 1 | 4090 dense+graph v2 full run | local watchdog | Freeze phase done (val_MPJPE=25.37 mm); end-to-end phase running |
| 2 | A800 dense+graph v2 scaling | local | Restarted on GPU 4 (PID 122582), freeze phase started |
| 3 | WebBridge multi-dataset loader | subagent | **Done** – pushed to main (commit `4475018`) |
| 4 | OmniMultiViewFusion v3 design | subagent | **Done** – pushed to main (commit `c9f2d2d`); A800 training running on GPU 5 |
| 5 | Robustness/eval pipeline | subagent | **Done** – pushed to main (commit `4475018`); no-graph eval paused to free GPU |
| 6 | 4090/A800 monitor | local | watching logs & GPUs, will run eval when checkpoint is ready |

## Latest 4090 results

- **No-graph ablation**: clean MPJPE ~25.3 mm (baseline).
- **Dense+graph v2 (freeze phase)**: val_MPJPE = 25.13 mm → 25.37 mm across 5 freeze epochs; now in end-to-end training.
- **A800**: v2 and v3 restarted on free GPUs 4/5 to avoid VLLM collisions; first epoch metrics pending.

## Next decisions

1. After dense+graph v2 finishes, if clean MPJPE < 9 mm, run robustness + official test-set inference.
2. If not, iterate on v3 or tune auxiliary-loss weights / view dropout.
3. Merge A800 results once available and compare scaling curves.

## Links

- GitHub issue #74: [Tracking] OmniMultiViewFusion v2 dense+graph v2 full run
- GitHub issue #75: [Tracking] Swarm iteration 19
