# Swarm Iteration 19 – Status & Plan

**Date**: 2026-08-07  
**Goal**: Push MotionFlow-MultiView toward ICRA/CVPR 2027 with larger-scale experiments, more complex models, and rigorous validation.

## Active workstreams

| # | Workstream | Owner | Status |
|---|-----------|-------|--------|
| 1 | 4090 dense+graph v2 full run | local watchdog | Freeze Epoch 1 done (val_MPJPE=25.13 mm); 4 freeze epochs remaining |
| 2 | A800 dense+graph v2 scaling | subagent | cloning + data sync in progress |
| 3 | WebBridge multi-dataset loader | subagent | manifests + loader in progress |
| 4 | OmniMultiViewFusion v3 design | subagent | design doc + prototype in progress |
| 5 | Robustness/eval pipeline | subagent | variable-view + camera-perturbation scripts in progress |
| 6 | 4090 monitor | subagent | watching log & GPU every 10–15 min |

## Latest 4090 results

- **No-graph ablation**: clean MPJPE  25.3 mm (baseline).
- **Dense+graph v2 (freeze phase)**: Epoch 1 val_MPJPE = 25.13 mm.

## Next decisions

1. After dense+graph v2 finishes, if clean MPJPE < 9 mm, run robustness + official test-set inference.
2. If not, iterate on v3 or tune auxiliary-loss weights / view dropout.
3. Merge A800 results once available and compare scaling curves.

## Links

- GitHub issue #74: [Tracking] OmniMultiViewFusion v2 dense+graph v2 full run
- GitHub issue #75: [Tracking] Swarm iteration 19
