# v43 / v44 Decision Criteria

This document records the decision rules for the next iteration based on the outcome of the currently queued experiments.

## Currently queued key experiments

| Name | Config | What it tests |
|------|--------|---------------|
| v42 | d64, expanded H36M+MPI, v36+physical+domain | Whether v40/v41 gains require v37 |
| v43 base | v42 + per-node adaptive residual | Whether per-node residual helps at d64 |
| v43 scaled | d128, expanded H36M+MPI, 10k samples/seq | Whether model capacity helps |
| v43 all-train | d128, full WebBridge mixed (1333/156 files) | Whether more data helps |
| v25 all-train baseline | d128, full WebBridge mixed, v25 only | Whether v31-v43 complexity is justified |

## Decision tree for v44

1. **If v25 all-train baseline is the best (≈ or < 17 mm):**
   - The v31-v43 stack is not paying off.
   - v44 will be built on the **v25 geometry-fusion baseline** with only the smallest, highest-ROI additions (e.g., domain weights, physical loss, adaptive residual if it can be ported).

2. **If v43 all-train is the best and beats v25 all-train:**
   - The complex stack is justified when scaled.
   - v44 will keep the v43 architecture and add stronger regularization / longer training / SWA to fight epoch-1 overfitting.

3. **If v43 scaled (expanded data) is the best but v43 all-train is not:**
   - Data quantity/diversity is not the bottleneck; model capacity is.
   - v44 will scale further or simplify data while increasing capacity.

4. **If v42 beats v43 base:**
   - The per-node adaptive residual is not helping (or even hurting).
   - Drop it and focus on v42 + stronger regularization.

## Immediate next actions after results

- Compare epoch-1 val_MPJPE first; do not rely on later epochs because all runs overfit quickly.
- If the best run still overfits after epoch 1, add SWA, increase weight decay/dropout, or reduce model capacity.
- If no run beats the historical v25 full (17.17 mm), pivot to a v25-based v44.
