# v31 Ablation Proposal: `hierarchical_more_dropout`

## Problem Statement

The v30 hierarchical multi-view encoder (`HierarchicalViewEncoderV30`) adds joint-, part-, and body-scale cross-view attention with gated residual fusion. While it is designed to be identity-at-init and uses stochastic depth, the v30 smoke recipe still overfits quickly on the small WebBridge/H36M/MPI mixed subset: v29a reached 28.12 mm in epoch 1 but degraded to 47.85 mm by epoch 2, and the v30 smoke run uses only `v30_dropout=0.1` and `v30_stochastic_depth_prob=0.1`. With `n_part_layers=2` and d=64, the model has enough capacity to memorize the 200-clip smoke training set. Stronger regularization inside the hierarchical block is the cheapest next knob to turn before scaling capacity or adding data.

## Concrete Proposed Change

Create a single ablation run that keeps the v30 architecture and the v30 smoke recipe identical, but increases stochastic regularization in the hierarchical encoder:

- `v30_dropout`: 0.1 → **0.3**
- `v30_stochastic_depth_prob`: 0.1 → **0.2**
- Keep `v30_n_part_layers=2` and `use_hierarchical_multiview_v30`.
- Keep physical-space temporal loss with a 1-epoch warmup (`v29_physical_loss_warmup_epochs=1`).
- **Do not use TTE** in any form (per project constraint).
- Disable variable-view training for this ablation so the effect of dropout is not confounded with view-subset curriculum.

This is a pure hyper-parameter ablation: no source files are modified; the variant is expressed only through CLI flags in a new local smoke script.

## Expected Impact on val_MPJPE / Overfitting

- **Epoch 1 val_MPJPE** may be slightly worse than the 0.1-dropout baseline because more activations are zeroed during training.
- **Epoch 2–3 val_MPJPE** should degrade more slowly; the gap between train and val loss should shrink.
- If the hypothesis is correct, the best validation MPJPE will occur later in training and the final val_MPJPE will be lower than the v30 smoke baseline.
- The ablation directly tests whether v30 overfitting is a capacity-vs-data problem that can be mitigated by regularization rather than by architectural changes.

## Main Risk

Too much dropout can **underfit or destabilize the gated residual path**. `HierarchicalViewEncoderV30` initializes its output projections to zero and its residual gate near zero; aggressive dropout may prevent the gate from ever opening enough to learn meaningful multi-scale fusion. If train loss plateaus high while val loss stays flat, the dropout rates are too large and should be tapered (e.g., 0.2/0.15) or combined with more training data / longer warmup rather than pushed higher.

## Launch

Run as a local RTX 4090 smoke test via `scripts/launch_v31_hierarchical_more_dropout_local4090.sh`.
