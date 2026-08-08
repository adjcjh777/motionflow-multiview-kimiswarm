# v31 `hierarchical_wider_d128`

## Problem statement

The v30 hardened hierarchical encoder fixed v29's instability with dataset-aware part groups, gated cross-scale fusion, and stochastic depth, but the smoke tests were forced to run at `d=64` to fit on the local RTX 4090. The v30a A800 run uses `d=128`, yet it still overfits after the first epoch unless the regularisation is tuned carefully. The core question for v31 is whether the v30 encoder can exploit a wider token dimension (`d=128`) without collapsing into the v29a-style overfitting pattern, provided the physical loss is properly warmed up and the v30 regularisers are active.

## Proposed change

Run the **v30 hardened hierarchical encoder at full width (`d=128`)** while keeping the backbone otherwise identical to v30a. Concretely:

- Set `--d 128 --residual_hidden 256 --n_st_layers 3` so the hierarchical encoder, residual MLPs, and ST transformer all operate at full width.
- Keep `--use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_n_heads 4`.
- Increase the v30 regularisation to match the wider capacity: `--v30_dropout 0.15 --v30_stochastic_depth_prob 0.15`.
- Keep variable-view training (`--use_variable_view_training ...`) so the wider encoder still learns permutation-invariant view mixing.
- Use the v29 physical-space temporal loss **with a 3-epoch linear warmup** (`--use_physical_space_temporal_loss_v29 --v29_physical_loss_warmup_epochs 3`) so the physical prior does not dominate early training.
- Add a small weight-decay penalty (`--weight_decay 1e-4`) to further discourage overfitting at `d=128`.
- **Do not use any TTE module** (TTE is broken and must not be used).

Because the v30 encoder's internal dimension is tied to the base model width `--d`, widening the encoder requires widening the whole model; there is no separate `v30_d` flag. This variant therefore tests the maximum-capacity v30 encoder under the current interface.

## Expected impact on `val_MPJPE` / overfitting

We expect epoch-1 `val_MPJPE` to be lower than the `d=64` smoke run because the wider cross-view attention has more capacity to model part/body relationships and camera geometry. If the v30 regularisers (gated residuals, stochastic depth, dropout) and the physical-loss warmup are sufficient, the validation curve should remain stable through epoch 3-5 rather than ballooning as in v29a. A successful run would produce a best `val_MPJPE` in the mid-to-high 20 mm range on the WebBridge/H36M/MPI mix and justify `d=128` as the default width for v31.

## Main risk

The main risk is **capacity-driven overfitting despite regularisation**. Wider features give the model more freedom to memorise training poses; if `d=128` plus variable-view training is still too much capacity, validation error will rise after epoch 1 just like v29a. The secondary risk is **OOM on the local 4090 smoke**, which is mitigated by reducing batch size and keeping the smoke script lightweight.
