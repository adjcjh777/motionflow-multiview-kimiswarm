# v31 `hierarchical_no_variable_views`

## Problem statement

v30 hardened the v29 hierarchical encoder with dataset-aware part groups, gated cross-scale fusion, and stochastic depth. The v30 smoke run combines it with variable-view training (random view subsets + permutations), the set-view aggregator, and the physical-space temporal loss. While variable-view training is meant to make the model robust to arbitrary view subsets, it is also a strong source of training noise: each batch exposes a different effective number of views, and the hierarchical part/body attention may be receiving conflicting part-token statistics. The result is often an epoch-1 validation drop followed by rapid overfitting (v29a went from 28.12 mm at epoch 1 to 81.08 mm by epoch 3). We need an ablation that isolates the contribution of the hierarchical encoder from the variable-view curriculum.

## Proposed change

Run the **v30 hierarchical encoder with a fixed, full set of views** throughout training. Concretely:

- Keep `--use_hierarchical_multiview_v30` with the same v30 settings (`--v30_n_part_layers 2`, `--v30_stochastic_depth_prob 0.1`).
- **Remove** `--use_variable_view_training` and all its curriculum flags (`--variable_view_min_views`, `--variable_view_max_views`, `--variable_view_max_views_start`, `--variable_view_curriculum_alpha`, `--variable_view_permute`).
- Keep `--use_camera_view_embedding` and `--use_set_view_aggregator` so the model remains permutation-invariant over views, but the training batches always contain all available views in fixed order.
- Keep the v29 physical-space temporal loss, but with a warmup (`--v29_physical_loss_warmup_epochs 1`) so the loss does not dominate early training.
- Do **not** use any TTE module.

This is the simplest possible v31 baseline: it tells us whether the v30 hierarchical encoder can converge stably when it is not also asked to learn a variable-view curriculum.

## Expected impact on `val_MPJPE` / overfitting

We expect a smoother validation curve than v29a. Without variable-view subsets, the model sees the same camera geometry in every batch, so the hierarchical encoder can learn cleaner part/body correspondences. In the smoke setting, epoch-1 `val_MPJPE` should be comparable to or slightly better than the v30 smoke run; the real question is whether the later epochs still overfit. If overfitting disappears, we can conclude that the variable-view curriculum was the main cause of the v29/v30 instability. If overfitting remains, the problem lies in the hierarchical encoder capacity or the physical loss schedule.

## Main risk

The main risk is **reduced robustness at test time**. The model will not have been trained on dropped or permuted views, so its behavior under missing views may degrade. This is acceptable for an ablation: it directly measures whether variable-view training is worth the noise it introduces.
