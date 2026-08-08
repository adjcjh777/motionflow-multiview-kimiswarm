# v31 `outlier_view_adaptive_threshold`

## Problem statement

The v30 smoke run combines the hardened hierarchical encoder, variable-view training, and the physical-space temporal loss, but it leaves the v25 outlier-view detector disabled. As a result, views corrupted by the `outlier_view_prob=0.3` augmentation are handled only indirectly by robust DLT reweighting and geometry-aware attention. The robust loss can tolerate a few bad views, but it does not explicitly down-weight a view whose 2-D projections are globally inconsistent with the multi-view consensus. Earlier v25 experiments showed that the outlier-view detector can improve stability, yet its fixed z-score threshold (`z_thresh=3.0`, `soft_beta=1.0`) may be too lenient for the augmented WebBridge/H36M mixed setting and too aggressive for clean validation clips. A threshold that adapts online to the residual distribution should give better trade-off between rejecting true outliers and keeping clean-but-noisy views.

## Proposed change

Run a v31 smoke that starts from the v30 smoke recipe and explicitly enables the v25 outlier-view detector with a more adaptive operating point:

- Keep `--use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1` and the v29 physical-space temporal loss with `--v29_physical_loss_warmup_epochs 1`.
- Keep variable-view training and the set-view aggregator so the result is comparable to the v30 smoke baseline.
- Add `--v25_use_outlier_view_detector --v25_outlier_z_thresh 2.0 --v25_outlier_soft_beta 2.0`. The lower threshold and sharper soft-exp gate force the detector to react earlier to inconsistent views, while the existing `OutlierViewDetector` still learns a multiplicative scale on the threshold and beta.
- Retain the same outlier augmentation (`--outlier_view_prob 0.3 --outlier_view_max_views 1`) so the detector receives real corrupted views during training.
- Do not use any TTE module.

This is an ablation on how aggressively the adaptive threshold should act; the baseline comparison is the v30 smoke run that leaves the detector off.

## Expected impact on `val_MPJPE` / overfitting

We expect a small improvement in epoch-1 validation MPJPE relative to the v30 smoke because corrupted views are explicitly down-weighted before triangulation and geometry attention. The sharper, lower threshold should make the model less sensitive to the augmentation-induced variance and reduce late-epoch overfitting, provided the physical loss warmup keeps the floor/bone terms from dominating early. If the detector is too aggressive, validation may instead degrade because low-confidence but correct views are suppressed; the smoke run will reveal whether `z_thresh=2.0` is on the sensible side of that trade-off.

## Main risk

The main risk is **over-aggressive outlier suppression**. At `z_thresh=2.0` the detector may flag views that are merely noisy, especially on MPI-INF-3DHP clips where calibration or motion blur increases residual variance. This can leave too few reliable views and raise MPJPE. If the smoke shows degradation, the next step is to raise `z_thresh` back to 2.5–3.0 or to wire per-joint adaptive scales in `OutlierViewDetector` so different body parts can have different thresholds.
