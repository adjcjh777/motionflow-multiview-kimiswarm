# v10 Proposal: Cardinality-Stratified Variable-View Training with View-Normalized Reprojection Loss

## 1. Problem in the current v7/v8/v9 pipeline

The variable-view training path in `experiments/train_omniview_fusion_v5_webbridge_multi.py` currently:

- Samples a random subset size `k ~ Uniform(min_views, k_max_eff)` **independently per batch** (lines 562–570).
- Optionally permutes the chosen view order (`variable_view_permute`).
- Applies a global curriculum that only raises `k_max_eff` from 4 to 14 over epochs.

This has three practical weaknesses that explain the v9 instability:

1. **Unbalanced view-count distribution.** Because `k` is sampled uniformly, batches with `k ≈ 2–4` are as frequent as batches with `k ≈ 12–14`. The model therefore sees many easy high-view examples and relatively few hard low-view examples, hurting robustness at small view counts.
2. **Reprojection loss scale depends on `k`.** `_reprojection_loss` masks inactive views, but the residual is summed over **all** `(B, T, V, J)` and divided by `mask.sum()`. In practice this still leaves a per-sample gradient scale that grows with `k` in a non-linear way because the Gauss-Newton / DLT backend is more stable with more views. The result is a loss term whose magnitude is hard to balance against the MSE loss and that can explode when the model has not yet learned reliable per-view confidences.
3. **Monotonic loss is expensive and noisy.** It requires a second full-view forward pass (line 666) and compares subset vs. full error with a fixed margin. In v9 it interacts badly with the reprojection loss and can dominate early training.

## 2. Specific, implementable v10 change

Introduce a **view-count-stratified training batch** and a **per-active-view reprojection normalization**.

### 2.1 Stratified view-count batching

Replace the per-batch random `k` with a batch sampler that guarantees each epoch covers the supported view counts in a controlled frequency.

```text
For each epoch:
    view_counts = curriculum(k_min, k_max_eff)
    For each batch in train_loader:
        k = sample_from_curriculum(view_counts, frequency="uniform_per_count")
        select exactly k active views per sample
        optionally permute selected views
```

Pseudo-code (new module `motionflow_mv/training/view_count_sampler.py`):

```python
def sample_k_for_epoch(k_min, k_max, epoch, total_epochs, alpha=2.0):
    """Return a list of k values whose distribution follows the epoch curriculum."""
    progress = min(1.0, epoch / max(1, total_epochs - 1))
    k_max_eff = k_min + int((k_max - k_min) * (progress ** alpha))
    # Ensure every count is seen at least once per epoch.
    return list(range(k_min, k_max_eff + 1))
```

### 2.2 View-normalized robust reprojection loss

Rewrite `_reprojection_loss` in `experiments/train_omniview_fusion_v5_webbridge_multi.py` so that the loss is normalized **per sample, per active view**, and **joint**, removing the `k`-dependent scale:

```python
def _reprojection_loss(pred_3d, points_2d, K, R, t, view_mask):
    # x_pred: (B, T, V, J, 2)
    diff = project(pred_3d, K, R, t) - points_2d
    mask = view_mask.unsqueeze(-1).unsqueeze(-1)  # (B, T, V, J, 1)
    sq = (diff ** 2) * mask
    # Sum over view and joint; normalize by active views * joints per sample.
    per_sample = sq.sum(dim=(2, 3))  # (B, T)
    active = mask.sum(dim=(2, 3)).squeeze(-1) + 1e-8  # (B, T)
    return (per_sample / active).mean()
```

Additionally, **clip per-sample reprojection loss** to a maximum value (e.g. `100.0 px²`) to guard against the large residuals that occur in early training with `k=2` and noisy camera perturbations.

### 2.3 Retire the monotonic loss in v10

Set `monotonic_loss_weight=0.0` for v10. The monotonic ranking loss is redundant once the reprojection loss is properly normalized and the stratified sampler guarantees the model sees the full-view baseline frequently. Re-introduce it only after v10 stabilizes.

### 2.4 Keep v8’s robust DLT path unchanged

Continue using `use_full_precision_dlt=True` and `use_robust_dlt_reweight=True` as in v8. The clamping of the precision matrix in `motionflow_mv/fusion/omniview_fusion_v5.py` (lines 511) already protects the robust reweight path; the v10 change is purely in how training batches and losses are constructed.

## 3. Validation plan

| Stage | Command / action | Pass criterion |
|-------|------------------|----------------|
| Smoke | `python experiments/train_omniview_fusion_v5_webbridge_multi.py --smoke --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 4` | Runs without error, loss finite |
| Small fast | Reuse `scripts/tmux_v9_small_fast.sh` but with `monotonic_loss_weight=0.0`, stratified sampler flag, and new reprojection loss. Train 10 epochs. | Step-50 loss < 150; no divergence |
| Per-k eval | After fast run, evaluate on validation set with fixed `k = 2, 4, 8, 14` active views. | MPJPE improves monotonically with `k`; low-k MPJPE lower than v9 baseline |
| Full run | 60 epochs on `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`, d=128, same hparams as v8. | Step-50 loss ~55–80 and decreasing; final MPJPE comparable or better than v8 |

Add a unit test in `tests/test_variable_view_training_v10.py` that:
- Builds a synthetic batch with `k=2` and `k=14`;
- Verifies the new `_reprojection_loss` returns the same scalar for the same per-view error regardless of `k`;
- Verifies the stratified sampler covers every `k` in the curriculum at least once per epoch.

## 4. Expected impact

- **Training stability.** Normalizing reprojection loss per active view removes the `k`-dependent scale jump that likely caused v9’s loss to blow up to ~3000. Early losses should stay in the same range as v7/v8 (~50–80 at step 50).
- **MPJPE / robustness.** Stratified sampling ensures the model trains on low-view cases more often, which should reduce MPJPE at `k < 8` without hurting full-view performance.
- **Runtime.** Retiring the monotonic loss removes the extra full-view forward pass per batch, saving ~10–15 % training time.

## 5. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Stratified sampler lowers effective batch diversity | Shuffle the list of `k` values and still randomize which exact views are selected; keep `variable_view_permute=True` |
| Per-view normalization makes high-k batches dominate | Monitor per-k validation MPJPE; if high-k degrades, add a small `1/k` weight to the reprojection term |
| Clipping hides large outliers too aggressively | Set clip at a percentile of the current batch rather than a fixed value, or start with a high clip (e.g. 200 px²) and anneal |
| Existing v7/v8 checkpoints are not directly comparable | Keep all other hyperparameters identical to v8; only the sampler and reprojection loss change |

## 6. Minimal flag set for a v10 run

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --use_mixed_loader \
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
  --use_full_precision_dlt \
  --use_robust_dlt_reweight \
  --use_domain_embedding \
  --use_camera_view_embedding \
  --use_set_view_aggregator \
  --use_variable_view_training \
  --variable_view_min_views 2 --variable_view_max_views 14 \
  --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 \
  --variable_view_permute \
  --use_stratified_view_count \
  --reproj_loss_weight 0.05 \
  --monotonic_loss_weight 0.0 \
  # ... remaining v8 flags unchanged
```

The only new trainer-side flag is `--use_stratified_view_count`; the reprojection change is a drop-in replacement of the existing `_reprojection_loss` function.
