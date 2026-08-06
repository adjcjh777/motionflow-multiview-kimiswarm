# Occlusion-Robust Visibility Transformer (ORVT)

## One-sentence hypothesis

Replacing the per-view MLP visibility gate with a small **geometry-aware cross-view visibility transformer** that reasons jointly over views, joints, and camera rays will improve occlusion robustness while preserving the clean-MPJPE of the iter14 anchor.

## Related existing files/modules

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py` — iter14 anchor model; exposes the `_visibility_multiplier(feat, confidences)` hook.
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_visibility_model.py` — baseline visibility-gated variant (single MLP head, no cross-view reasoning).
- `motionflow_mv/fusion/visibility_gated_fusion.py` — older temporal-only visibility gate.
- `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py` — base class providing per-frame feature extractor and spatio-temporal transformer.
- `motionflow_mv/data/occlusion_aug.py` — synthetic view/joint occlusion augmentation.
- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` — training entry point for the anchor.
- `experiments/eval_occlusion_robustness.py` — evaluation harness for occlusion scenarios.

## Proposed code changes

### 1. New transformer visibility head

Create `motionflow_mv/fusion/cross_view_visibility_transformer.py`:

```python
class CrossViewVisibilityTransformer(nn.Module):
    """Predict per-view/per-joint visibility from spatio-temporal features.

    Tokens are (view, joint) pairs. Each token contains:
      - the ST feature f_{v,j} in R^d
      - the detector confidence c_{v,j}
      - a learned ray-direction embedding r_{v,j}
    A small transformer encoder lets views attend to each other so that
    occlusion decisions are consistent across the rig.
    """
```

Signature:
- `__init__(self, d: int, n_heads: int = 4, n_layers: int = 2, n_views: int = 4, j: int = 17)`
- `forward(self, feat, confidences, rays=None) -> logits (B*T, V, J)`

### 2. New model subclass

Create `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_visibility_transformer_model.py`:

```python
class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibilityTransformer(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
        visibility_n_layers: int = 2,
        visibility_n_heads: int = 4,
    ):
        ...
```

- Override `_visibility_multiplier(self, feat, confidences, rays=None)` to call the new head.
- Re-use all anchor machinery; only the visibility computation changes.

### 3. New supervised visibility loss

Create `motionflow_mv/losses/visibility_supervision_loss.py`:

```python
def visibility_supervision_loss(pred_logits, occluder_mask, confidences,
                                pos_weight=1.0, normalize=True):
    """BCE between predicted visibility logits and the ground-truth visible mask.

    occluder_mask: 1 = visible, 0 = artificially occluded during training.
    """
```

Export in `motionflow_mv/losses/__init__.py`.

### 4. Minimal training script patch

Add to `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`:

- `--model_type visibility_transformer`
- `--visibility_loss_weight` (default 0.0)
- `--occlusion_view_rate`, `--occlusion_joint_rate` (default 0.0)
- In the training loop, when `occlusion_view_rate + occlusion_joint_rate > 0`:
  1. Generate a synthetic occlusion mask `M` via `OcclusionAugmenter`.
  2. Apply it to `x`.
  3. Forward returns `(pred, weights, visibility_logits)`.
  4. Add `visibility_loss_weight * BCE(visibility_logits, M)`.

## Training/smoke plan (≤5 epochs on RTX 4090)

Use the MPI-INF-3DHP S2/Seq1 validation split as a proxy benchmark, matching the iter14 anchor protocol:

```bash
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --model_type visibility_transformer \
  --visibility_loss_weight 0.1 \
  --occlusion_view_rate 0.2 \
  --occlusion_joint_rate 0.1 \
  --epochs 5 --batch_size 8 --train_samples 2000
```

Estimated runtime on RTX 4090: **~45–60 min** for 5 epochs (the anchor 10-epoch run is ~1.5 h on the same GPU).

Smoke test:

```bash
python -m pytest tests/test_occlusion_robust_visibility_transformer.py -v
```

The smoke test will instantiate the model with a small clip, run forward/backward, and assert visibility logits have the correct shape and range.

## Success metrics

| Metric | Target | How measured |
|--------|--------|--------------|
| Clean MPJPE on MPI-INF-3DHP S2/Seq1 | ≤ 9.50 mm | `experiments/eval_full_metrics.py` with clean data |
| Occluded-view MPJPE (1 of 4 views dropped) | ≤ 11.0 mm | `experiments/eval_occlusion_robustness.py --occlusion_mode view --occlusion_rate 0.25` |
| Occluded-joint MPJPE (30% joints dropped per view) | ≤ 12.0 mm | `experiments/eval_occlusion_robustness.py --occlusion_mode joint --occlusion_rate 0.3` |
| Visibility AUC vs synthetic masks | ≥ 0.80 | Offline diagnostic on held-out clips |

**Stop rule:** If clean MPJPE degrades by >0.40 mm relative to the anchor (9.32 mm), the change is reverted or the visibility loss weight is reduced.

## Risk and fallback

- **Risk:** The transformer visibility head may overfit to synthetic occlusion patterns and hurt clean-MPJPE.
  - *Mitigation:* Keep the head small (≤2 layers, ≤4 heads), add dropout, and tune `visibility_loss_weight` from 0.05 to 0.2.
- **Risk:** Extra compute increases per-epoch time beyond the ≤5-epoch smoke budget.
  - *Mitigation:* Cache ray embeddings and share the feature extractor; the visibility head is only evaluated once per forward pass.
- **Risk:** The new loss destabilizes principal-point correction.
  - *Mitigation:* Use the existing `pp_pretrain_epochs` option to freeze the PP head before adding visibility supervision.
- **Fallback (fully reversible):** Delete or disable the new model class, revert `losses/__init__.py`, and run the original `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` with the same seed. No existing experiments are modified.
