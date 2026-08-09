# v46 Sparse-View Generalization (SVG) for MotionFlow-MultiView

**Status:** Proposal / in-progress
**Labels:** `experiment`, `P1-next`
**Tracking issue:** #160
**Depends on:** #154, v45-AGF

## Motivation

Current best models (v25 geometry fusion, v45-AGF) are trained and validated with a fixed number of calibrated camera views. Real-world multi-view video capture, however, often produces missing, occluded, or ad-hoc camera configurations. v46 makes the fusion model robust to **sparse and variable camera counts** without overdesign: add random view-dropout training, a view-agnostic set aggregator, and a per-view reliability head that explicitly masks missing views.

## Design principles

1. **Geometry fusion remains the foundation.** We do not replace v25/v45-AGF; we make it robust to missing views.
2. **Sparse-view training is an augmentation.** No changes to the dataset itself—drop views on the fly.
3. **Variable-view evaluation is a first-class metric.** Report MPJPE@k for k = 2, 3, 4, and full views.
4. **Keep it small.** The new module is a lightweight MLP head; no extra graph or heavy transformer.

## Architecture

```text
Input (B, T, V, J, C) features from v25 geometry fusion
    |
    ▼
[ SparseViewGeneralizationV46 ]
    |
    ├── View-agnostic set aggregator (Induced Set Attention Block)
    │     handles variable V by treating views as an unordered set.
    |
    ├── Per-view reliability head
    │     predicts reliability r_v ∈ (0,1) for each available view,
    │     masked to 0 for dropped/missing views.
    |
    └── Weighted triangulation output
          feeds r_v into MultiViewGeometryFusionV25 weighted DLT.
```

## API

`motionflow_mv/fusion/sparse_view_generalization_v46.py` exposes:

```python
class SparseViewGeneralizationV46(nn.Module):
    def __init__(self, in_channels: int, n_views: int, hidden: int = 64):
        ...

    def forward(self, x: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, V, J, C) multi-view feature tokens.
            view_mask: (B, T, V) bool; True for available views.
        Returns:
            reliability: (B, T, V, J) positive weights, zero where masked.
        """
```

## Training flags

Add to `OmniMultiViewFusionV5` / trainer:

- `use_v46_sparse_view_generalization`
- `v46_svg_view_dropout_prob` (default 0.3)
- `v46_svg_min_views` (default 2)
- `v46_svg_hidden` (default 64)
- `v46_svg_use_curriculum` (default True)

## Data augmentation

`motionflow_mv/data/view_dropout_augmentation_v46.py`:

```python
def drop_views(views: torch.Tensor, cameras: dict, prob: float, min_views: int) -> tuple:
    """Randomly drop camera views for a clip during training."""
```

Drop is applied **inside the training loop** before feeding the model, preserving the original loader and manifest.

## Evaluation

Extend `experiments/eval_variable_views.py` to report:

- `MPJPE@k` for k = 2, 3, 4
- Dropout robustness curve
- Comparison against the same model trained without view dropout

## Experiments

| Stage | Hardware | Config | Goal |
|-------|----------|--------|------|
| Smoke | RTX 4090 | `configs/benchmark_v46_svg_smoke.yaml` | val_MPJPE < 80 mm on 500-sample run |
| Full  | A800-D   | `v46_svg_all_train` manifest | Match v45/v25 at full views; 10% better at 2-3 views |
| Ablation | RTX 4090 | no dropout / no reliability / no curriculum | Identify necessary components |

## Success criteria

1. Smoke test passes with no NaN/OOM and val_MPJPE < 80 mm.
2. Full A800 run starts and completes ≥1 epoch.
3. At 2-3 views, MPJPE improves by ≥10% over a v45/v25 model trained without view dropout.
4. No regression at full views.

## Paper story fit

v46 supports the paper claim: *Our multi-view fusion model is practical for real-world capture because it generalizes to sparse and variable camera configurations.*

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Existing `VariableViewSetAggregator` already handles variable views | Make v46 an explicit training augmentation + reliability head, not just a mask |
| View dropout breaks triangulation with <2 views | Enforce `min_views >= 2` and use robust fallback |
| Longer convergence | Use curriculum (start with full views, gradually increase dropout) |
