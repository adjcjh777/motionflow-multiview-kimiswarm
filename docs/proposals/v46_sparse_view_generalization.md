# v46 Sparse-View Generalization (SVG) for MotionFlow-MultiView

**Status:** Proposal / in-progress
**Labels:** `experiment`, `P1-next`
**Tracking issue:** #160
**Depends on:** #154, v45-AGF

---

## 1. Summary

v46 Sparse-View Generalization (SVG) makes `OmniMultiViewFusionV5` robust to **sparse and variable camera counts** without changing the core geometry-fusion backbone. Real-world multi-view capture often has missing, occluded, or ad-hoc cameras. v46 addresses this through three small additions:

1. **Random view-dropout training augmentation** — drop views on the fly during training.
2. **View-agnostic set aggregator** — treats available views as an unordered set, so any `V >= 2` works.
3. **Per-view reliability head** — predicts a masked reliability score for each view and feeds it into weighted triangulation.

At full views, the model should match v45/v25. At 2–3 views, it should outperform a model trained without view dropout.

---

## 2. Motivation

Our best models (v25 geometry fusion, v45-AGF) are trained and validated with a fixed number of calibrated cameras. In practice, captures may have:

- Occluded or temporarily missing cameras.
- Ad-hoc rigs with different numbers of views per sequence.
- Storage or bandwidth constraints that force view subsampling.

v46 makes the fusion model **practical for real-world capture** by generalizing to sparse and variable camera configurations, while keeping the change minimal and self-contained.

---

## 3. Design Principles

1. **Geometry fusion remains the foundation.** We do not replace v25/v45-AGF; we make it robust to missing views.
2. **Sparse-view training is an augmentation.** No changes to the dataset or manifest—views are dropped on the fly.
3. **Variable-view evaluation is a first-class metric.** Report `MPJPE@k` for `k = 2, 3, 4` and full views.
4. **Keep it small.** The new module is a lightweight MLP head; no extra graph or heavy transformer.

---

## 4. Architecture

```text
Input (B, T, V, J, C) features from v25 geometry fusion
    |
    ▼
[SparseViewGeneralizationV46]
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

### 4.1 Module API

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

---

## 5. Training Configuration

### 5.1 Model / Trainer Flags

Add to `OmniMultiViewFusionV5` / trainer:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v46_sparse_view_generalization` | bool | `False` | Master switch for the v46 module. |
| `v46_svg_view_dropout_prob` | float | `0.3` | Probability of dropping each view during training. |
| `v46_svg_min_views` | int | `2` | Minimum number of views retained after dropout. |
| `v46_svg_hidden` | int | `64` | Hidden dimension of the reliability MLP. |
| `v46_svg_use_curriculum` | bool | `True` | Start with full views, ramp dropout gradually. |

### 5.2 View-Dropout Augmentation

`motionflow_mv/data/view_dropout_augmentation_v46.py`:

```python
def drop_views(views: torch.Tensor, cameras: dict, prob: float, min_views: int) -> tuple:
    """Randomly drop camera views for a clip during training."""
```

Drop is applied **inside the training loop** before feeding the model, preserving the original loader and manifest.

---

## 6. Evaluation

Extend `experiments/eval_variable_views.py` to report:

- `MPJPE@k` for `k = 2, 3, 4`
- Dropout robustness curve
- Comparison against the same model trained without view dropout

---

## 7. Experiments

| Stage | Hardware | Config | Goal |
|-------|----------|--------|------|
| Smoke | RTX 4090 | `configs/benchmark_v46_svg_smoke.yaml` | val_MPJPE < 80 mm on 500-sample run |
| Full  | A800-D   | `v46_svg_all_train` manifest | Match v45/v25 at full views; 10% better at 2-3 views |
| Ablation | RTX 4090 | no dropout / no reliability / no curriculum | Identify necessary components |

---

## 8. Success Criteria

1. Smoke test passes with no NaN/OOM and val_MPJPE < 80 mm.
2. Full A800 run starts and completes ≥1 epoch.
3. At 2-3 views, MPJPE improves by ≥10% over a v45/v25 model trained without view dropout.
4. No regression at full views.

---

## 9. Paper Story Fit

v46 supports the paper claim: *Our multi-view fusion model is practical for real-world capture because it generalizes to sparse and variable camera configurations.*

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Existing `VariableViewSetAggregator` already handles variable views | Make v46 an explicit training augmentation + reliability head, not just a mask. |
| View dropout breaks triangulation with <2 views | Enforce `min_views >= 2` and use robust fallback. |
| Longer convergence | Use curriculum (start with full views, gradually increase dropout). |

---

# User Guide: Enabling v46 Sparse-View Generalization

## Quick Start

1. **Ensure you are on the `v46-svg` branch.**
2. **Use the smoke config to test locally:**
   ```bash
   bash scripts/run_v46_svg_smoke_local_4090.sh
   ```
3. **To enable in a custom run, add to your YAML config or CLI:**
   ```yaml
   model:
     use_v46_sparse_view_generalization: true
     v46_svg_view_dropout_prob: 0.3
     v46_svg_min_views: 2
     v46_svg_hidden: 64
     v46_svg_use_curriculum: true
   ```

## Enabling in Training

### YAML Configuration

```yaml
model:
  use_v46_sparse_view_generalization: true
  v46_svg_view_dropout_prob: 0.3
  v46_svg_min_views: 2
  v46_svg_hidden: 64
  v46_svg_use_curriculum: true

training:
  # Existing v25/v45-AGF training settings remain unchanged.
```

### CLI Override

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --config configs/benchmark_v46_svg_smoke.yaml \
  --use_v46_sparse_view_generalization \
  --v46_svg_view_dropout_prob 0.3 \
  --v46_svg_min_views 2
```

## Running Evaluation

After training, evaluate on variable view subsets:

```bash
python experiments/eval_variable_views.py \
  --checkpoint outputs/v46_svg_smoke/best.pth \
  --config configs/benchmark_v46_svg_smoke.yaml \
  --view_subsets 2,3,4,full \
  --out outputs/v46_svg_eval.json
```

The output JSON contains `MPJPE@k` entries for each requested subset.

## Interpreting Results

- **`MPJPE@full`**: Should be within ~1 mm of the v45/v25 baseline at full views.
- **`MPJPE@2`, `MPJPE@3`, `MPJPE@4`**: Should be lower (better) than the same model trained without view dropout.
- **Dropout robustness curve**: Evaluate across multiple dropout rates to understand degradation as views are removed.

## When to Use v46

Use v46 when:

- Your capture rig has a variable number of active cameras.
- You expect occlusions or missing views at inference time.
- You want a single model that works across 2–8+ views without retraining.

Do not use v46 if:

- You always have a fixed, full camera rig and want the absolute lowest full-view error (v25/v45-AGF baseline is sufficient).
- Your minimum available views are < 2 (triangulation is ill-posed).

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NaN` during training | Dropout left < 2 views or reliability weights exploded. | Lower `v46_svg_view_dropout_prob`, raise `v46_svg_min_views`. |
| Slower convergence | Curriculum not enabled or dropout too aggressive too early. | Set `v46_svg_use_curriculum: true`. |
| No improvement at sparse views | Base model was already robust; or dropout rate too low. | Increase `v46_svg_view_dropout_prob` to 0.5 and add ablations. |

## See Also

- Issue #160 — v46 tracking
- Issue #154 — v25 all-train baseline dependency
- `docs/swarm_iter23_action_plan.md` — full agent task list
- `motionflow_mv/fusion/sparse_view_generalization_v46.py` — module implementation
- `motionflow_mv/data/view_dropout_augmentation_v46.py` — dropout helper
