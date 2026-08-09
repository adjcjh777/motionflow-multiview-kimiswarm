# v47 Combined Architecture: Sparse-View Generalization + Temporal Aggregation

**Status:** Proposal / implementation in progress
**Labels:** `experiment`, `P1-next`
**Tracking issue:** #162
**Depends on:** #160 (v46-SVG)

## 1. Summary

v47 adds a lightweight **temporal aggregation head** on top of v46 Sparse-View Generalization (SVG). v46 makes `OmniMultiViewFusionV5` robust to missing views by training with random view dropout and a per-view reliability head, but per-frame estimates can still be noisy when only 2–3 views are available. v47 closes that gap by fusing evidence across time, so sparse-view configurations approach the accuracy of full-view capture without changing the underlying geometry-fusion backbone.

The resulting architecture is both **view-sparse** and **temporally coherent**:

- Trains with variable/dropped views (v46).
- Aggregates evidence across time so that 2–3 view configurations approach the accuracy of 4+ view configurations.
- Keeps the per-frame triangulation foundation (v25) and the geometry-aware reliability head (v45/v46) unchanged.

## 2. Motivation

When only a few camera views are available, each instantaneous 3D estimate is under-constrained and noisy. Human observers compensate for partial visibility by integrating information over time; v47 mirrors that behaviour with a small transformer head that refines per-frame poses using neighbouring frames.

The change is intentionally minimal:

- It reuses the v46 base model unchanged.
- It operates on the **output** 3D poses rather than on mid-level ray tokens, keeping the module small and easy to freeze/unfreeze.
- It starts as the identity map, so training is stable and the v46 baseline is preserved at warm-up.

## 3. Design Principles

1. **v46 first.** v47 is an *extension* of v46, not a replacement. It reuses the `SparseViewGeneralizationV46` reliability weights and the view-dropout augmentation.
2. **Temporal aggregation is a lightweight post-triangulation smoother.** We do not replace per-frame geometry fusion with a heavy temporal model; we add a small temporal refinement head on top of the per-frame 3D poses.
3. **Handle variable-length clips and view masks.** The temporal module accepts missing frames/clips and missing views without special-casing.
4. **Identity at init.** With zeroed weights the temporal path is a no-op, preserving v46 per-frame behaviour during warm-up.

## 4. Architecture

```text
Input: (B, T, V, J, 2) 2D keypoints + cameras (K, R, t)
        |
        ▼
[ v46 Sparse-View Generalization ]
        |
        ├── View-dropout augmentation (training only)
        ├── v25 MultiViewGeometryFusionV25 (per-frame geometry fusion)
        ├── v45 AdaptiveGeometryFusionV45 reliability weights
        └── Sparse-view triangulated pose P_t  (B, T, J, 3)
                |
                ▼
        [ v47 Temporal Aggregation Module ]
                |
                ├── Temporal attention over (time, joint) tokens
                │     queries/keys share the same (time, joint) layout,
                │     masked for clip boundaries and missing views.
                ├── Sparse-view aware positional bias
                │     encodes how many views contributed to each frame
                │     so the module discounts under-constrained frames.
                └── Residual refinement ΔP_t  (B, T, J, 3)
                        |
                        ▼
                Final refined pose  P_t + g(t) · ΔP_t
```

The architecture is deliberately shallow:

- **Per-frame base:** v25 + v45/v46 (unchanged).
- **Temporal head:** a small transformer encoder with `num_layers=2`, `d_model=64`, `n_heads=4`, operating on `(joint, 3)` trajectories per batch element.
- **View-count conditioning:** each token is concatenated with a scalar `log(n_views_t)` so the temporal head knows how much to trust each frame.
- **Residual gate:** a learnable scalar `g` initialised to `0.0`; the temporal path warms up from identity.

## 5. Module API

`motionflow_mv/fusion/temporal_aggregation_v47.py`:

```python
class TemporalAggregationV47(nn.Module):
    def __init__(
        self,
        n_joints: int = 17,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        temporal_window: Optional[int] = None,
        dropout: float = 0.1,
        residual_gate_init: float = 0.0,
        use_view_count_conditioning: bool = True,
    ):
        ...

    def forward(
        self,
        poses_3d: torch.Tensor,        # (B, T, J, 3)
        view_mask: torch.Tensor,       # (B, T, V)
        clip_mask: Optional[torch.Tensor] = None,  # (B, T) True = valid frame
    ) -> torch.Tensor:
        """Return temporally refined 3D poses (B, T, J, 3)."""
        ...
```

The module is called from a new `use_v47_temporal_aggregation` path inside `OmniMultiViewFusionV5` after the v46 triangulation output.

## 6. Integration Plan

### 6.1 Files Touched

| File | Change |
|------|--------|
| `motionflow_mv/fusion/temporal_aggregation_v47.py` | New temporal aggregation module. |
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Add `use_v47_temporal_aggregation` flag and call the module. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Expose CLI flags and wire into the training loop. |
| `tests/test_temporal_aggregation_v47.py` | Unit tests for variable-length clips and view masks. |
| `configs/benchmark_v47_temporal_svg_smoke.yaml` | Smoke config. |
| `scripts/run_v47_temporal_svg_smoke_local_4090.sh` | Smoke script. |

### 6.2 New Training Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v47_temporal_aggregation` | bool | `False` | Master switch for the v47 temporal head. |
| `v47_temporal_d_model` | int | `64` | Hidden dimension of the temporal transformer. |
| `v47_temporal_n_heads` | int | `4` | Number of attention heads. |
| `v47_temporal_num_layers` | int | `2` | Number of transformer encoder layers. |
| `v47_temporal_window` | int | `None` | Temporal window size; `None` = full-clip attention. |
| `v47_temporal_dropout` | float | `0.1` | Dropout inside the temporal transformer. |
| `v47_temporal_loss_weight` | float | `0.01` | Weight of the temporal smoothness loss. |
| `v47_use_view_count_conditioning` | bool | `True` | Concatenate `log(n_views_t)` to each token. |

### 6.3 Training Recipe

1. Start from a trained v46-SVG checkpoint (or train v46 from scratch).
2. Freeze v25/v45/v46 weights for the first epoch to let the temporal head learn on stable per-frame estimates.
3. Unfreeze and fine-tune end-to-end with the same view-dropout curriculum.
4. Apply a small temporal smoothness loss on the refined trajectory:

```python
loss_temporal = v47_temporal_loss_weight * mean(|P_t - P_{t-1}|)
```

## 7. Evaluation

Extend `experiments/eval_variable_views.py` to report, under variable view counts `k ∈ {2, 3, 4, full}`:

| Metric | Description |
|--------|-------------|
| `MPJPE@k` (v46) | Per-frame v46 baseline |
| `MPJPE@k` (v47) | With temporal aggregation |
| `MPJPE@k Δ` | Relative improvement |
| `temporal_jerk@k` | Mean 3rd derivative magnitude of refined trajectory |

Expected target: v47 improves v46 by **≥5% MPJPE at k ≤ 3** with no regression at full views and smoother temporal jerk.

## 8. Experiments

| Stage | Hardware | Config | Goal |
|-------|----------|--------|------|
| Smoke | RTX 4090 | `configs/benchmark_v47_temporal_svg_smoke.yaml` | val_MPJPE < 75 mm on 500-sample run |
| Full | A800-D | v46-SVG checkpoint + v47 head | ≥5% improvement at 2–3 views over v46 |
| Ablation | RTX 4090 | no view-count conditioning / local window only / no smoothness loss | Identify necessary components |

## 9. Success Criteria

1. Smoke test passes with no NaN/OOM and val_MPJPE < 75 mm.
2. At 2–3 views, v47 outperforms v46 by ≥5% MPJPE.
3. No regression at full views versus v46.
4. `temporal_jerk` is lower (smoother) than v46.
5. A800 full run completes ≥1 epoch.

## 10. Paper Story Fit

v47 supports the paper claim: *Our model is practical for real-world capture because it works with sparse and variable camera rigs **and** produces temporally coherent 3D human pose.* The combination of view-sparse training and temporal aggregation mirrors how human observers compensate for partial visibility by integrating information over time.

## 11. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Temporal head over-smoothes fast motion | Keep residual gate small at init; use clip-masked attention; tune `v47_temporal_loss_weight` |
| Added latency for online/streaming use | Default to a local 7-frame window; full-clip attention is optional |
| v46 not yet merged | Wait for v46-SVG branch (#160) to land; v47 builds on top of it |
| Training instability when unfreezing v46 | Use staged unfreeze: temporal head first, then all layers |

## 12. Relation to Other Variants

- **v26/v35 temporal modules:** v47 is simpler and operates on the *output* 3D poses rather than on mid-level ray tokens. It can be combined with v26/v35 if desired, but the standalone design keeps the change minimal.
- **v45-AGF:** v47 reuses v45 reliability weights as part of the v46 base; the temporal head sees only the triangulated pose.
- **v36/v37 self-evolution:** future work could let the temporal head also predict per-frame uncertainty and feed it back into v37 reliability gates.

## 13. Next Steps

1. Wait for v46-SVG implementation and smoke results (#160).
2. Implement `TemporalAggregationV47` and unit tests.
3. Wire v47 flag into `OmniMultiViewFusionV5` and the trainer.
4. Run smoke on RTX 4090 and compare with v46 baseline.
5. Queue full A800 run starting from the best v46 checkpoint.

---

# User Guide: Enabling v47 Temporal Aggregation

## Quick Start

1. **Ensure you are on the `v47-temporal` branch and v46-SVG is enabled.**
2. **Run the smoke test locally:**
   ```bash
   bash scripts/run_v47_temporal_svg_smoke_local_4090.sh
   ```
3. **To enable in a custom run, add to your YAML config or CLI:**
   ```yaml
   model:
     use_v47_temporal_aggregation: true
     v47_temporal_d_model: 64
     v47_temporal_n_heads: 4
     v47_temporal_num_layers: 2
     v47_temporal_window: null
     v47_temporal_dropout: 0.1
     v47_temporal_loss_weight: 0.01
     v47_use_view_count_conditioning: true
   ```

## YAML Configuration

A minimal v47-enabled YAML snippet is shown below. It assumes v25/v45/v46 are already enabled; only the v47 flags need to be added.

```yaml
model:
  use_v46_sparse_view_generalization: true
  v46_svg_view_dropout_prob: 0.3
  v46_svg_min_views: 2
  v46_svg_hidden: 64
  v46_svg_use_curriculum: true

  # v47 temporal head
  use_v47_temporal_aggregation: true
  v47_temporal_d_model: 64
  v47_temporal_n_heads: 4
  v47_temporal_num_layers: 2
  v47_temporal_window: null        # null = full-clip attention
  v47_temporal_dropout: 0.1
  v47_temporal_loss_weight: 0.01
  v47_use_view_count_conditioning: true

training:
  # Existing v46 training settings remain unchanged.
```

## CLI Override

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --config configs/benchmark_v47_temporal_svg_smoke.yaml \
  --use_v47_temporal_aggregation \
  --v47_temporal_d_model 64 \
  --v47_temporal_n_heads 4 \
  --v47_temporal_num_layers 2 \
  --v47_temporal_window 7 \
  --v47_temporal_dropout 0.1 \
  --v47_temporal_loss_weight 0.01 \
  --v47_use_view_count_conditioning
```

## Running Evaluation

After training, evaluate on variable view subsets:

```bash
python experiments/eval_variable_views.py \
  --checkpoint outputs/v47_temporal_svg_smoke/best.pth \
  --config configs/benchmark_v47_temporal_svg_smoke.yaml \
  --view_subsets 2,3,4,full \
  --out outputs/v47_temporal_svg_eval.json
```

The output JSON contains `MPJPE@k` and `temporal_jerk@k` for each requested subset.

## Interpreting Results

- **`MPJPE@full`**: Should be within ~1 mm of the v46 baseline at full views.
- **`MPJPE@2`, `MPJPE@3`, `MPJPE@4`**: Should be lower (better) than the v46 baseline trained without temporal aggregation.
- **`temporal_jerk@k`**: Lower values indicate smoother trajectories. v47 should reduce jerk, especially at sparse views.

## When to Use v47

Use v47 when:

- Your capture rig is sparse (2–4 views) and per-frame triangulation is noisy.
- You need temporally coherent output for downstream animation or biomechanics.
- You already have a v46-SVG checkpoint and want a quick incremental gain.

Do not use v47 if:

- You need a purely online/realtime pipeline and cannot tolerate full-clip latency (use a local `v47_temporal_window` instead, or skip v47).
- Your sequences are very short (< 3 frames), because the temporal head has little context to exploit.

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NaN` during training | Residual gate too large or temporal loss weight too high. | Lower `v47_temporal_loss_weight`, reduce `v47_temporal_num_layers`, or use a smaller learning rate. |
| Over-smoothed fast motion | Temporal window too wide or smoothness loss too strong. | Reduce `v47_temporal_window` to 7, or lower `v47_temporal_loss_weight`. |
| No improvement at sparse views | v46 base is not yet trained; temporal head has poor per-frame input. | Train or load a strong v46 checkpoint before enabling v47. |
| OOM on long clips | Full-clip attention with large `clip_len`. | Set `v47_temporal_window` to a fixed value (e.g., 7) or reduce batch size. |

## See Also

- Issue #162 — v47 tracking
- Issue #160 — v46-SVG dependency
- `docs/swarm_iter24_action_plan.md` — full agent task list
- `motionflow_mv/fusion/temporal_aggregation_v47.py` — module implementation
- `motionflow_mv/fusion/sparse_view_generalization_v46.py` — v46 base module
- `motionflow_mv/data/view_dropout_augmentation_v46.py` — v46 dropout helper
