# v47 Combined Architecture: Sparse-View Generalization + Temporal Aggregation

**Status:** Proposal / ready for design review  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #160 (depends on v46-SVG)  

## Motivation

v46 Sparse-View Generalization (SVG) makes `OmniMultiViewFusionV5` robust to missing views by training with random view dropout and a per-view reliability head.  When only a few views are available, however, the per-frame estimate can still be noisy because each instantaneous observation is under-constrained.  v47 closes that gap by adding a **temporal aggregation** stage: information from past and future frames is fused to compensate for the reduced number of simultaneous camera views.

The goal is a single architecture that is both **view-sparse** and **temporally-coherent**:

- Trains with variable/dropped views (v46).
- Aggregates evidence across time so that 2–3 view configurations approach the accuracy of 4+ view configurations.
- Keeps the per-frame triangulation foundation (v25) and the geometry-aware reliability head (v45/v46) unchanged.

## Design principles

1. **v46 first.** v47 is an *extension* of v46, not a replacement.  It reuses the `SparseViewGeneralizationV46` reliability weights and the view-dropout augmentation.
2. **Temporal aggregation is a lightweight post-triangulation smoother.** We do not replace per-frame geometry fusion with a heavy temporal model; we add a small temporal refinement head on top of the per-frame 3D poses.
3. **Handle variable-length clips and view masks.** The temporal module must accept missing frames/clips and missing views without special-casing.
4. **Identity at init.** With zeroed weights the temporal path is a no-op, preserving the v46 per-frame behaviour during warm-up.

## Proposed architecture

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

## Module API

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

## Integration plan

### Files touched (future IMPLEMENT task)

- `motionflow_mv/fusion/temporal_aggregation_v47.py` — new module.
- `motionflow_mv/fusion/omniview_fusion_v5.py` — add `use_v47_temporal_aggregation` flag and call the module.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — expose CLI flags and wire into the training loop.
- `tests/test_temporal_aggregation_v47.py` — unit tests for variable-length clips and view masks.
- `configs/benchmark_v47_temporal_svg_smoke.yaml` — smoke config.
- `scripts/run_v47_temporal_svg_smoke_local_4090.sh` — smoke script.

### New training flags

- `use_v47_temporal_aggregation` (default `False`)
- `v47_temporal_d_model` (default `64`)
- `v47_temporal_n_heads` (default `4`)
- `v47_temporal_num_layers` (default `2`)
- `v47_temporal_window` (default `None`, i.e. full-clip attention; set to `7` for local window)
- `v47_temporal_dropout` (default `0.1`)
- `v47_temporal_loss_weight` (default `0.01`)
- `v47_use_view_count_conditioning` (default `True`)

### Training recipe

1. Start from a trained v46-SVG checkpoint (or train v46 from scratch).
2. Freeze v25/v45/v46 weights for the first epoch to let the temporal head learn on stable per-frame estimates.
3. Unfreeze and fine-tune end-to-end with the same view-dropout curriculum.
4. Apply a small temporal smoothness loss on the refined trajectory:

```python
loss_temporal = v47_temporal_loss_weight * mean(|P_t - P_{t-1}|)
```

## Evaluation

Extend `experiments/eval_variable_views.py` to report, under variable view counts `k ∈ {2, 3, 4, full}`:

| Metric | Description |
|--------|-------------|
| `MPJPE@k` (v46) | Per-frame v46 baseline |
| `MPJPE@k` (v47) | With temporal aggregation |
| `MPJPE@k Δ` | Relative improvement |
| `temporal_jerk@k` | Mean 3rd derivative magnitude of refined trajectory |

Expected target: v47 improves v46 by **≥5% MPJPE at k ≤ 3** with no regression at full views and smoother temporal jerk.

## Experiments

| Stage | Hardware | Config | Goal |
|-------|----------|--------|------|
| Smoke | RTX 4090 | `configs/benchmark_v47_temporal_svg_smoke.yaml` | val_MPJPE < 75 mm on 500-sample run |
| Full | A800-D | v46-SVG checkpoint + v47 head | ≥5% improvement at 2–3 views over v46 |
| Ablation | RTX 4090 | no view-count conditioning / local window only / no smoothness loss | Identify necessary components |

## Success criteria

1. Smoke test passes with no NaN/OOM and val_MPJPE < 75 mm.
2. At 2–3 views, v47 outperforms v46 by ≥5% MPJPE.
3. No regression at full views versus v46.
4. `temporal_jerk` is lower (smoother) than v46.
5. A800 full run completes ≥1 epoch.

## Paper story fit

v47 supports the paper claim: *Our model is practical for real-world capture because it works with sparse and variable camera rigs **and** produces temporally coherent 3D human pose.*  The combination of view-sparse training and temporal aggregation mirrors how human observers compensate for partial visibility by integrating information over time.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Temporal head over-smoothes fast motion | Keep residual gate small at init; use clip-masked attention; tune `v47_temporal_loss_weight` |
| Added latency for online/streaming use | Default to a local 7-frame window; full-clip attention is optional |
| v46 not yet merged | Wait for v46-SVG branch (#160) to land; v47 builds on top of it |
| Training instability when unfreezing v46 | Use staged unfreeze: temporal head first, then all layers |

## Relation to other variants

- **v26/v35 temporal modules:** v47 is simpler and operates on the *output* 3D poses rather than on mid-level ray tokens.  It can be combined with v26/v35 if desired, but the standalone design keeps the change minimal.
- **v45-AGF:** v47 reuses v45 reliability weights as part of the v46 base; the temporal head sees only the triangulated pose.
- **v36/v37 self-evolution:** future work could let the temporal head also predict per-frame uncertainty and feed it back into v37 reliability gates.

## Next steps

1. Wait for v46-SVG implementation and smoke results (#160).
2. Implement `TemporalAggregationV47` and unit tests.
3. Wire v47 flag into `OmniMultiViewFusionV5` and the trainer.
4. Run smoke on RTX 4090 and compare with v46 baseline.
5. Queue full A800 run starting from the best v46 checkpoint.
