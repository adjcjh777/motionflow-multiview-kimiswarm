# Agent-19: v47 Combined Architecture Proposal

**Task:** ANALYZE — Propose v47 combined architecture (v46-SVG + temporal aggregation)  
**Output:** `docs/proposals/v47_combined_architecture.md`  
**Tracking issue:** #160  

## Summary

v47 combines the upcoming **v46 Sparse-View Generalization (SVG)** module with a lightweight **temporal aggregation** head.  The idea is to compensate for the noise inherent in 2–3 view configurations by fusing information across time, while keeping the v46 per-frame geometry fusion and reliability head unchanged.

## Key design decisions

- **Builds on v46:** reuses `SparseViewGeneralizationV46`, view-dropout augmentation, and v45/v46 reliability weights.
- **Lightweight temporal head:** small transformer encoder (`d=64`, 2 layers, 4 heads) operating on per-frame triangulated 3D poses `(B, T, J, 3)`.
- **View-count conditioning:** each temporal token is concatenated with `log(n_views_t)` so the model discounts under-constrained frames.
- **Identity at init:** residual gate initialised to `0.0`; v46 behaviour is preserved at the start of training.
- **Staged training:** freeze v46 base for one epoch, then fine-tune end-to-end.

## Expected API

New module: `motionflow_mv/fusion/temporal_aggregation_v47.py`

```python
class TemporalAggregationV47(nn.Module):
    def forward(self, poses_3d, view_mask, clip_mask=None) -> torch.Tensor:
        # returns refined (B, T, J, 3)
```

New flag: `use_v47_temporal_aggregation`

## Evaluation target

- Improve v46 MPJPE by **≥5% at k ∈ {2, 3} views**.
- No regression at full views.
- Lower `temporal_jerk` (smoother trajectories).

## Risks noted

- Temporal head may over-smooth fast motion → mitigated by residual gate and tunable loss weight.
- Latency for streaming → default to a 7-frame local window; full-clip attention is optional.
- v46 must land first → v47 is queued behind #160.

## Full proposal

See `docs/proposals/v47_combined_architecture.md`.
