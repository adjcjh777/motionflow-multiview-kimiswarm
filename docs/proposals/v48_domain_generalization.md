# v48: Domain Generalization and 3DPW Integration

**Status:** Proposal / ready for design review  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #162 (depends on v47-temporal)  

## Motivation

v46 Sparse-View Generalization makes the model robust to missing views, and v47 Temporal Aggregation fuses evidence across time to compensate for sparse observations. The next logical step is **domain generalization**: ensuring the same model works well across studio multi-view (H36M, MPI-INF-3DHP, AIST++) and in-the-wild monocular (3DPW) data, without per-domain re-training.

v48 closes three remaining gaps:

1. **In-the-wild data is under-used.** 3DPW is already in the mixed manifest, but only in `pseudo` mode (synthetic 4-view rig). The `actual` single-camera mode is the closest to real-world deployment and is currently ignored.
2. **Domain-specific dynamics are not modeled.** The temporal head treats studio and in-the-wild motion with the same smoothing, even though frame rates, noise, and occlusions differ.
3. **Static domain weights are brittle.** The current v41 domain-weighted loss uses hand-tuned scalars and cannot adapt to per-domain difficulty during training.

## Design principles

1. **Build on v46/v47.** v48 is an extension, not a replacement. It reuses sparse-view reliability, view-dropout augmentation, and temporal aggregation.
2. **Minimal new modules.** Reuse existing `DomainAdaptationWrapper` GRL+FiLM skeleton and the v41 DDWL design where possible.
3. **Actual-mode 3DPW as a first-class benchmark.** Add a loader and eval path for the real moving-camera 3DPW sequences.
4. **Domain-invariant, not domain-specific.** The goal is a single checkpoint that generalizes, not a suite of per-domain adapters.

## Proposed architecture

```text
Input: (B, T, V, J, 2/3) 2D keypoints + cameras
        |
        ▼
[ v46 Sparse-View Generalization ]
        |
        ├── View-dropout augmentation (domain-aware in v48)
        ├── v25 MultiViewGeometryFusionV25
        ├── v45 AdaptiveGeometryFusionV45 reliability weights
        └── Sparse-view triangulated pose P_t  (B, T, J, 3)
                |
                ▼
        [ v47 Temporal Aggregation Module ]
                |
                ├── Temporal attention over (time, joint) tokens
                ├── Domain-conditional FiLM offsets (new in v48)
                ├── View-count positional bias
                └── Residual refinement ΔP_t
                        |
                        ▼
        [ v48 Domain-Invariant Sparse-View Refinement ]
                |
                ├── Instance-normalized reliability features
                ├── Gradient-reversal domain regularization
                └── Final refined pose P'_t
```

### Module API

`motionflow_mv/fusion/domain_generalization_v48.py`:

```python
class DomainInvariantSparseViewV48(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_views: int,
        num_domains: int = 6,  # h36m, mpi, aist, shelf, campus, 3dpw
        hidden: int = 64,
        dropout: float = 0.1,
        use_grl: bool = True,
        grl_lambda: float = 0.1,
    ):
        ...

    def forward(
        self,
        feat: torch.Tensor,        # (B, T, V, J, C)
        dataset_id: torch.Tensor,  # (B,)
        view_mask: torch.Tensor,   # (B, T, V)
    ) -> torch.Tensor:
        """Return domain-invariant per-view weights (B, T, V, J)."""
        ...
```

### 3DPW actual-mode loader

`motionflow_mv/data/webbridge_3dpw_actual_loader.py`:

```python
class WebBridge3DPWActualDataset(Dataset):
    """Load 3DPW `actual` .npz files with per-frame moving camera.

    Returns the same fields as WebBridgeCanonical17Dataset but with
    V=1 and time-varying camera intrinsics/extrinsics.
    """
    ...
```

## Integration plan

### Files touched (future IMPLEMENT task)

- `motionflow_mv/fusion/domain_generalization_v48.py` — new domain-invariant sparse-view refinement.
- `motionflow_mv/data/webbridge_3dpw_actual_loader.py` — new actual-mode loader.
- `motionflow_mv/data/view_dropout_augmentation_v46.py` — per-dataset dropout schedule.
- `motionflow_mv/fusion/temporal_aggregation_v47.py` — optional `dataset_id` input for domain-conditional offsets.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — v48 CLI flags, DDWL, pass `dataset_id` to v47 head.
- `experiments/eval_variable_views.py` — 3DPW actual-mode benchmark and per-domain MPJPE@k.
- `configs/benchmark_v48_domain_generalization_smoke.yaml` — smoke config.
- `scripts/run_v48_domain_generalization_smoke_local_4090.sh` — smoke script.

### New training flags

- `use_v48_domain_generalization` (default `False`)
- `v48_dg_hidden` (default `64`)
- `v48_dg_grl_lambda` (default `0.1`)
- `v48_dg_use_domain_film` (default `True`)
- `v48_dg_use_ddwl` (default `True`)
- `v48_dg_ddwl_temperature` (default `2.0`)
- `v48_dg_ddwl_warmup_epochs` (default `1`)
- `v48_3dpw_actual_val_paths` (default `None`)
- `v48_dropout_per_domain` (default `{"0": 0.30, "1": 0.30, "5": 0.15}`)

### Training recipe

1. Warm-start from the best v47 checkpoint.
2. Freeze v25/v45/v46/v47 weights for 1 epoch; train only v48 domain-invariant wrapper and DDWL state.
3. Unfreeze all layers and continue with the full mixed manifest (H36M/MPI/AIST/3DPW pseudo).
4. Apply per-domain view dropout: 3DPW gentler (`p=0.15`) than studio domains (`p=0.30`).
5. Use DDWL to up-weight the hardest domain (usually DPW early, MPI later).
6. Validate on H36M/MPI/AIST val, 3DPW pseudo val, and 3DPW actual val.

## Evaluation

Extend `experiments/eval_variable_views.py` to report, per domain and view count `k`:

| Metric | Description |
|--------|-------------|
| `MPJPE@k` (per domain) | MPJPE on each domain's val split at view count k |
| `MPJPE@1` (3DPW actual) | Real monocular in-the-wild benchmark |
| `domain_gap` | Max difference in MPJPE across domains |
| `domain_discriminator_acc` | Should stay near chance (0.5) if features are domain-invariant |

Expected target: v48 reduces the 3DPW↔studio gap by ≥20% relative to v47 without regressing H36M/MPI/AIST full-view accuracy.

## Experiments

| Stage | Hardware | Config | Goal |
|-------|----------|--------|------|
| Smoke | RTX 4090 | `configs/benchmark_v48_domain_generalization_smoke.yaml` | val_MPJPE finite on all domains; no NaN/OOM |
| Full | A800-D | v47 checkpoint + v48 head | Reduce cross-domain gap; 3DPW actual MPJPE@1 reported |
| Ablation | RTX 4090 | no GRL / no FiLM / no DDWL / fixed dropout | Identify necessary components |

## Success criteria

1. Smoke test passes with no NaN/OOM and finite per-domain val_MPJPE.
2. 3DPW pseudo val_MPJPE is within 1.5× of MPI/H36M val_MPJPE.
3. 3DPW actual `MPJPE@1` is finite and improves over a v47 baseline run on actual mode.
4. No regression on H36M/MPI/AIST full-view MPJPE@k versus v47.
5. Domain discriminator accuracy stays within `[0.45, 0.55]` after convergence.

## Paper story fit

v48 supports the claim: *Our model generalizes across camera rigs, frame rates, and environments, from controlled multi-view studios to unconstrained in-the-wild video.* Combining sparse-view training, temporal aggregation, and domain-invariant learning produces a practical pose estimator for real-world deployment.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| 3DPW pseudo labels are synthetic | Use `actual` mode for real evaluation; keep pseudo as auxiliary training only. |
| GRL destabilizes training | Start with `grl_lambda=0.01`; freeze backbone for first epoch. |
| 3DPW dominates/under-represented | Combine DDWL with domain-balanced sampling. |
| Temporal head over-smoothes 3DPW motion | Use domain-conditional temporal window (shorter for 3DPW). |

## Relation to other variants

- **v41 DDWL:** v48 implements the DDWL loss from the v41 redesign and integrates it into the v47 trainer.
- **v33 domain view curriculum:** v48 reuses the idea of domain-conditional view selection but applies it inside the v46 dropout/augmentation path.
- **v46/v47:** v48 is a thin layer on top; it does not replace the sparse-view or temporal modules.

## Next steps

1. Wait for v47-temporal smoke results (#162).
2. Implement 3DPW `actual`-mode loader and eval benchmark.
3. Add `DomainInvariantSparseViewV48` and DDWL to the v47 trainer.
4. Smoke on RTX 4090 and compare per-domain MPJPE@k with v47.
5. Queue full A800 run starting from the best v47 checkpoint.
