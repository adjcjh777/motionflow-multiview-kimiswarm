# v48: Domain Generalization and 3DPW Integration

**Status:** Implementation in progress  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #164 (depends on v47-temporal #162)

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

---

# User Guide: Enabling v48 Domain Generalization and 3DPW Actual-Mode Evaluation

## Quick Start

1. **Ensure you are on the `v48-domain` branch and that v46-SVG and v47-temporal are already enabled.**
2. **Run the smoke test locally once the v48 code is wired:**
   ```bash
   bash scripts/run_v48_domain_smoke_local_4090.sh
   ```
3. **To enable in a custom run, add to your YAML config or CLI:**
   ```yaml
   model:
     use_v48_domain_generalization: true
     v48_dg_hidden: 64
     v48_dg_grl_lambda: 0.1
     v48_dg_use_domain_film: true
     v48_dg_use_ddwl: true
     v48_dg_ddwl_temperature: 2.0
     v48_dg_ddwl_warmup_epochs: 1
     v48_3dpw_actual_val_paths: null
     v48_dropout_per_domain:
       "0": 0.30
       "1": 0.30
       "5": 0.15
   ```

## Enabling in Training

### YAML Configuration

A minimal v48-enabled YAML snippet builds on the v47 smoke config and adds the domain-generalization block:

```yaml
model:
  # v46 / v47 base flags are required and unchanged.
  use_v46_sparse_view_generalization: true
  use_v47_temporal_aggregation: true

  # v48 domain generalization
  use_v48_domain_generalization: true
  v48_dg_hidden: 64
  v48_dg_grl_lambda: 0.1
  v48_dg_use_domain_film: true
  v48_dg_use_ddwl: true
  v48_dg_ddwl_temperature: 2.0
  v48_dg_ddwl_warmup_epochs: 1

  # Optional: per-domain view dropout schedule.
  v48_dropout_per_domain:
    "0": 0.30   # H36M / MPI-style studio multi-view
    "1": 0.30   # MPI-INF-3DHP
    "5": 0.15   # 3DPW in-the-wild (gentler dropout)

  # Optional: list of 3DPW `actual` .npz files for real moving-camera validation.
  v48_3dpw_actual_val_paths: null

training:
  # Existing v46/v47 training settings remain unchanged.
  # Typical warm-start recipe: freeze v25/v45/v46/v47 for 1 epoch,
  # then unfreeze and continue with the mixed manifest.
```

### CLI Override

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --config configs/benchmark_v48_domain_smoke.yaml \
  --use_v48_domain_generalization \
  --v48_dg_hidden 64 \
  --v48_dg_grl_lambda 0.1 \
  --v48_dg_use_domain_film \
  --v48_dg_use_ddwl \
  --v48_dg_ddwl_temperature 2.0 \
  --v48_dg_ddwl_warmup_epochs 1
```

## Running Evaluation

### Per-dataset MPJPE on standard val splits

After training, evaluate the checkpoint on each domain separately:

```bash
python experiments/eval_variable_views.py \
  --checkpoint outputs/omniview_fusion_v48_domain_smoke_local_4090.pth \
  --config configs/benchmark_v48_domain_smoke.yaml \
  --view_subsets 1,2,3,4,full \
  --per_domain \
  --out outputs/v48_domain_eval.json
```

The output JSON contains `MPJPE@k` per domain plus the cross-domain gap.

### 3DPW `actual`-mode monocular benchmark

To enable the real moving-camera benchmark, point `v48_3dpw_actual_val_paths` at the 3DPW `actual` validation `.npz` files and run:

```bash
python experiments/eval_variable_views.py \
  --checkpoint outputs/omniview_fusion_v48_domain_smoke_local_4090.pth \
  --config configs/benchmark_v48_domain_smoke.yaml \
  --view_subsets 1 \
  --eval_3dpw_actual \
  --out outputs/v48_3dpw_actual_eval.json
```

`MPJPE@1` on the 3DPW `actual` split is the primary in-the-wild metric.

## Interpreting Results

- **`MPJPE@full` (studio domains)**: Should be within ~1 mm of the v47 baseline at full views.
- **`MPJPE@1` (3DPW actual)**: Should be finite and lower than a v47 baseline run on the same actual-mode data.
- **`domain_gap`**: The maximum difference in `MPJPE@full` across the studio domains and 3DPW pseudo. v48 targets a ≥20% reduction versus v47.
- **`domain_discriminator_acc`**: Should stay near chance (`[0.45, 0.55]`) after convergence, confirming that the adapter is producing domain-invariant features.
- **`MPJPE@2` / `MPJPE@3` (studio)**: Should remain comparable to v47; the goal is no regression at sparse views while improving cross-domain transfer.

## When to Use v48

Use v48 when:

- You need a single checkpoint that works across both studio multi-view (H36M, MPI-INF-3DHP, AIST++) and in-the-wild monocular (3DPW) data.
- You have 3DPW data available in either `pseudo` (synthetic 4-view) or `actual` (real moving-camera) mode.
- You already have a strong v46/v47 checkpoint and want to add domain-invariant refinement without changing the backbone.

Do not use v48 if:

- You only train and evaluate on a single studio domain (v46/v47 is sufficient).
- 3DPW data is not available and no cross-domain generalization is needed.
- You cannot afford the extra memory / compute from the gradient-reversal and DDWL losses.

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NaN` during training | `v48_dg_grl_lambda` too high, or DDWL weights exploded. | Lower `v48_dg_grl_lambda` to `0.01`, increase `v48_dg_ddwl_warmup_epochs`, or reduce the learning rate. |
| 3DPW `actual` eval fails | `v48_3dpw_actual_val_paths` is `null` or points at `pseudo` files. | Check that the paths are real 3DPW `actual` `.npz` files with `V=1` and per-frame camera poses. |
| No cross-domain improvement | Domain labels are missing or all samples map to the same domain id. | Verify the manifest provides `dataset_id` (0=H36M, 1=MPI, 5=3DPW, ...). |
| Over-smoothing on 3DPW fast motion | v47 temporal window too wide for in-the-wild framerates. | Set `v47_temporal_window: 7` or use a domain-conditional window once implemented. |
| v48 flags ignored | `use_v48_domain_generalization` is true but v46/v47 are not enabled. | Ensure both `use_v46_sparse_view_generalization` and `use_v47_temporal_aggregation` are set. |

## See Also

- Issue #164 — v48 tracking
- Issue #162 — v47-temporal dependency
- Issue #160 — v46-SVG dependency
- `docs/swarm_iter25_action_plan.md` — full agent task list
- `motionflow_mv/fusion/domain_adapter_v48.py` — domain adapter module
- `motionflow_mv/fusion/temporal_aggregation_v47.py` — v47 base module
- `motionflow_mv/data/webbridge_mixed_dataset.py` — mixed loader with 3DPW support
