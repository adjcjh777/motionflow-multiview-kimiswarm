# v49: WebBridge Multi-View Data Pipeline

**Status:** Proposal / ready for design review  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #166 (proposed)  
**Depends on:** v46-SVG (#160), v47-temporal (#162), v48-domain (#164)

---

## 1. Problem statement

The WebBridge mixed-data loader (`motionflow_mv/data/webbridge_mixed_dataset.py`) is **static**: it reads a fixed YAML manifest, samples clips uniformly, and applies fixed augmentation. After v46–v48 we have:

- v46 sparse-view reliability and on-the-fly view dropout.
- v47 temporal aggregation across clips.
- v48 domain-invariant refinement and per-domain view dropout.

But the **data pipeline itself does not close the feedback loop**. Hard frames (occlusions, calibration drift, MPI 28→17 mapping errors, noisy 3DPW pseudo-labels) are sampled at the same rate as easy frames, and domain imbalance is left to the loss or sampler to fix. We waste training budget on already-solved examples and under-sample the failure modes that actually limit generalisation.

v49 makes the WebBridge multi-view data pipeline **self-evolving**: it consumes per-sample reprojection / reliability / uncertainty feedback from the model and uses it to reweight, filter, and augment training clips.

---

## 2. Proposed approach

v49 is a thin, optional wrapper around the existing `WebBridgeMixedDataset`. It reuses the canonical `(T, V, J)` loader and adds three minimal components:

```text
WebBridgeCanonical17Dataset
        |
        v
[SelfEvolvingClipSamplerV49]  <- difficulty scores from model feedback
        |
        v
[DataQualityGateV49]          <- flags/quarantines noisy pseudo-labels
        |
        v
[DomainAdaptiveAugmentationV49] <- per-domain view-dropout / noise schedule
        |
        v
batch -> v46/v47/v48 model
        |
        v
reprojection residual + v46 reliability  ->  update difficulty scores
```

### How it fits the v46–v48 stack and the overall pipeline

- **v46 sparse-view generalisation:** The v46 reliability head already predicts per-view-joint reliability. v49 uses this reliability to score clip difficulty and to drive a domain-aware view-dropout schedule.
- **v47 temporal aggregation:** Hard temporal clips identified by v49 become the training material for the v47 temporal head, preventing it from only seeing easy, static poses.
- **v48 domain generalisation:** v49’s per-domain difficulty scores naturally balance the mixed manifest, complementing v48’s DDWL / FiLM by ensuring each domain is sampled according to current model error rather than file count.
- **Overall pipeline:** v49 closes the **self-evolution loop**: model → reprojection/residual → data sampler → harder training distribution → better model. It is the data-side counterpart to v37 self-critique view reliability and v45 adaptive geometry fusion.

---

## 3. Concrete code-level changes

### New files

| Path | Purpose |
|------|---------|
| `motionflow_mv/data/webbridge_self_evolving_v49.py` | Core v49 data-pipeline classes. |
| `tests/test_webbridge_self_evolving_v49.py` | Unit tests for sampler, gate, and augmentation scheduler. |
| `configs/benchmark_v49_webbridge_data_pipeline_smoke.yaml` | RTX 4090 smoke config. |
| `scripts/run_v49_webbridge_data_pipeline_smoke_local_4090.sh` | Local smoke script. |

### `motionflow_mv/data/webbridge_self_evolving_v49.py`

```python
class SelfEvolvingClipSamplerV49:
    """Maintains an EMA difficulty score per clip and samples accordingly."""

    def __init__(self, clip_keys, temperature: float = 1.0,
                 hard_sample_fraction: float = 0.2,
                 per_domain_hard_cap: float = 0.4):
        ...

    def update(self, errors: dict[str, float]) -> None:
        """Update EMA scores with per-clip MPJPE / reprojection error."""

    def sample_weights(self, dataset_ids: torch.Tensor) -> torch.Tensor:
        """Return sampling probabilities with per-domain hard caps."""


class DataQualityGateV49:
    """Flag or quarantine clips whose pseudo-labels / captures are unreliable."""

    def __call__(self, reprojection_error, view_reliability) -> dict[str, float]:
        ...


class DomainAdaptiveAugmentationV49:
    """Wraps v46 view dropout and outlier augmentation with per-domain schedules."""

    def __init__(self, dropout_per_domain: dict[int, float],
                 update_interval: int = 1):
        ...

    def step(self, domain_errors: dict[int, float]) -> None:
        """Increase augmentation for domains with high current error."""


class WebBridgeSelfEvolvingDatasetV49(Dataset):
    """Wrapper that combines the above and exposes a state_dict for checkpointing."""

    def __init__(self, base_dataset, sampler, gate, augmenter):
        ...

    def update_feedback(self, batch_clip_keys, pred_3d, target_3d,
                        cameras, view_mask) -> None:
        ...
```

### Files to modify

| File | Change |
|------|--------|
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Add v49 CLI flags; build `WebBridgeSelfEvolvingDatasetV49` when enabled; after each validation epoch compute per-clip feedback and call `update_feedback(...)`; save/load sampler state with checkpoint. |
| `motionflow_mv/data/view_dropout_augmentation_v46.py` | No breaking changes; reuse via `DomainAdaptiveAugmentationV49`. |

### New training flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v49_webbridge_self_evolving_pipeline` | bool | `False` | Master switch. |
| `v49_data_reweight_temperature` | float | `1.0` | Temperature for converting error to sampling weight. |
| `v49_data_update_interval_epochs` | int | `1` | Re-compute clip difficulty every N epochs. |
| `v49_data_hard_sample_fraction` | float | `0.2` | Top fraction of clips by error to up-sample. |
| `v49_data_quality_threshold_mm` | float | `100.0` | MPJPE threshold above which a clip is flagged for quarantine. |
| `v49_data_use_reprojection_feedback` | bool | `True` | Use reprojection residual as a difficulty signal. |
| `v49_data_use_view_reliability_feedback` | bool | `True` | Use v46/v37 reliability as a signal. |
| `v49_data_per_domain_hard_cap` | float | `0.4` | Max fraction of hard samples allowed from one domain. |

---

## 4. Risks / failure modes

| Risk | Why it happens | Mitigation |
|------|----------------|------------|
| **Overfits to label noise** | Hard-sample mining samples noisy 3DPW / mapping-error clips. | `DataQualityGateV49` quarantines outliers; cap hard-sample fraction per domain. |
| **Domain imbalance amplified** | One domain (e.g. MPI) is always hardest and dominates. | Per-domain hard cap + keep a minimum uniform-sampling baseline (`1 - v49_data_hard_sample_fraction`). |
| **Stale feedback** | EMA lags behind sudden model improvement. | Update every epoch; use low `v49_data_update_interval_epochs`. |
| **3DPW pseudo-label corruption** | 3DPW pseudo-multi-view labels are synthetic and noisy. | Exclude 3DPW from score updates by default, or use a higher `v49_data_quality_threshold_mm`. |
| **CPU / memory overhead** | Maintaining per-clip scores for a large manifest. | Store scores as a dict keyed by `(file, start_frame)`; only hold scores for the current manifest. |

---

## 5. Success metrics and recommended smoke/full experiment

### Smoke test (RTX 4090)

- **Config:** `configs/benchmark_v49_webbridge_data_pipeline_smoke.yaml` (d=64, 2 epochs, 500 samples, clip_len 9, batch_size 4).
- **Command:** `bash scripts/run_v49_webbridge_data_pipeline_smoke_local_4090.sh`
- **Pass criteria:**
  1. No NaN/OOM.
  2. Sampler scores update after the first validation epoch.
  3. `val_MPJPE` is finite and within 5% of the v48 smoke baseline.
  4. At least one clip is flagged and re-weighted.

### Full experiment (A800-D)

- **Config:** `configs/benchmark_v49_webbridge_data_pipeline_full.yaml` on top of the best v48 checkpoint.
- **Hardware:** A800-D, d=128, n_st_layers=3, batch_size=16, clip_len 13, full `configs/splits/webbridge_all_train_mixed.yaml`.
- **Target:**
  - No regression in full-view `val_MPJPE` vs. v48.
  - `MPJPE@2/3/4` improves by ≥3% relative to v48 on the same manifest.
  - Per-domain gap (max − min domain `val_MPJPE`) reduces by ≥5%.
  - Hard-clip reprojection error reduces by ≥10%.

### Key metrics to report

- `val_MPJPE` and `MPJPE@k` for `k = 2, 3, 4, full`.
- Per-domain `val_MPJPE` and `domain_gap`.
- Fraction of clips gated / reweighted per epoch.
- Mean reprojection error on the hard-sample subset.
- Per-joint error on hips, hands, and wrists (historical failure joints).

---

## 6. Self-evolution feedback loop

v49 explicitly closes the data-side self-evolution loop:

1. **Forward pass:** The v46/v47/v48 model predicts 3D pose and per-view reliability `r`.
2. **Residual computation:** Reproject the 3D pose back to each view to get per-view-joint reprojection error `e`. Combine with `r` and v37 self-critique scores.
3. **Score update:** `DataQualityGateV49` and `SelfEvolvingClipSamplerV49` turn `e` and `r` into a per-clip difficulty score, updated via EMA after every validation epoch.
4. **Distribution shift:** The next epoch samples hard-but-trustworthy clips more often and suppresses noisy/corrupt clips.
5. **Repeat:** The improved model produces better `r` and `e`, refining the data distribution.

This is the data-pipeline analogue of v37 self-critique view reliability and v45 adaptive geometry fusion: instead of only adapting network weights, the training corpus itself adapts to the model’s current weaknesses.

---

## 7. Next steps

1. Wait for v48-domain smoke results (#164) to land.
2. Implement `motionflow_mv/data/webbridge_self_evolving_v49.py` and unit tests.
3. Wire v49 flags into `experiments/train_omniview_fusion_v5_webbridge_multi.py`.
4. Add smoke config/script and run on RTX 4090.
5. If smoke passes, queue a full A800 run on top of the best v48 checkpoint and compare against the v48 baseline.

---

## See also

- `docs/proposals/v46_sparse_view_generalization.md`
- `docs/proposals/v47_combined_architecture.md`
- `docs/proposals/v48_domain_generalization.md`
- `docs/proposals/v37_self_critique_view_reliability.md`
- `motionflow_mv/data/webbridge_mixed_dataset.py`
- `motionflow_mv/data/view_dropout_augmentation_v46.py`
