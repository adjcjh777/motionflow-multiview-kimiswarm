# v49: Multi-View Camera Embedding

**Status:** Design note / not yet implemented  
**Labels:** `experiment`, `P2-nice`  
**Tracking issue:** #166 (proposed)  
**Depends on:** v31 camera-view embedding, v37 self-critique view reliability, v45 adaptive geometry fusion, v46 sparse-view generalization (conceptually; v49 builds on top)

---

## 1. Problem statement

`OmniMultiViewFusionV5` already has a camera-conditioned view embedding, but it reasons about cameras one view at a time:

* `CameraConditionedViewEmbedding` flattens `(K, R, t)` into a 21-D vector and feeds each view through a small MLP. It is permutation-invariant but has **no notion of the multi-view rig topology**.
* `CameraConditionedViewEmbeddingV31` added local descriptors plus a pairwise self-attention branch, which is a step toward rig-level reasoning, but the pairwise features are computed once and never updated by the model's own feedback.
* v33 proposed a ray-aware per-joint extension, yet the camera embedding still does not form an explicit, learnable **multi-view camera rig representation** that can be refined during training or used by downstream sparse-view / temporal / domain modules.

As a result, when views are dropped (v46), when temporal aggregation runs (v47), or when domains change (v48), the model has no compact latent description of the whole camera rig to tell it, for example, "cameras 1 and 3 are close together and see a joint from a similar angle, so their disagreement is more informative than the disagreement of two distant cameras." v49 closes that gap with a small, explicit **multi-view camera embedding** module.

---

## 2. Proposed approach

Add `MultiViewCameraEmbeddingV49` as a drop-in replacement/augmentation for the existing per-view camera embedding. It encodes the camera **rig** as a small graph rather than a bag of views, and it can be gated by the model's own reliability/uncertainty feedback (v37, v39, v46, v45).

```text
Input: K, R, t, optional view_mask, optional per-view reliability r_v
        |
        ▼
[Local camera descriptor]  ──► per-view tokens  (B, V, d_local)
        |
        ▼
[Pairwise rig graph]  ──► pairwise edge tokens  (B, V, V, d_pair)
        |
        ▼
[Cross-view message passing]  ──► multi-view rig tokens per view  (B, V, d)
        |
        ├──► used as camera-conditioned view embedding in v5 encoder
        └──► (optional) global rig token for v46/v47/v48 heads
```

**Key properties**

* **Permutation equivariant:** reordering the input views reorders the output tokens.
* **Variable-view:** respects `view_mask` so dropped/occluded views do not participate in message passing.
* **Identity at init:** final projection is zero-initialized, so warm-starting from a v31/v45/v46 checkpoint is safe.
* **Self-evolution aware:** optional `reliability` input (from v37/v39/v46) gates the message passing, so the rig representation is updated by the model's own confidence estimates.

**How it fits**

* **v46 sparse-view generalization:** the embedding sees `view_mask` and produces meaningful tokens even when only 2–3 views remain, because pairwise rig geometry still constrains the available views.
* **v47 temporal aggregation:** the per-frame temporal head can reuse the same per-view rig tokens, optionally adding a global rig token to its input so it knows the rig layout across frames.
* **v48 domain generalization:** camera parameters replace dataset-specific learned view embeddings, making the model easier to transfer across rigs; the domain adapter can also condition on the global rig token.
* **Overall pipeline:** the camera embedding is the first step after input, so a better rig-level representation propagates through geometry fusion, triangulation, and refinement.

---

## 3. Concrete code-level changes

### 3.1 New module

`motionflow_mv/fusion/multi_view_camera_embedding_v49.py`:

```python
class MultiViewCameraEmbeddingV49(nn.Module):
    def __init__(
        self,
        d: int,
        local_hidden: int = 32,
        pairwise_hidden: int = 32,
        rig_hidden: int = 64,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_reliability_feedback: bool = False,
    ):
        ...

    def forward(
        self,
        K: torch.Tensor,          # (B, V, 3, 3)
        R: torch.Tensor,          # (B, V, 3, 3)
        t: torch.Tensor,          # (B, V, 3)
        view_mask: torch.Tensor | None = None,  # (B, V)
        reliability: torch.Tensor | None = None,  # (B, V)
    ) -> torch.Tensor:
        """Return multi-view rig tokens (B, V, d)."""
        ...
```

The implementation reuses the local-descriptor and pairwise-descriptor ideas from `camera_view_embedding_v31.py` and adds:

1. A **rig-level self-attention** layer over views (batch-first MHA) with `key_padding_mask` built from `view_mask`.
2. An optional **reliability gate** that multiplies messages by per-view reliability before aggregation.
3. A **global rig token** branch (mean of rig tokens) returned as an optional second output for downstream heads.

### 3.2 Model wiring

In `motionflow_mv/fusion/omniview_fusion_v5.py`:

```python
# Existing
self.use_camera_view_embedding_v31 = use_camera_view_embedding_v31
self.use_multi_view_camera_embedding_v49 = use_multi_view_camera_embedding_v49

# Instantiate
if self.use_multi_view_camera_embedding_v49:
    from motionflow_mv.fusion.multi_view_camera_embedding_v49 import MultiViewCameraEmbeddingV49
    self.multi_view_camera_embedding_v49 = MultiViewCameraEmbeddingV49(
        d=d,
        local_hidden=camera_view_embedding_hidden,
        rig_hidden=v49_mvce_rig_hidden,
        n_heads=v49_mvce_n_heads,
        dropout=v49_mvce_dropout,
        use_reliability_feedback=v49_mvce_use_reliability_feedback,
    )
else:
    self.multi_view_camera_embedding_v49 = None
```

Forward pass:

```python
if self.use_multi_view_camera_embedding_v49:
    # reliability from v37/v39/v46 if available, else None
    view_emb = self.multi_view_camera_embedding_v49(K, R, t, view_mask, reliability)
else:
    view_emb = ...  # existing v31 or base embedding path
```

### 3.3 New training flags

Add in `experiments/train_omniview_fusion_v5_webbridge_multi.py` and expose via YAML/CLI:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_multi_view_camera_embedding_v49` | bool | `False` | Master switch |
| `v49_mvce_rig_hidden` | int | `64` | Hidden dim of the rig graph branch |
| `v49_mvce_n_heads` | int | `4` | Heads in cross-view rig attention |
| `v49_mvce_dropout` | float | `0.0` | Dropout in rig graph layers |
| `v49_mvce_use_reliability_feedback` | bool | `False` | Gate rig messages with v37/v46 reliability |
| `v49_mvce_loss_weight` | float | `0.0` | Optional auxiliary loss on rig-token consistency (see self-evolution loop) |

### 3.4 Smoke / test harness

Create:

* `configs/benchmark_v49_multi_view_camera_embedding_smoke.yaml`
* `scripts/run_v49_multi_view_camera_embedding_smoke_local_4090.sh`
* `tests/test_multi_view_camera_embedding_v49.py` — check:
  * Output shape `(B, V, d)` for variable `V`
  * Permutation equivariance
  * `view_mask` zeroes out masked views
  * Gradient flow with optional `reliability` input

---

## 4. Risks / failure modes

| Risk | Mitigation |
|------|------------|
| Pairwise rig attention is `O(V²)` and may OOM with `V=14` at A800 batch sizes | Keep `v49_mvce_rig_hidden` small (≤64); use view masking to skip padded views; fallback to local-only branch if OOM |
| Overfitting to a specific rig's pairwise geometry | Zero-initialize final projection; reuse camera perturbation curriculum; validate on a different dataset's rig without retraining |
| Amplifies calibration noise because baseline/angle features are explicit | Normalize by rig diameter; rely on the model's reliability gate to down-weight noisy pairwise edges |
| Marginal gain if v31 pairwise attention already captures rig geometry | Direct ablation against `use_camera_view_embedding_v31` on the same smoke config; if gain < 0.5 mm, document as negative ablation |
| Reliability feedback can destabilize early training | Disable `v49_mvce_use_reliability_feedback` for the first epoch; warm-start reliability head from a v37/v46 checkpoint |
| Breaks warm-start from v31/v45/v46 checkpoints | Keep final projection zero-initialized; only load the new v49 parameters when warm-starting with `strict=False` |

---

## 5. Success metrics and recommended experiments

### Smoke (RTX 4090)

* **Config:** `configs/benchmark_v49_multi_view_camera_embedding_smoke.yaml`
* **Recipe:** `d=64`, `clip_len=9`, `train_samples=500`, 5 epochs, mixed H36M/MPI manifest, camera perturbation curriculum enabled.
* **Expected outcome:**
  * `val_MPJPE` finite and `< 80 mm` (matching v46/v48 smoke thresholds)
  * No NaN/OOM
  * `tests/test_multi_view_camera_embedding_v49.py` passes
  * Permutation-invariance error `< 1e-4`

### Full (A800-D)

* **Config:** `configs/benchmark_v49_multi_view_camera_embedding_full.yaml` (to be created after smoke)
* **Recipe:** `d=128`, full WebBridge/H36M/MPI mixed manifest, warm-start from best v48 checkpoint, train 5+ epochs.
* **Expected outcome:**
  * Full-view `val_MPJPE` within `1 mm` of v48 baseline
  * `MPJPE@2` and `MPJPE@3` improved by `≥5%` over v48 (or v46 if v48 not yet ready)
  * Cross-dataset transfer (H36M → MPI or MPI → H36M) shows no regression

### Metrics

| Metric | Target |
|--------|--------|
| `val_MPJPE` (smoke) | `< 80 mm`, finite |
| `MPJPE@2/3/4` vs v46/v48 | `≥5%` relative improvement at sparse views |
| `MPJPE@full` vs v48 | No regression (`Δ < 1 mm`) |
| `reprojection_error` | Not degraded |
| `permutation_invariance_error` | `< 1e-4` |
| `cross_dataset_MPJPE` | Within `2 mm` of within-dataset full-view result |

---

## 6. Self-evolution feedback loop

The v49 embedding is a natural place to close the **self-evolution** loop that runs through v37, v39, v45, and v46:

```text
multi-view camera embedding  ──► geometry fusion / triangulation
         ▲                            |
         │                            ▼
         └──────── reliability  ←  reprojection residuals / uncertainty
```

* **Input:** v37 `SelfCritiqueViewReliabilityV37`, v39 reliability-coupled graph gates, or v46 sparse-view reliability head already produce per-view reliability scores `r_v`.
* **Update:** when `v49_mvce_use_reliability_feedback=True`, these scores scale the messages in the rig-graph self-attention. A view the model currently trusts receives stronger outgoing edges; an outlier view is effectively masked.
* **Feedback:** the refined multi-view camera embedding then feeds into the same triangulation/attention path, changing the 3D pose, which changes reprojection residuals, which updates reliability for the next iteration/forward pass.
* **Optional explicit loop:** an auxiliary `v49_mvce_loss_weight > 0` can enforce that the global rig token is consistent before and after a small camera perturbation, nudging the embedding to be calibration-robust in the same way the physical losses nudge the skeleton to be plausible.

This keeps the design minimal: the core module is just a camera-embedding upgrade, but it is shaped so the existing self-evolution machinery can plug into it cleanly.

---

## 7. Relation to other variants

* **v31 camera-view embedding upgrade:** v49 is a direct follow-up; it keeps the local+pairwise idea but adds rig-level self-attention and reliability feedback.
* **v33 camera geometry embedding:** v33 focused on per-joint ray features and calibration-aware triangulation. v49 stays at the **camera-level** (not per-joint) to keep cost low; the two can be combined later.
* **v45 adaptive geometry fusion:** v45 predicts per-view triangulation weights; its reliability output can be the `reliability` input to v49.
* **v46 sparse-view generalization:** v49's `view_mask` handling and variable-view support are prerequisites for v46; the rig graph should make sparse-view inference more robust.
* **v47/v48:** v49 is upstream of these heads and should not conflict with them.

---

## 8. Next steps

1. Wait for the v44 architecture decision and the A800 priority queue results (per `AGENTS.md`).
2. Implement `MultiViewCameraEmbeddingV49` and the unit test.
3. Wire the v49 flag into `OmniMultiViewFusionV5` and `train_omniview_fusion_v5_webbridge_multi.py`.
4. Run the RTX 4090 smoke and compare directly against `use_camera_view_embedding_v31`.
5. If smoke meets targets, create the A800 full config and queue the run behind the v46–v48 priority queue.
6. Document whether the reliability-feedback path (`v49_mvce_use_reliability_feedback`) is beneficial in an ablation.
