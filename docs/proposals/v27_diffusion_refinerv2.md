# v27: Diffusion Pose Refiner v2 (`diffusion_refinerv2`)

**Task identifier:** `design_v27_diffusion_refinerv2`  
**Status:** Design / candidate v27 direction  
**Depends on:** v20 (`docs/proposals/v20_diffusion_refinement.md`), v25 (`docs/proposals/v25_multiview_geometry_fusion.md`), v26 (`docs/proposals/v26_temporal_geometry_fusion.md`)  

## 1. Problem

v25/v26 replace the old triangulation core with a geometry-first pipeline (ray tokens, geometry-aware attention, learned depth proposals). However, the final output is still a **single point estimate**: the refined 3D pose after Gauss-Newton / geometry fusion. In two failure modes the deterministic output is sub-optimal:

1. **Multi-modal residual errors** from few/occluded views. When only 2–4 views are visible or a joint is occluded, the geometry fusion output is close to the true pose but has structured, multi-modal residual error that a small deterministic MLP (or even v20, which sees only pooled features) cannot fully resolve.
2. **Under-use of geometry features by the refiner.** v20 (`motionflow_mv/fusion/diffusion_pose_refiner_v20.py:81-124`) only receives `pose_init` and the pooled per-view feature `feat_pooled`. It never sees the rich geometry-aware features that v25/v26 already computed (ray-intersection quality, depth-proposal scores, per-view visibility).
3. **No temporal consistency inside the refiner.** v20 processes each frame independently, even though v26 already established temporal ray correspondences.

`diffusion_refinerv2` addresses this by turning the final residual correction into a **geometry- and temporally-conditioned diffusion process** that reasons over the uncertainty left after v25/v26.

## 2. Proposed method

### 2.1 New module

**File:** `motionflow_mv/fusion/diffusion_pose_refiner_v27.py`

```text
DiffusionPoseRefinerV27(
    j: int = 17,
    in_dim: int = 64,                # dimension of feat_pooled
    geom_dim: int = 64,              # dimension of geometry features from v25/v26
    residual_hidden: int = 128,
    num_diffusion_steps: int = 100,
    num_inference_steps: int = 5,
    temporal_window: int = 3,        # must be odd, default 3 (±1 frames)
    n_heads: int = 4,
    use_geometry_conditioning: bool = True,
    use_temporal_conditioning: bool = True,
    beta_schedule: str = "cosine",
)
```

The module is a **drop-in replacement** for `DiffusionPoseRefinerV20` at the current hook in `motionflow_mv/fusion/omniview_fusion_v5.py:805-821`.

### 2.2 Inputs / outputs

```python
pred_3d_v27, diff_loss = self.diffusion_refiner_v27(
    pose_init=pred_3d_gn,                       # (B, T, J, 3) from v26/v25
    feat=feat_pooled,                           # (B*T, d) pooled per-view features
    geom_feat=geom_feat,                        # (B, T, J, geom_dim) OPTIONAL
    visibility=visibility,                      # (B, T, V, J) OPTIONAL
)
```

* `pred_3d_v27`: `(B, T, J, 3)` refined 3D pose.
* `diff_loss`: scalar diffusion loss (added to the total training objective).

### 2.3 What changes from v20

| Aspect | v20 (`diffusion_pose_refiner_v20.py`) | v27 (`diffusion_pose_refiner_v27.py`) |
|--------|----------------------------------------|----------------------------------------|
| Conditioning | `noisy_residual + pose_init + feat_pooled` | Adds `geom_feat` from v25/v26 ray/depth attention |
| Architecture | 2-layer per-frame joint transformer | 2-layer **spatio-temporal** transformer (joint × temporal window) |
| Geometry awareness | None | Scales joint-level conditioning by per-joint visibility / ray-intersection confidence |
| Temporal handling | None | Self-attention over `[-(W-1)/2, ..., +(W-1)/2]` frames |
| Warm start | v20 output ≈ pose_init at init if last layer zeroed | Same: final projection initialised to zero, so v27 starts as identity |

### 2.4 Geometry feature source

To avoid changing v25/v26 internals, `geom_feat` is produced by a tiny **on-the-fly** extractor inside `DiffusionPoseRefinerV27`:

```python
GeometryFeatureExtractor(
    in_dim: int = 64,
    out_dim: int = 64,
)
```

It recomputes, per joint and frame:
* the per-view ray-intersection score already used by v25 (`ray_intersection_logit` in `multiview_geometry_fusion_v25.py`),
* the depth-proposal softmax entropy from the v25 `DepthProposalTriangulation` head,
* the per-joint visibility `visibility[:, :, v, j]` averaged over views.

These three scalar fields are projected to `geom_dim` and fed into the denoiser. Because the extractor is tiny and operates on the same camera/keypoint tensors already available in `OmniMultiViewFusionV5.forward`, it does not require v25/v26 to expose new intermediate outputs.

### 2.5 Spatio-temporal denoiser

Inside `DiffusionPoseRefinerV27`:

1. **Flatten/reshape** `(B, T, J, 3)` → `(B*T, J, 3)`.
2. **Conditioning projection** concatenates `noisy_residual`, `pose_init`, `feat_pooled`, and `geom_feat`, then projects to `residual_hidden`.
3. **Time embedding** is added as a broadcast bias (same as v20).
4. **Spatio-temporal attention blocks** (2 layers) alternate:
   * joint-level self-attention (anatomical reasoning), and
   * temporal self-attention over `temporal_window` neighbours (motion smoothness).
5. **Output head** predicts the noise (residual correction), initialised to zero.

### 2.6 Integration point

Replace the block at `motionflow_mv/fusion/omniview_fusion_v5.py:805-821` with:

```python
# v27 geometry-aware temporal diffusion refiner.
if self.use_diffusion_refiner_v27 and self.diffusion_refiner_v27 is not None:
    pred_3d, diff_loss = self.diffusion_refiner_v27(
        pose_init=pred_3d_gn.view(B, T, J, 3),
        feat=feat_pooled,                       # (B*T, d)
        visibility=visibility.view(B, T, V, J),
    )
    pred_3d = pred_3d.view(B * T, J, 3)
    epi_loss = epi_loss + diff_loss
else:
    # existing deterministic MLP or v20 path
    ...
```

A new flag is added to `OmniMultiViewFusionV5.__init__` alongside the existing v20 flag (`omniview_fusion_v5.py:122`):

```python
use_diffusion_refiner_v27: bool = False,
```

## 3. Expected impact

Assume the decision gates in `docs/proposals/v27_next_iteration_decision_matrix.md` hold and v25/v26 are within ~1 mm of the v18 baseline (20.24 mm).

| Metric | Expected change vs. v26 |
|--------|-------------------------|
| `val_MPJPE` | **-5 % to -8 %** (e.g. ~1.0–1.6 mm if baseline ~20 mm) |
| 2-view MPJPE | **-8 % to -12 %**; largest gain because few-view ambiguity is most multi-modal |
| 4-view MPJPE | **-6 % to -9 %** |
| 8-view MPJPE | **-4 % to -6 %** |
| 14-view MPJPE | **-2 % to -4 %**; near-saturated, geometry already strong |

**Why this is plausible:** v20 already showed that diffusion can model residual corrections; adding geometry and temporal conditioning should let the refiner focus on the residual subspace where v25/v26 are uncertain, rather than re-learning geometry from scratch.

**Caveat:** This alone is unlikely to satisfy the v27 success criterion of a 10 % drop over the best v25/v26 baseline. It should be combined with the smaller **uncertainty-aware depth-proposal** direction (`v27_next_iteration_decision_matrix.md` §2.1) if G1/G4 hold.

## 4. Implementation cost

| Item | Estimate |
|------|----------|
| New code | `motionflow_mv/fusion/diffusion_pose_refiner_v27.py` (~300–350 lines) |
| Tests | `tests/test_diffusion_pose_refiner_v27.py` (~120 lines) |
| Plumbing | One flag in `OmniMultiViewFusionV5.__init__` and one block in `forward` (~10 lines) |
| Parameters | ~0.5–1.0 M extra vs. v20 (small spatio-temporal transformer) |
| Training time | **+15 % to +25 %** per epoch due to diffusion loss + temporal attention |
| Memory | **+10 % to +15 %** from temporal window attention logits |
| Data | None beyond current WebBridge/H36M/MPI mix |

## 5. Risks / mitigation

| Risk | How to detect | Mitigation |
|------|---------------|------------|
| Diffusion loss destabilises early training | `diff_loss` >> `mpjpe_loss` in first 500 steps | Clamp predicted residual to `[-1, 1]` m and initialise final layer to zero; start with `diff_loss_weight=0.1` |
| Temporal window over-smooths fast motion | Qualitative failure on fast gestures; `val_MPJPE` rises at high-speed clips | Default `temporal_window=3`; keep first experiment per-frame and add temporal only if per-frame version works |
| Geometry feature extractor is noisy/irrelevant | Ablate `use_geometry_conditioning=False` and see no regression | Make extractor tiny; if ablation is neutral, drop it and keep only temporal diffusion |
| Inference cost too high for real-time eval | 5-step DDPM already 5× v20; benchmark with `num_inference_steps=1` as fallback | Provide deterministic single-step mode: set `num_inference_steps=1` to match MLP latency |
| Overfits to camera layout because of ray features | Cross-dataset gap (H36M → MPI) widens | Strong intrinsic/extrinsic augmentation already in v25; freeze extractor for first epoch if needed |

## 6. Minimal experiment plan

### 6.1 Flags / config names

Add to the YAML / argparser:

```yaml
model:
  use_diffusion_refiner_v27: true
  v27_diffusion_residual_hidden: 128
  v27_diffusion_num_inference_steps: 5
  v27_diffusion_temporal_window: 3
  v27_diffusion_use_geometry_conditioning: true
  v27_diffusion_use_temporal_conditioning: true
  v27_diffusion_loss_weight: 0.1
```

### 6.2 Smoke test

1. **Module-level smoke** (run locally on RTX 4090):

```bash
python motionflow_mv/fusion/diffusion_pose_refiner_v27.py
```

Expected: passes forward/backward checks for `(B=2, T=9, V=4, J=17)`.

2. **Unit test**:

```bash
pytest tests/test_diffusion_pose_refiner_v27.py -q
```

Tests should cover:
* Inference shape `(B, T, J, 3)`.
* Training returns `(refined, loss)`.
* Identity-at-init when final layer zeroed.
* Gradient flow through `pose_init`, `feat`, and `visibility`.
* Temporal boundary handling (`T < temporal_window`).
* Geometry-conditioned and geometry-free modes.

3. **Full-pipeline smoke** on a small WebBridge subset (≤1k steps):

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --config configs/train_ray_attention_reproducible.yaml \
  --use_temporal_geometry_fusion_v26 \
  --use_diffusion_refiner_v27 \
  --v27_diffusion_num_inference_steps 5 \
  --smoke_test \
  --max_steps 1000
```

4. **Decision gate check** before committing GPU time:
* Ensure v25 small `val_MPJPE` clears G1 (`< 20.24 mm`).
* Ensure v26 small `val_MPJPE` clears G4 (`< v25 small`).
* If either gate fails, do not launch v27 diffusion; revisit geometry core first.

## 7. Simpler variant (if too vague)

If extracting/maintaining geometry features proves brittle, fall back to **temporal-only v27**: keep the v20 architecture but add a `temporal_window` self-attention over the noisy residual and `pose_init` sequences, using only `feat_pooled` for conditioning. This is a ~50-line change on top of v20, costs much less engineering, and still captures temporal smoothness.

## 8. References

* `motionflow_mv/fusion/omniview_fusion_v5.py:805-821` — current v20 / MLP insertion point
* `motionflow_mv/fusion/omniview_fusion_v5.py:302-312` — current v20 instantiation
* `motionflow_mv/fusion/diffusion_pose_refiner_v20.py:81-124` — v20 refiner class
* `motionflow_mv/fusion/multiview_geometry_fusion_v25.py` — v25 ray tokens and depth-proposal head
* `motionflow_mv/fusion/temporal_geometry_fusion_v26.py` — v26 temporal extension
* `docs/proposals/v27_next_iteration_decision_matrix.md` — gates and success criteria
