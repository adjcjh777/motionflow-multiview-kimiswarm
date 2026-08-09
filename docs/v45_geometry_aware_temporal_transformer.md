# v45: Geometry-Aware Multi-View Temporal Transformer

**Status:** Proposal (pending v44 decision)
**Labels:** `experiment`, `P1-next`
**Depends on:** v42/v43/v44 A800 results (issues #152, #153, #154)

## 1. Motivation

Current results show two clear trends:

1. **Strong geometry fusion is the dominant driver of accuracy.** The v25 geometry-fusion baseline reaches **17.17 mm MPJPE** on A800, while the v31–v43 complex stacks remain in the 26–37 mm range locally and have not clearly surpassed v25 at scale.
2. **Temporal and view-joint graph modeling gives small but consistent gains.** v35 (temporal VJGN) improves over v34 by ~0.1 mm, and v36 (uncertainty-gated iterative refinement) reaches 26.42 mm locally, suggesting that structured temporal reasoning can help when built on a strong geometric foundation.

The v31–v43 architectures have largely explored *additive complexity* (graph networks, uncertainty gates, outlier detectors, physical losses) on top of a feature backbone. v45 instead asks:

> Can we build a compact, geometry-aware temporal transformer that directly reasons about multi-view rays, temporal dynamics, and joint-level uncertainty in a single coherent architecture, rather than stacking independent modules?

If the v44 decision lands on a v25-based foundation (Branch A or D in `docs/v44_decision_plan.md`), v45 becomes the natural next step: keep the proven geometry-fusion backbone, but replace the disjoint refinement stages with a unified temporal transformer that operates on geometry-aware tokens.

## 2. Architecture Sketch

v45 is a single-stage network with four components:

```text
Input: B x T x V x J x C 2D keypoints + camera parameters
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Geometry-Aware Token Embedding                          │
│    • Each token represents one (view, joint) pair.          │
│    • Embed 2D keypoint, ray direction, and camera center.    │
│    • Add learned view/joint/position embeddings.             │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────
│ 2. Ray-Conditioned Cross-View Encoder (single block)      │
│    • Self-attention over view dimension, biased by epipolar  │
│      and ray-intersection geometry (reuses v34 ray-attention │
│      kernels but without the heavy graph stage).            │
│    • Produces per-view latent geometry tokens.              │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Temporal Pose Transformer (TPT)                         │
│    • Stack of transformer layers over time.                 │
│    • Tokens are the fused multi-view joint features from    │
│      step 2, concatenated across views into one joint token.│
│    • Temporal self-attention plus learned temporal position. │
│    • Optional: causal masking for online inference.         │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Geometry-Guided 3D Lift + Uncertainty Head             │
│    • From each joint token, regress 3D position and          │
│      per-joint log-variance (aleatoric uncertainty).         │
│    • Auxiliary triangulation loss against pseudo-GT 3D.      │
│    • Optional: skeleton-aware physical prior (v40) as a     │
│      soft post-hoc regularizer.                               │
└─────────────────────────────────────────────────────────────┘
```

### Key design principles

| Principle | How v45 implements it | What it avoids from v31–v43 |
|---|---|---|
| **Unified attention, not module stacking** | One cross-view block + one temporal block | Separate graph, uncertainty, outlier, and physical modules |
| **Geometry as first-class signal** | Ray embedding and ray-intersection bias in every token | Retrofitting geometry bias into a 2D feature backbone |
| **Uncertainty-aware output** | Per-joint log-variance head, precision-weighted MSE | External outlier-rejection stage |
| **Temporal reasoning by design** | Transformer over time, not per-frame + post smoothing | Expensive TTE module that broke in v29 |

## 3. Expected Experiments

### 3.1 Smoke tests (RTX 4090)

| Config | Purpose | Success criterion |
|---|---|---|
| `v45_d64_clip9_smoke` | Verify the model trains end-to-end | val_MPJPE < 95 mm on 50-sample smoke |
| `v45_no_temporal_baseline` | Temporal block replaced by a single MLP over time | Validate temporal contribution |
| `v45_no_ray_bias` | Remove ray-intersection bias in cross-view block | Validate geometry contribution |

### 3.2 Full runs (A800)

| Config | Purpose |
|---|---|
| `v45_d128_full` | Full-scale v45 on WebBridge/H36M/MPI mixed manifest |
| `v45_plus_v40_physical` | Add v40 skeleton-aware physical loss as auxiliary term |
| `v45_plus_v41_domain_weights` | Add v41 domain weights for WebBridge vs H36M/MPI balance |
| `v45_vs_v25_head_to_head` | Same data/config as v25 all-train baseline, only model differs |

### 3.3 Ablation targets

1. **Cross-view block depth:** 1 vs 2 vs 4 layers.
2. **Temporal modeling:** temporal transformer vs 1D temporal convolution vs no temporal.
3. **Uncertainty head:** per-joint log-variance vs no uncertainty vs shared global uncertainty.
4. **Geometry bias:** ray-intersection bias, epipolar bias, both, neither.

## 4. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| TTE-like temporal overfitting (v29) | Use shorter clip lengths first (9 frames), causal masking optional |
| OOM from full attention over views × joints × time | Factor attention: view-first, then time; limit temporal window or use sparse attention |
| v25 baseline still beats v45 | Keep v45 as an ablation branch; do not merge unless it beats v25 by >5% |
| Longer training needed | Add SWA/EMA and early stopping from the start |

## 5. Definition of Done

- [ ] v45 smoke config created under `configs/`.
- [ ] RTX 4090 smoke run reaches val_MPJPE < 95 mm.
- [ ] A800 full run completes at least one epoch with no OOM.
- [ ] Head-to-head comparison with v25 all-train baseline is recorded in `docs/v45_results.md`.
- [ ] Decision: merge into main, iterate, or abandon and document lessons.

## 6. Paper Story Fit

If v45 succeeds, the paper narrative becomes:

> Geometry-aware multi-view fusion remains the foundation, but a lightweight temporal transformer operating on ray-aware tokens can further refine dynamic poses and provide calibrated per-joint uncertainty, closing the gap between strong geometric baselines and structured temporal reasoning.

If v45 does not beat v25, the result still strengthens the argument that **geometry fusion is the primary signal** and that future work should focus on better geometric priors rather than larger attention stacks.
