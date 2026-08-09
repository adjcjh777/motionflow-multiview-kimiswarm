# v52 Self-Supervised Multi-View Pre-Training Head (SS-MVP)

**Author:** design-swarm-agent-v52  
**Tracking issue:** #182  
**Depends on:** v46 SVG, v50 SEFH, v51 CDSVR  
**Target pipeline stage:** between 2D pose extraction and multi-view fusion / physical-space alignment.

## 1. Motivation

The MotionFlow pipeline currently learns multi-view fusion only from paired 3D supervision.  v52 adds an auxiliary *self-supervised pre-training head* that exploits the inherent redundancy of calibrated multi-view video itself: the same 3D pose projects consistently across views and evolves smoothly in time.  By randomly masking (view, joint, time) tokens and asking the network to reconstruct them from the remaining tokens, the model learns richer per-view/per-joint representations before any full 3D labels are required.  This is aligned with the paper story

```
multi-view video → human pose extraction → multi-view fusion and calibration → physical-space alignment → optimized motionflow pipeline
```

and is expected to make v46 sparse-view generalization and v51 cross-domain reliability more robust when labelled 3D data is scarce or when the test domain shifts.

## 2. Architecture

SS-MVP is an optional auxiliary head inserted **after the per-view feature tokens are produced and before triangulation**.  It is a lightweight transformer with two sub-heads:

1. **Masked Token Encoder (MTE)** – reasons over visible tokens to fill in masked ones.
2. **Geometry-Aware Reconstruction Head (GARH)** – predicts the masked 2D re-projection and a soft 3D consistency score.

The module is identity-at-init: the final residual projection is zero-initialised, so when `use_v52_self_supervised_pretraining=True` the main 3D pose branch is unchanged at epoch 0.

### 2.1 Token construction

Let the existing per-view feature tensor be

```
F ∈ R^(B×T×V×J×D)
```

where `B` is batch, `T` clip length, `V` views, `J` joints, `D` feature dim.  Flatten to token sequence

```
F_tokens ∈ R^(N×D) ,  N = B·T·V·J
```

Add sinusoidal or learned positional embeddings for view index `v`, joint index `j`, and time `t`.

### 2.2 Masking strategy

For each token draw a Bernoulli mask with rate `v52_mask_ratio` (default 0.25).  Masked tokens are replaced by a learned `[MASK]` vector.  To avoid degeneracy, the mask is constrained so that every joint keeps at least `v52_min_visible_views` unmasked views per time step (using the existing `view_mask`).

Three corruption modes are sampled with equal probability during training:

| Mode | What is masked | What the head must reconstruct |
|------|----------------|----------------------------------|
| `view` | one view for a joint | the 2D keypoint of that view from the others |
| `joint` | a joint across all views | the 3D position / 2D projections of that joint |
| `time` | a short contiguous time window for one view | the missing 2D track segment from temporal neighbours |

### 2.3 Masked transformer

A small 4-layer transformer encoder with `v52_hidden` (default 64) and `v52_n_heads` (default 4) processes the masked tokens.  Multi-head self-attention attends only to unmasked tokens (standard attention mask) and outputs updated tokens

```
F_tokens = TransformerEncoder(F_tokens + pos_embed) ∈ R^(N×D)
```

### 2.4 Reconstruction heads

From the masked positions we extract two predictions:

**2D re-projection head**

```
p̂_2d ∈ R^(B×T×V×J×2)  =  Linear_2d(F̂_tokens)
L_2d = Σ mask · || p̂_2d − p_2d_gt ||²_2 / Σ mask
```

**3D consistency head (geometry pretext)**

A per-joint 3D position is triangulated only from the *predicted* 2D points of the masked subset using the existing differentiable DLT routine, producing `X̂_3d ∈ R^(B×T×J×3)`.  The geometry loss is the re-projection error back to all views:

```
L_geo = Σ_{v=1..V} || Π_v(X̂_3d) − p_2d_gt[v] ||²_2
```

A third **contrastive view-consistency loss** pulls embeddings of the same 3D point across different views and pushes away different joints/time steps:

```
L_cont = − log  exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ)
```

where `z_i` is the D-dimensional token at a sampled (view, joint, time).

Total auxiliary loss:

```
L_v52 = v52_loss_2d_weight · L_2d
      + v52_loss_geo_weight · L_geo
      + v52_loss_cont_weight · L_cont
```

Default weights: `0.1`, `0.1`, `0.01` respectively.  The loss is applied only during pre-training / warm-up epochs controlled by `v52_warmup_epochs`; after that the head can be frozen or kept with weight zero.

## 3. Inputs / Outputs

### Inputs

| Symbol | Shape | Description |
|--------|-------|-------------|
| `F` | `(B, T, V, J, D)` | per-view feature tokens from the main ST transformer |
| `p_2d` | `(B, T, V, J, 2)` | original 2D keypoints |
| `conf` | `(B, T, V, J)` | detection confidences |
| `K, R, t` | `(B·T, V, 3, 3)` / `(B·T, V, 3)` | calibrated camera parameters |
| `view_mask` | `(B, T, V)` or `(B·T, V)` | existing binary view mask |
| `domain_id` | `(B,)` | optional domain label for domain-conditional masking |

### Outputs

| Symbol | Shape | Description |
|--------|-------|-------------|
| `p̂_2d` | `(B, T, V, J, 2)` | reconstructed 2D keypoints for masked tokens |
| `X̂_3d` | `(B, T, J, 3)` | triangulated 3D pose from masked subset |
| `mask` | `(B, T, V, J)` | boolean mask indicating which tokens were corrupted |
| `aux_loss` | scalar | `L_v52`, added to the training objective |

The main forward branch continues with unmasked / original features so triangulation, v46/v47/v48/v50/v51 heads, and physical-space alignment are unchanged.

## 4. Config flags

```yaml
use_v52_self_supervised_pretraining: false
v52_mask_ratio: 0.25                    # fraction of tokens masked
v52_min_visible_views: 2                # keep at least this many views unmasked per joint
v52_hidden: 64                          # transformer hidden dim
v52_n_heads: 4                          # attention heads
v52_n_layers: 4                         # transformer layers
v52_dropout: 0.1
v52_loss_2d_weight: 0.1
v52_loss_geo_weight: 0.1
v52_loss_cont_weight: 0.01
v52_temp_cont_window: 3                 # temporal window for time-mode corruption
v52_warmup_epochs: 2                    # epochs with active v52 loss; 0 = always active
v52_identity_init: true                 # zero-initialise final projection
v52_use_view_pos_embed: true
v52_use_joint_pos_embed: true
v52_use_time_pos_embed: true
v52_contrastive_temperature: 0.1
v52_corruption_mix: [0.34, 0.33, 0.33]  # view / joint / time mode probabilities
```

## 5. Expected MPJPE impact

- **H36M / MPI-INF-3DHP (full supervision):** modest gain of ≈0.5–1.5 mm by improving the learned per-view features used by v25/v45 triangulation and v50/v51 reliability.
- **Sparse-view settings (v46):** larger relative gain of 1.5–3 mm because masked-view reconstruction explicitly trains the network to operate when some views are missing.
- **Cross-domain / limited-label scenarios (v48, v51):** 2–4 mm improvement when 3D labels are scarce; the auxiliary loss acts as a regulariser that uses cheap unlabelled multi-view video.

## 6. Risks

See `docs/swarm_iter26/reports/agent_self_supervised_multiview_pretraining_risks.md` for a detailed risk matrix.

## 7. Five-step implementation plan

1. **Add the module file** (`motionflow_mv/fusion/self_supervised_multiview_pretraining_v52.py`) implementing `SelfSupervisedMultiViewPretrainingV52(nn.Module)` with the masked transformer, two reconstruction heads, and contrastive loss.  Guarantee identity-at-init via zero-initialised final layers.

2. **Wire into `OmniMultiViewFusionV5`**: add the `use_v52_*` flags in `__init__`, instantiate the head after the per-view ST transformer features are available (around the v46/v47 integration point), and add a `v52_aux_loss` term to the existing auxiliary-loss summation in the trainer.

3. **Update the trainer** (`experiments/train_omniview_fusion_v5_webbridge_multi.py`) to apply the masking only during training, pass `view_mask` and `domain_id` to the head, and respect `v52_warmup_epochs` so the loss is zeroed after the warm-up.

4. **Add smoke config** (`configs/benchmark_v52_ssmvp_smoke.yaml`) with `use_v52_self_supervised_pretraining=true`, `clip_len=3`, `train_samples=50`, and run the smoke script.  Acceptance: smoke finishes, val_MPJPE finite, and the auxiliary loss does not dominate the total loss.

5. **Ablate on the v46/v51 baseline**: run a controlled 2-epoch comparison (baseline vs. baseline+v52) on the local RTX 4090 using the same seed.  If val_MPJPE improves or is neutral, add the A800 queue entry to `scripts/launch_v33_a800_queue.py`.
