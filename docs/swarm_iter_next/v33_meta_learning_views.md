# v33: Meta-learning few-shot view adaptation

**Direction slug:** `meta_learning_views`  
**Date:** 2026-08-08  
**Target venue:** ICRA/CVPR 2027  
**Related prior work:** v5 variable-view training, v31 geometry-aware camera embeddings, v32 trajectory consistency / domain-aware curriculum.

---

## 1. Problem statement and motivation

The current `OmniMultiViewFusionV5` pipeline (v31/v32) is trained with **variable-view subsets** and **camera-conditioned view embeddings**, which lets it run with arbitrary active views at inference. However, the model is still trained in a *standard supervised* way: every batch samples random subsets from the same fixed camera rigs, and the model never explicitly learns to **adapt its view-level representations to a new, unseen camera layout from only a handful of examples**.

In real deployments, rigs can be ad-hoc (2–14 cameras, different baselines, partial occlusion, new room geometries). A few calibration frames are usually available, but not enough to retrain the whole network. We therefore want to add a **meta-learning (few-shot) view-adaptation stage** that, given a small support set from a new camera layout, quickly adjusts the view-level feature embedding/aggregation so that the same backbone generalises better to that layout.

Why this fits v33:

- It builds directly on the existing v5 variable-view machinery and v31 camera embedding.
- It addresses the ICRA/CVPR 2027 deployment requirement of robustness to **arbitrary, possibly unseen, camera rigs**.
- It is orthogonal to the v31/v32 geometry, physical, and temporal losses, so it can be combined with them.

---

## 2. Proposed architecture changes

### 2.1 New module

**File:** `motionflow_mv/fusion/meta_view_adaptation_v33.py`

Introduce a small, first-order MAML/Reptile-style adapter:

```python
class MetaViewAdapterV33(nn.Module):
    """First-order meta-learner for few-shot view adaptation.

    Given a support set (views, cameras, GT poses) from a new rig, performs a
    few gradient-descent steps on a lightweight per-view residual and/or on the
    camera-conditioned view embedding.  The adapted residual is then added to
    the view tokens before the spatio-temporal transformer.
    """
```

Key components:

- `ViewAdaptationResidual(d, n_views)`: a per-view residual vector `Δe_v` that is added to the camera embedding.
- `inner_step(support_x, support_K, support_R, support_t, support_y)`: performs `v33_meta_inner_steps` gradient steps on the residual using the support batch.
- `forward(query_x, ...)`: returns adapted view tokens for the query set.

The adapter is **zero-initialised** so that, when disabled or at initialization, the network behaviour is identical to the baseline.

### 2.2 Integration into `OmniMultiViewFusionV5`\n
In `motionflow_mv/fusion/omniview_fusion_v5.py`:

- Add constructor flags:
  - `use_meta_view_adaptation_v33: bool = False`
  - `v33_meta_inner_lr: float = 1e-4`
  - `v33_meta_inner_steps: int = 3`
  - `v33_meta_support_views: int = 4`
  - `v33_meta_loss_weight: float = 0.5`
  - `v33_meta_adapt_mode: str = "camera_emb"` (options: `"camera_emb"`, `"view_weights"`, `"aggregator"`)
- After the existing `camera_view_embedding` block, if `use_meta_view_adaptation_v33` is enabled, instantiate `MetaViewAdapterV33`.
- The adapter consumes the camera-conditioned view embedding, produces an adapted residual, and feeds it back into the view tokens before the ST transformer.

In `experiments/train_omniview_fusion_v5_webbridge_multi.py`:

- Add CLI flags mirroring the model kwargs above.
- In `build_model_from_args`, pass them to `OmniMultiViewFusionV5`.
- In `build_compute_loss`, when meta-adaptation is enabled:
  1. Sample a **support** view subset and a **query** view subset from the same clip (disjoint or overlapping; disjoint is the default).
  2. Run the inner loop on the support batch to adapt the view residual.
  3. Compute the outer loss (MPJPE + auxiliary losses) on the query batch using the adapted model state.
  4. Add `v33_meta_loss_weight * query_loss` to the total training loss.

### 2.3 Data / preprocessing changes

No new dataset files are required. The existing variable-view training already samples random subsets via `augment_clip(..., variable_view_subset=True)`. The meta-learning wrapper reuses this logic but performs **two independent samples per clip**:

- **Support set**: `k_s ~ Uniform(v33_meta_min_support_views, v33_meta_support_views)` views.
- **Query set**: `k_q ~ Uniform(variable_view_min_views, variable_view_max_views)` views, disjoint from support when possible.

Both sets are drawn from the same underlying `(x, K, R, t)` rig, so the model learns to adapt to the geometry of the current camera layout rather than memorising fixed view identities.

New training-only CLI flags:

- `--v33_meta_min_support_views` (default 2)
- `--v33_meta_disjoint_support_query` (default True)
- `--v33_meta_support_prob` (default 0.5, probability of applying meta-learning on a given batch)

### 2.4 Evaluation protocol

Add a few-shot adaptation benchmark script (proposal only, no source edit now):

- `experiments/eval_meta_view_adaptation_v33.py`
  - Takes a trained v33 checkpoint and a held-out `.npz` sequence.
  - Splits the sequence into a **support** prefix (e.g. 10 clips) and a **query** suffix.
  - Performs `v33_meta_inner_steps` adaptation steps on the support prefix.
  - Reports `query_MPJPE` and `PA_MPJPE`.
  - Also reports the standard full-view baseline `MPJPE@V` for comparison.

---

## 3. Training command / ablation flags

### Minimal CPU smoke (proposed)

```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --smoke \
  --use_meta_view_adaptation_v33 \
  --v33_meta_inner_steps 1 \
  --v33_meta_support_views 2 \
  --v33_meta_loss_weight 0.5
```

### Full WebBridge + H36M + MPI mixed run (proposed)

```bash
python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
  --use_mixed_loader \
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
  --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight \
  --use_domain_embedding \
  --use_deformable_cross_view_attention_v18 \
  --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
  --v25_use_geometry_attention --v25_use_learned_depth_triangulation \
  --v25_use_geometry_bundle_adjustment \
  --use_camera_view_embedding --use_set_view_aggregator \
  --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
  --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
  --use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1 \
  --use_physical_space_temporal_loss_v29 \
  --use_meta_view_adaptation_v33 \
  --v33_meta_inner_lr 1e-4 \
  --v33_meta_inner_steps 3 \
  --v33_meta_support_views 4 \
  --v33_meta_loss_weight 0.5 \
  --v33_meta_adapt_mode camera_emb \
  --d 64 --residual_hidden 128 --n_st_layers 2 --graph_num_layers 1 \
  --n_joint_layers 1 --n_heads 4 --clip_len 9 --epochs 20 \
  --batch_size 8 --train_samples 1000 --val_stride 10 \
  --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 \
  --early_stopping_patience 5 \
  --output outputs/omniview_fusion_v33_meta_learning_views.pth
```

### Ablations to run

| Flag | Purpose |
|------|---------|
| `--v33_meta_adapt_mode camera_emb` | Adapt only the camera-conditioned view embedding residual. |
| `--v33_meta_adapt_mode view_weights` | Adapt a per-view scalar weight before aggregation. |
| `--v33_meta_adapt_mode aggregator` | Adapt the ISAB/Perceiver inducing points. |
| `--v33_meta_inner_steps 1,3,5` | Sensitivity to inner-loop length. |
| `--v33_meta_inner_lr 1e-5,1e-4,1e-3` | Inner-loop learning rate sweep. |
| `--v33_meta_loss_weight 0.0,0.1,0.5,1.0` | Weight of the meta-adaptation query loss. |
| `--v33_meta_support_views 2,4,6` | Few-shot budget. |

---

## 4. Expected metrics and baseline to beat

### Primary baselines

- **v31/v32 full-view baseline** on the mixed manifest: roughly **25–28 mm MPJPE** (clean, full 14 views), depending on the exact v32 combination.
- **v32 variable-view baseline**: MPJPE should degrade gracefully from `k=14` to `k=2`. We target no regression at `k=14` and improvement at low `k`.

### Targets for v33

| Scenario | Metric | Target |
|----------|--------|--------|
| Full-view inference | `val_MPJPE` | ≤ baseline (≤ 28 mm) |
| Variable-view `k=2..6` | `MPJPE@k` | ≥ 5–10% better than v32 curve |
| Few-shot adaptation on held-out rig (4 support views) | `query_MPJPE` | Within 5% of the full-trained baseline on the same rig |
| Training stability | `val_MPJPE` after epoch 1 | < 35 mm (no collapse) |
| Overhead | wall-time per epoch | < 1.5× baseline (first-order only) |

### New metrics to log

- `meta/query_mpjpe`: MPJPE on the query set after inner-loop adaptation.
- `meta/support_mpjpe`: MPJPE on the support set before/after adaptation (diagnostic).
- `meta/adapt_loss`: the outer-loop meta-adaptation loss.
- `meta/grad_norm`: norm of the inner-loop gradient (watch for instability).

---

## 5. Risks / unknowns

1. **Gradient overhead.** Even first-order MAML needs a forward+backward pass on the support set and another on the query set. On the RTX 4090 / A800 this may push memory limits when `d=128` or large clips are used. Mitigation: keep `v33_meta_inner_steps` small (1–3) and use the `camera_emb` mode, which has the fewest parameters.
2. **Second-order effects.** If we ever move to full MAML (rather than first-order / Reptile), the Hessian-vector products through the ST transformer could be expensive. The proposal explicitly stays first-order.
3. **Support/query split semantics.** Disjoint subsets from the same rig are easy to define, but ensuring they share enough geometry for adaptation while remaining non-overlapping requires careful masking in the existing `view_mask` path.
4. **Interaction with domain curriculum.** The domain-aware view curriculum clamps subsets to the real camera count per domain. Meta-learning across H36M (4 views) and MPI (14 views) may need domain-specific support-view budgets. The flag `v33_meta_support_views` may need to be domain-aware.
5. **Held-out evaluation data.** A true few-shot adaptation benchmark requires a camera rig or sequence not seen during training. The current mixed manifest only splits by subject/sequence; we may need to reserve an entire camera rig or a synthetic rig for validation.
6. **Orthogonality with v31/v32 losses.** The physical-space losses (v29) and trajectory-consistency loss (v32) operate after the ST transformer and should not conflict with meta-view adaptation, but the combined loss surface may become noisier. A warmup on the meta-loss weight is advised.

---

## 6. Files that would be touched (implementation checklist)

- **New:** `motionflow_mv/fusion/meta_view_adaptation_v33.py`
- **Edit:** `motionflow_mv/fusion/omniview_fusion_v5.py` — add v33 flags and instantiate the adapter.
- **Edit:** `experiments/train_omniview_fusion_v5_webbridge_multi.py` — add CLI args and the support/query meta-learning loop in `build_compute_loss`.
- **New:** `experiments/eval_meta_view_adaptation_v33.py` (evaluation script for the held-out benchmark).
- **New:** `tests/test_meta_view_adaptation_v33.py` — smoke tests for forward pass, inner-loop adaptation, and zero-identity init.
- **New:** `scripts/run_v33_meta_learning_views_smoke.sh` — local RTX 4090 / CPU smoke launcher.

---

## 7. Next immediate step

Implement the `MetaViewAdapterV33` module and a CPU smoke test that verifies the inner-loop adaptation does not crash and that, with `--v33_meta_loss_weight 0.0`, the model exactly matches the existing v32 baseline. Once the smoke passes, queue a short d=64 full run on the local RTX 4090 before committing A800-D resources.
