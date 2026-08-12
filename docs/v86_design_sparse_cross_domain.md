# v86 Design — Sparse-View / Cross-Domain MotionFlow-MultiView

**Status:** Design proposal — ready for implementation and A800 scheduling.  
**Date:** 2026-08-12  
**Target protocols:**
- H36M true-GT standard (S1,5,6,7,8 → S9/S11)
- H36M true-GT + AIST++ mixed-dataset cross-domain transfer
- Optional: H36M + AIST++ + Shelf/Campus detected  

**Related files:**
- `motionflow_mv/fusion/random_view_dropout_v85.py`
- `motionflow_mv/fusion/view_count_conditioning_v86.py`
- `motionflow_mv/fusion/separate_sparse_view_head_v86.py`
- `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`
- `motionflow_mv/fusion/omniview_fusion_v5.py`
- `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- `configs/splits/h36m_true_gt_standard.yaml`
- `configs/splits/h36m_true_gt_aist_mixed_train_val_a800.yaml`
- `configs/splits/mix_h36m_aist_shelf.yaml`

---

## 1. Motivation and positioning

v85 introduced **training-time random whole-view dropout + active-view-count embedding** to address the catastrophic failure of learned MotionFlow models when fewer than four views are available. Initial v85 training looks promising (val MPJPE falling to ~36 mm by Epoch 2), but two questions remain unanswered:

1. **Is the simple count embedding strong enough?** A scalar embedding keyed only by `k` may not provide enough signal to the geometry-attention layers to adapt their behaviour across `k=2/3/4`.
2. **Does the same architecture work across datasets?** All v85 experiments are H36M-only. For CVPR 2027 the paper needs honest cross-domain numbers (AIST++ → H36M, MPI detected-2D, Shelf/Campus).

**v86** therefore unifies three mechanisms that were developed as separate ablations into a single **sparse-view / cross-domain** model:

| Component | File | Purpose |
|-----------|------|---------|
| v85 random view dropout | `random_view_dropout_v85.py` | Force the model to see `k=2/3/4` inputs during training |
| v86 stronger count conditioning | `view_count_conditioning_v86.py` | Replace the scalar count embedding with an MLP-generated count token |
| v86 separate sparse-view head | `separate_sparse_view_head_v86.py` | Route `k < n_views` samples to a dedicated identity-at-init correction head |
| Cross-domain mixed loader | `configs/splits/h36m_true_gt_aist_mixed_train_val_a800.yaml` | Train on H36M + AIST++ with domain embeddings |

The goal is a **single model** that:
- Matches or beats v25 stability on full-view H36M true-GT (~30 mm).
- Produces finite, non-catastrophic MPJPE@k for `k=2` and `k=3` without a DLT fallback.
- Transfers across datasets (AIST++ → H36M) better than the v25 mixed-dataset baseline (33.42 mm).

---

## 2. Problem being solved

### 2.1 Sparse-view failure (H36M true-GT)

| Variant | k=2 S9/S11 | k=3 S9/S11 | k=4 S9/S11 | Source |
|---------|-----------:|-----------:|-----------:|--------|
| v25 stability (learned) | 3482.62 / 3376.04 | 1042.45 / 1030.19 | 116.98 / 110.58 | `docs/results_true_gt_h36m.md` |
| v25 + DLT fallback | 58.18 / 49.35 | 33.32 / 25.28 | 116.98 / 110.58 | `outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json` |

The learned model is catastrophically bad for `k<4`, while direct confidence-weighted DLT on the same 2D observations is reasonable. v85 attacks this by training with dropout; v86 adds stronger count-conditioning and a dedicated sparse branch.

### 2.2 Cross-domain gap

| Experiment | Train | Test | MPJPE | Source |
|------------|-------|------|------:|--------|
| v25 stability | H36M true-GT | H36M S9/S11 | **30.83 mm** | `docs/results_true_gt_h36m.md` |
| v25 mixed H36M+AIST++ | H36M + AIST++ | H36M S9/S11 | **33.42 mm** | `docs/paper_draft_icra_cvpr_2027.md` |
| AIST++-only fast v2 | AIST++ | H36M S9/S11 | **93.94 mm** | `outputs/eval_aistpp_only_medium_a800_fast_v2_h36m_test.json` |

The mixed-dataset run already slightly beats single-dataset v25 on some metrics, but it diverged after Epoch 3. v86 aims to stabilise mixed training by combining sparse-view robustness with cross-domain regularisation.

---

## 3. Architecture sketch

```
points_2d, K, R, t, confidence, view_mask
            │
            ▼
    ┌─────────────────────┐
    │  Initial DLT seed   │  ← triangulate_initial (confidence-weighted)
    └─────────────────────┘
            │
            ▼
    ┌─────────────────────┐
    │   v85 random view   │  ← training-time whole-view dropout (k=2/3/4)
    │   dropout + count   │
    │   embedding         │
    └─────────────────────┘
            │
            ▼
    ┌─────────────────────┐
    │  RayTokenizer       │  ← (B, T, V, J, d)
    └─────────────────────┘
            │
            ▼
    ┌─────────────────────────────┐
    │  v86 stronger count         │  ← MLP-generated view-count token
    │  conditioning               │    added to every ray token
    └─────────────────────────────┘
            │
            ▼
    ┌─────────────────────┐
    │  v25/v83 geometry   │  ← geometry-aware cross-view attention +
    │  attention          │    optional view-conditioned temporal attn
    └─────────────────────┘
            │
            ▼
    ┌─────────────────────┐
    │  Depth-proposal     │  ← v25 learned triangulation
    │  triangulation head │
    └─────────────────────┘
            │
            ▼
    ┌─────────────────────────────┐
    │  v86 separate sparse-view   │  ← only active when k < n_views
    │  head                       │    pools active-view tokens,
    │                             │    adds count embedding,
    │                             │    predicts residual around DLT seed
    └─────────────────────────────┘
            │
            ▼
    ┌─────────────────────┐
    │  Temporal pose      │  ← v81/v82-style residual refinement on
    │  attention /        │    the triangulated 3-D pose
    │  residual MLP         │
    └─────────────────────┘
            │
            ▼
      pred_3d_refined
```

### 3.1 Key design decisions

1. **Geometry first.** The initial DLT seed is always computed from the active views. The learned components refine it, never replace it.
2. **Count-aware tokens.** Both the ray tokens and the sparse-view head receive an explicit, MLP-conditioned active-view-count signal.
3. **Separate sparse branch.** A dedicated head for `k < n_views` prevents the full-view refinement path from being contaminated by sparse-view gradients and vice-versa.
4. **Identity at init.** The v86 count-conditioning and sparse-view heads are initialised to produce zero output, so training starts from the proven v25/v85 baseline.
5. **Domain embeddings.** When training on mixed datasets, a small learned domain embedding is added to the ray tokens; at inference it is selected by the dataset ID or set to the H36M domain for zero-shot transfer.

---

## 4. Component specifications

### 4.1 v85 random view dropout (kept unchanged)

File: `motionflow_mv/fusion/random_view_dropout_v85.py`

For each training clip a random subset of entire views is dropped, subject to `min_views=2`. The active-view count is embedded into the ray tokens. v86 keeps this mechanism exactly as in v85.

Flags:
```bash
--use_random_view_dropout_v85
--v85_dropout_prob 0.3
--v85_min_views 2
--v85_use_count_embedding true   # enabled; v86 adds its own stronger token on top
```

### 4.2 v86 stronger view-count conditioning

File: `motionflow_mv/fusion/view_count_conditioning_v86.py`

Replaces the scalar count embedding with:

```python
class ViewCountConditioningV86(nn.Module):
    def __init__(self, d=128, n_views=4, hidden=64, n_layers=2, dropout=0.1):
        ...
```

Forward:
```
count = sum(view_mask, dim=-1)              # (B, T)
count_emb = Embedding(max_views+1, hidden)  # (B, T, hidden)
token   = MLP(count_emb)                    # (B, T, d)
out     = tokens + gate * token[:, :, None, None, :]
```

The final MLP layer and the gate are zero-initialised, so the module is identity at the start of training.

Flags:
```bash
--use_v86_strong_count_conditioning
--v86_count_hidden 64
--v86_count_n_layers 2
--v86_count_dropout 0.1
```

### 4.3 v86 separate sparse-view head

File: `motionflow_mv/fusion/separate_sparse_view_head_v86.py`

A lightweight head that is applied only when `active_views < n_views`:

```python
class SeparateSparseViewHeadV86(nn.Module):
    def __init__(
        self,
        d: int = 128,
        n_views: int = 4,
        n_joints: int = 17,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_count_embedding: bool = True,
    ):
        ...
```

Forward:
```
pooled  = mean(tokens * view_mask) over active views  # (B, T, J, d)
count_emb = Embedding(active_count)                    # (B, T, d)
features = pooled + count_emb[:, :, None, :]
residual = MLP([features, pred_3d_init])             # (B, T, J, 3)
gate     = tanh(residual_gate)                         # scalar, init 0
pred_3d_sparse = pred_3d_init + gate * residual
```

The final MLP layer and the residual gate are zero-initialised. The head is therefore identity at training start.

It is inserted in `MultiViewGeometryFusionV25.forward` after the depth-proposal triangulation head:

```python
if self.use_v86_separate_sparse_view_head:
    sparse_mask = view_mask.sum(dim=-1) < self.n_views
    if sparse_mask.any():
        pred_sparse = self.separate_sparse_view_head_v86(
            tokens, pred_3d_init, view_mask
        )
        pred_3d_ref = torch.where(sparse_mask[..., None, None], pred_sparse, pred_3d_ref)
```

Flags:
```bash
--use_v86_separate_sparse_view_head
--v86_ssv_head_hidden 128
--v86_ssv_head_n_layers 2
--v86_ssv_head_dropout 0.1
--v86_ssv_head_use_count_embedding
```

### 4.4 Cross-domain training

The model uses the existing `use_domain_embedding` mechanism in `OmniMultiViewFusionV5`:

```python
if self.use_domain_embedding and domain_id is not None:
    domain_emb = self.domain_embedding(domain_id)  # (B, d)
    feat = feat + domain_emb.view(B, 1, 1, 1, self.d)
```

When `--use_mixed_loader` is set, `experiments/train_omniview_fusion_v5_webbridge_multi.py` loads a manifest containing multiple datasets and passes per-sample `domain_id` labels.

Mixed-dataset manifests already available:
- `configs/splits/h36m_true_gt_aist_mixed_train_val_a800.yaml` — H36M true-GT + AIST++
- `configs/splits/mix_h36m_aist_shelf.yaml` — H36M + AIST++ + Shelf/Campus

Flags:
```bash
--use_mixed_loader
--mixed_manifest configs/splits/h36m_true_gt_aist_mixed_train_val_a800.yaml
--num_domains 2
--use_domain_embedding
```

---

## 5. Training recipe

### 5.1 Stage 1 — H36M true-GT single-dataset sparse-view tuning

Goal: match v25 stability on `k=4` and eliminate the catastrophic `k<4` failure.

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run_v86_sparse_cross_domain_medium_a800.sh
```

Key hyperparameters (v25 stability base):

| Setting | Value |
|---|---|
| d | 128 |
| residual_hidden | 256 |
| n_st_layers | 3 |
| n_heads | 4 |
| clip_len | 13 |
| batch_size | 16 |
| train_samples | 4096 |
| epochs | 20 |
| lr | 1e-4 |
| lr_warmup_epochs | 4 |
| max_grad_norm | 1.0 |
| ema_decay | 0.999 |
| weight_decay | 1e-4 |
| early_stopping_patience | 3 |

v86-specific flags:

```bash
--use_random_view_dropout_v85 --v85_dropout_prob 0.3 --v85_min_views 2 --v85_use_count_embedding
--use_v86_strong_count_conditioning --v86_count_hidden 64 --v86_count_n_layers 2 --v86_count_dropout 0.1
--use_v86_separate_sparse_view_head --v86_ssv_head_hidden 128 --v86_ssv_head_n_layers 2 --v86_ssv_head_dropout 0.1 --v86_ssv_head_use_count_embedding
```

Expected output:
- Checkpoint: `outputs/ablations/v86_sparse_cross_domain_medium_a800.pth`
- Log: `outputs/ablations/v86_sparse_cross_domain_medium_a800.log`

### 5.2 Stage 2 — Cross-domain mixed training (H36M + AIST++)

Goal: improve cross-domain transfer while preserving sparse-view robustness.

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_v86_sparse_cross_domain_mixed_a800.sh
```

Changes from Stage 1:
- Replace `--mixed_manifest configs/splits/h36m_true_gt_standard.yaml` with `--mixed_manifest configs/splits/h36m_true_gt_aist_mixed_train_val_a800.yaml`
- Set `--num_domains 2`
- Keep v86 sparse-view components enabled
- Keep the same conservative optimisation recipe (low LR, warmup, EMA)

Optional: warm-start from the Stage 1 checkpoint to preserve H36M sparse-view behaviour.

### 5.3 Stage 3 (optional) — H36M + AIST++ + Shelf/Campus

Goal: test robustness on a third, very different camera layout (Shelf/Campus has 3 views and a different skeleton mapping).

```bash
--mixed_manifest configs/splits/mix_h36m_aist_shelf.yaml
--num_domains 3
```

This is exploratory; the primary paper numbers are expected from Stage 1 and Stage 2.

---

## 6. Evaluation protocol

### 6.1 Full-view H36M test (k=4)

```bash
python experiments/eval_omniview_fusion_v5.py \
    --model_class omniview_v5 \
    --checkpoint outputs/ablations/v86_sparse_cross_domain_medium_a800.pth \
    --config outputs/ablations/v86_sparse_cross_domain_medium_a800.config.json \
    --dataset_manifest configs/splits/h36m_true_gt_standard.yaml \
    --eval_subjects S9 S11 \
    --output outputs/eval_v86_sparse_cross_domain_true_gt_h36m_test.json
```

Gate: test MPJPE ≤ 31.56 mm (v25 stability) and PA-MPJPE ≤ 34.35 mm.

### 6.2 Variable-view / sparse-view test (k=2/3/4)

Use the existing variable-view eval scripts with the v86 checkpoint:

```bash
# No-fallback learned-model eval
bash scripts/eval_variable_views_v86_sparse_cross_domain_medium_a800_split_k.sh

# DLT-fallback eval (fallback only if k<4 is still catastrophic)
bash scripts/eval_variable_views_v86_sparse_cross_domain_medium_a800_dlt_fallback.sh
```

Primary gates:
- `MPJPE@k=2` < 100 mm (without DLT fallback)
- `MPJPE@k=3` < 50 mm (without DLT fallback)
- `MPJPE@k=4` ≤ 31.56 mm

### 6.3 Cross-domain test

Evaluate the mixed-dataset checkpoint on H36M true-GT S9/S11:

```bash
python experiments/eval_omniview_fusion_v5.py \
    --checkpoint outputs/ablations/v86_sparse_cross_domain_mixed_medium_a800.pth \
    --dataset_manifest configs/splits/h36m_true_gt_standard.yaml \
    --eval_subjects S9 S11 \
    --output outputs/eval_v86_sparse_cross_domain_mixed_h36m_test.json
```

Gate: test MPJPE ≤ 33.42 mm (v25 mixed baseline) and better PA-MPJPE.

---

## 7. Expected outcomes and decision rules

| Scenario | Interpretation | Next action |
|---|---|---|
| v86 matches v25 on k=4 **and** k<4 MPJPE < 100/50 mm | Sparse-view problem structurally fixed | Run mixed-dataset Stage 2; update paper with MPJPE@k curves |
| v86 helps k<4 but hurts k=4 | Trade-off between robustness and full-view accuracy | Tune `v85_dropout_prob` or `v86_ssv_head_hidden`; consider curriculum dropout |
| v86 count conditioning helps but sparse head does not | Stronger count signal is enough; separate head is overhead | Drop `--use_v86_separate_sparse_view_head`, keep count conditioning |
| Neither v86 component helps v85 | Random dropout alone is insufficient | Investigate v83 view-conditioned temporal attention + dropout |
| Mixed training diverges again | Dataset imbalance or domain gap too large | Reduce AIST++ fraction; add domain-adversarial loss or gradient reversal |

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| v86 sparse head overfits to H36M's specific 4→2/3 view dropout | Use domain-agnostic CamPE; add AIST++ to training |
| Separate sparse head creates a discontinuity at `k=4` boundary | Identity-at-init + smooth gating; validate MPJPE@k curve |
| Stronger count conditioning still leaves k<4 catastrophic | Keep DLT fallback at inference; treat learned k<4 as long-term goal |
| Mixed-dataset training destabilises full-view H36M performance | Warm-start from Stage 1; freeze v86 sparse head during first epochs |
| A800 disk full (99%) | Re-use v85/v86 manifest; run cleanup dry-run before large writes |
| GPU 6/7 occupied by v85 | Queue v86 after v85 eval suite; never use GPUs 0–5 |

---

## 9. Implementation checklist

- [ ] Create `scripts/run_v86_sparse_cross_domain_medium_a800.sh` (Stage 1, H36M true-GT)
- [ ] Create `scripts/run_v86_sparse_cross_domain_mixed_medium_a800.sh` (Stage 2, H36M + AIST++)
- [ ] Create `configs/ablations/v86_sparse_cross_domain_medium_a800.yaml`
- [ ] Create `configs/ablations/v86_sparse_cross_domain_mixed_medium_a800.yaml`
- [ ] Verify v86 modules are wired in `MultiViewGeometryFusionV25.forward`
- [ ] Add variable-view eval scripts for v86
- [ ] Run local smoke (3 epochs, RTX 4090) before A800 launch
- [ ] Schedule A800 runs on GPU 6 or 7 only

---

## 10. Relation to existing v86 ablations

Two narrower v86 ablations are already tracked:

- `docs/experiment_plan_v86.md` — ablates the v85 count embedding (`v86_no_count_embedding`)
- `configs/ablations/v86_strong_count_conditioning_medium_a800.yaml` — enables only `ViewCountConditioningV86`
- `configs/ablations/v86_separate_sparse_view_head_medium_a800.yaml` — enables only `SeparateSparseViewHeadV86`

This design doc supersedes those individual ablations as the **unified v86 target**. The individual ablations remain useful as staged experiments to isolate the contribution of each component; results from them should be merged into this document once available.
