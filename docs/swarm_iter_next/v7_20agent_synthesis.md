# Swarm Iteration Next — v7 20-Agent Synthesis

> Tracking issue: #84 (v6 experiments) / #85 (v7 design)  
> Last updated: 2026-08-07  
> Goal: ICRA/CVPR 2027 — beat the v4 baseline (~25.29 mm) with a monotonic variable-view curve.

## 1. Current status at a glance

| Metric | v4 baseline | v5 ablation | v6 (training) | Target |
|--------|-------------|-------------|---------------|--------|
| Clean MPJPE | **25.29 mm** | 28.85 mm | TBD | < 25.0 mm |
| Variable-view k=2..13 | ~25.3 mm | ~25.3 mm | TBD | monotonic ↓ |
| Full-view k=14 | 25.29 mm | 28.9 mm | TBD | no regression |

- v6 is training on A800-D: `v6_mpi_isab`, `v6_mpi_perceiver`, `v6_h36m_isab` (30 epochs, 10k samples/epoch).
- The mixed-loader fix for `dataset_id` has been committed to `main` and synced to A800-D; smoke is running.

## 2. 20-agent swarm — high-level themes

The swarm produced proposals across four groups:

1. **Architecture / attention** (5 proposals): hierarchical temporal pyramid, learnable epipolar ray bias, factorised FlashAttention, graph-skeleton-view GNN, lightweight factorised T-V-J block.
2. **Training / data scale** (5 proposals): SSL masked-view pretraining, curriculum variable-view training, large-scale mixed-dataset training, synthetic augmentation mixer, distillation from a single-view teacher.
3. **Geometry & uncertainty** (5 proposals): full anisotropic precision DLT, uncertainty-gated aggregation, learned camera/view embedding, multi-task 2D/3D/camera/visibility head, Gaussian splatting / neural rendering consistency.
4. **Evaluation & deployment** (5 proposals): TTA per sequence, PA-MPJPE/PCK/AUC metrics, active view selection, cross-dataset robustness protocol, real-time lightweight deployment.

## 3. Prioritized v7 directions (ranked by impact and feasibility)

### 3.1 Mixed-dataset domain-aware training (highest ROI)

*Why:* v5/v6 are trained mostly on single datasets. WebBridge + H36M + MPI-INF-3DHP together give heterogeneous rigs, skeletons and noise characteristics. Domain-aware mixing should improve generalization and make use of the existing `WebBridgeMixedDataset`.

*What:*
- Add a `domain_id` embedding / small FiLM layer after the ST transformer, using the `dataset_id` already returned by the mixed loader.
- Add per-dataset view-count masking so H36M (4 views) and MPI (14 views) use their true active counts instead of fixed 14-view padding.
- Stage the mixing with a curriculum: early epochs train mostly within one domain, later epochs mix all domains evenly.

*Success metric:* mixed v7 training reaches clean MPJPE ≤ v4 baseline on both H36M and MPI-INF-3DHP; variable-view curve is monotonic.

### 3.2 Full anisotropic precision DLT + uncertainty gating (high ROI, low risk)

*Why:* v6 predicts a 2×2 image covariance `L` per view/joint but the DLT path still uses only scalar precision. Using the full precision matrix is the statistically optimal Gaussian triangulation weight.

*What:*
- Replace the scalar precision weight in `triangulation.py:triangulate_dlt_batched_lstsq` with the full 2×2 precision `Λ = (L L^T)^{-1}` (reuse `uncertainty_weighted_triangulation.py`).
- Gate the ISAB/Perceiver aggregator attention by the per-view uncertainty, so noisy views contribute less to the fused features.

*Success metric:* clean MPJPE drops on MPI-INF-3DHP by ≥ 0.5 mm with no regression on H36M.

### 3.3 Curriculum variable-view training (high ROI)

*Why:* v6 already samples random view subsets, but the minimum k is fixed. Raising the minimum k over epochs forces the model to handle harder subsets progressively and should improve monotonicity.

*What:*
- Replace uniform `k ~ Uniform(min, max)` with an epoch-dependent minimum: `k_min(e) = max(min_views, ceil(min_views + (V - min_views) * (e/E)^α))`.
- Add a cardinality-conditioned view embedding so the aggregator knows how many views are active.

*Success metric:* MPJPE@k is monotonic for k = 2..V; the full-view regression seen in v5 disappears.

### 3.4 Kinematic bone-length + PA training objective (medium ROI)

*Why:* Bone-length constraints reduce anatomical drift; PA-MPJPE is the standard pose-evaluation metric for ICRA/CVPR.

*What:*
- Add a soft bone-length preserving projection (BPP) after the residual head.
- Optionally train with a differentiable Procrustes alignment loss in addition to the world-space MSE.

*Success metric:* PA-MPJPE improves without hurting world-space MPJPE; temporal stability (velocity/acceleration) is maintained.

## 4. Proposed next exact training run

**Run name:** `v7_mixed_domain_precision_curriculum`

**Goal:** Beat the v4 baseline by combining mixed-dataset domain-aware training, full precision DLT, and curriculum variable-view training.

### Configuration

```yaml
model:
  base: omniview_fusion_v6
  use_mixed_loader: true
  use_full_precision_dlt: true
  use_uncertainty_gated_aggregator: true
  use_cardinality_embedding: true
  use_domain_film: true

training:
  mixed_manifest: configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml
  epochs: 60
  batch_size: 16
  train_samples: 10000
  clip_len: 13
  lr: 1e-3
  lr_cosine: true
  lr_warmup_epochs: 3
  variable_view_min_views: 2
  variable_view_max_views: 14
  variable_view_curriculum_alpha: 2.0
  monotonic_loss_weight: 0.1
  domain_loss_weight: 0.01
  warm_start: outputs/omniview_fusion_v6_mpi_isab.pth  # best v6 checkpoint
  freeze_encoder_epochs: 5

hardware:
  host: a800-D
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]
  tmux_session: v7_mixed_domain_precision_curriculum
```

### Launch steps

1. Wait for v6 first-epoch results and pick the best checkpoint (ISAB or Perceiver).
2. Implement and test the three v7 components with CPU smoke.
3. Launch on A800-D using tmux + `scripts/a800_session_monitor.sh`.
4. After the first 5 epochs, run eval on MPI-INF-3DHP and H36M clean + variable-view.
5. If clean MPJPE ≤ v4 baseline, run the full robustness matrix and variable-view curve.
6. Post results to issue #85 and open a PR to merge v7 into `main`.

## 5. Fallback plan

If the mixed training destabilises:
1. Disable `use_domain_film` and train domain-agnostic mixed loader.
2. Reduce `domain_loss_weight` to 0.001 or remove it.
3. Disable curriculum (fixed `k ~ Uniform(2, V)`).
4. Warm-start only from the v6 same-dataset checkpoint, not across datasets.

## 6. Open questions

1. Which v6 checkpoint is best: MPI-ISAB, MPI-Perceiver, or H36M-ISAB?
2. Does full precision DLT fit the existing batched triangulation without a major refactor?
3. Will the mixed loader scale to 14-view MPI and 4-view H36M with the same skeleton mapping?
4. Is the A800-D venv read-only dependency set stable for the new `torch.linalg` calls needed by precision DLT?

## 7. Related files

- Mixed loader: `motionflow_mv/data/webbridge_mixed_dataset.py`
- v6 model: `motionflow_mv/fusion/omniview_fusion_v5.py`
- v6 trainer: `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- Triangulation: `motionflow_mv/fusion/triangulation.py`
- Uncertainty triangulation: `motionflow_mv/fusion/uncertainty_weighted_triangulation.py`
- Domain adaptation: `motionflow_mv/models/domain_adaptation_wrapper.py`
- Synthesis: `docs/swarm_iter_next/v7_20agent_synthesis.md`
