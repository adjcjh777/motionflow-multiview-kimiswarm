# Cross-Dataset Generalization Roadmap — Iter 11+

## 1. Current State

The project now has a top-line fusion model, `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` (`motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`), that stacks cross-view spatio-temporal attention, uncertainty-weighted DLT, a differentiable Gauss-Newton triangulation head, and a residual MLP. The current best MPI-INF-3DHP validation MPJPE is ~11.17 mm (cross-view residual model, `RayAttentionFusionModelTemporalCrossviewResidual`), and a faster temporal-residual + reprojection run only reached 47.54 mm because of limited data/epochs. H36M WebBridge conversion is in progress at `data/webbridge/h36m`.

The WebBridge loader already supports Human3.6M, MPI-INF-3DHP, AIST++, Shelf/Campus, 3DPW (pseudo-GT), and CMU Panoptic (stub) in a canonical `(T, V, J, 3)` format. Evaluation is fragmented: `experiments/eval_cross_dataset_generalization.py` only tests the older `RayAttentionFusionModelV3`, and `experiments/eval_all_datasets.py` only loads `ray_attention_v3` / `ray_attention_v2` checkpoints. There is no unified cross-dataset training or evaluation harness for the new advanced model.

## 2. Concrete, Implementable Improvements

### 2.1 Unified Multi-Dataset Training with Domain-Adaptive Heads

**Problem.** Models trained only on MPI-INF-3DHP learn camera intrinsics, skeleton scales, and motion distributions specific to that dataset. When evaluated on H36M or 3DPW, MPJPE degrades because the feature distribution shifts.

**Solution.** Add a domain classifier branch on the pooled spatio-temporal features and use gradient reversal (Ganin & Lempitsky, 2015). During training, minimize 3D pose loss while maximizing domain-classification loss through the reversal layer. This forces the shared encoder to learn dataset-invariant features.

**Implementation steps.**

1. Add a small domain head in the advanced model:
   - Input: pooled feature `feat_pooled` (`B*T, J, d`).
   - Output: logits over `N` dataset domains.
2. Train with a batch that mixes samples from all available datasets.
3. Use `ReverseLayerF` (gradient sign flipped during backprop) between the encoder and the domain head.
4. Keep the existing 3D MSE + reprojection + uncertainty NLL losses.

**Expected impact.** Reduces source-target gap; improves zero-shot transfer from MPI-INF-3DHP to H36M and 3DPW.

### 2.2 Cross-Dataset Joint Canonicalization and Bone-Length Consistency

**Problem.** Datasets use different skeleton definitions (H36M: 32 joints, MPI-INF-3DHP / COCO: 17 joints). A model that memorizes joint indices overfits to one skeleton layout.

**Solution.** Map every dataset to a canonical 17-joint COCO-compatible skeleton at load time and add a bone-length consistency loss that is invariant to absolute scale.

**Implementation steps.**

1. Maintain a per-dataset joint mapping table in `motionflow_mv/data/skeleton_maps.py`.
2. In the training collate function, remap joints to canonical indices; drop joints that have no mapping.
3. Add a bone-length loss:

```
L_bone = mean_joints | ||pred_j - pred_parent(j)||_2 - b_j* ||_1
```

where `b_j*` is the expected bone length estimated from the training set (or a fixed prior).

**Expected impact.** Improves generalization to datasets with different skeleton conventions and reduces scale drift.

### 2.3 Geometry-Normalized Camera Conditioning

**Problem.** The current `camera_embed_mlp` concatenates raw flattened `K`, `R`, and `t`, which embeds absolute focal length and camera distance. Datasets have very different camera rigs.

**Solution.** Normalize camera inputs before the embedding:

- Normalize intrinsics: `K_norm = diag(1/w, 1/h, 1) * K`.
- Normalize extrinsics by scene scale: compute the mean camera distance to the subject and divide `t` by it.
- Feed the normalized 21-D vector plus the scale factor to a learned embedding.

**Implementation steps.**

1. In `ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model.py`, replace the raw `camera_embed_mlp` input with a normalized vector.
2. Pass the per-sample scene scale to the loss so 3D predictions can be rescaled if needed.

**Expected impact.** Reduces domain shift caused by different capture rigs (e.g., H36M 4 HD cameras vs. MPI-INF-3DHP 14 calibrated cameras).

### 2.4 Test-Time Adaptation on Target Sequences

**Problem.** Even a generalizable model can benefit from a few gradient steps on the target sequence before evaluation, because the target cameras, subject scale, and motion are fixed within a sequence.

**Solution.** Implement a TTA mode that freezes the encoder and updates only the residual MLP and triangulation head for a few iterations on the target clip, using self-supervised reprojection loss and bone-length smoothness as objectives.

**Implementation steps.**

1. Add `model.adapt_target(loader, steps=50, lr=1e-4)` that optimizes only the residual MLP and Gauss-Newton initialization.
2. Loss: `L = L_reproj + lambda_smooth * L_temporal_smoothness`.
3. Run before evaluation on each target sequence.

**Expected impact.** Can close the gap between source-trained and target-tuned MPJPE by 10–20% on new datasets.

### 2.5 Mixed-Source Pseudo-Label Curriculum

**Problem.** 3DPW has pseudo-GT (`*_pseudo.npz`) but may be noisy. Shelf/Campus have limited 3D labels.

**Solution.** Train with a curriculum: start on clean H36M/MPI-INF-3DHP real 3D GT, then mix in 3DPW pseudo labels with low weight and uncertainty-based filtering (only use pseudo labels whose predicted uncertainty is below a threshold).

## 3. Recommended Experiments and Metrics

### Experiments

1. **Baseline.** Train `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1` on MPI-INF-3DHP only; evaluate on H36M, 3DPW, and AIST++.
2. **Mixed training.** Train the same architecture on a mixed batch of MPI-INF-3DHP + H36M with per-dataset domain labels.
3. **+ Domain adversarial.** Add gradient-reversal domain head on top of mixed training.
4. **+ Camera normalization.** Add geometry-normalized camera embeddings.
5. **+ Bone-length loss.** Add skeleton-consistent bone-length regularizer.
6. **+ TTA.** Adapt on each target sequence before evaluation.
7. **Curriculum pseudo-labeling.** Gradually add 3DPW pseudo labels with uncertainty filtering.

### Metrics to Track

- **MPJPE** and **PA-MPJPE** in mm (primary).
- **PCK@50/100/150 mm** and **AUC**.
- **Cross-dataset delta**: `MPJPE_target - MPJPE_source` to measure generalization gap.
- **Per-joint breakdown** from `motionflow_mv/eval/metrics.py` to identify which joints fail on which dataset.
- **Domain classification accuracy** (diagnostic; should stay near chance if domain adaptation is working).
- **Reprojection error** in pixels for datasets without 3D GT.

## 4. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Different units (mm vs cm) | Convert all datasets to meters in the WebBridge loader; infer unit from GT magnitude. |
| Skeleton mismatch | Canonicalize to 17 joints; drop unmapped joints. |
| Domain classifier dominates | Use small domain-head weight; keep pose losses primary. |
| Pseudo-label noise | Filter by predicted uncertainty; start with small weight and ramp up. |
| Longer training time | Use mixed batches only; no need to pre-train separately per dataset. |

## 5. Code Sketch: Domain-Adaptive Training Loop

```python
# motionflow_mv/fusion/domain_adaptive_model.py
import torch
import torch.nn as nn

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainAdaptiveFusionModel(RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1):
    def __init__(self, n_domains=4, lambda_domain=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_domains = n_domains
        self.lambda_domain = lambda_domain
        self.domain_head = nn.Sequential(
            nn.Linear(self.d, self.d // 2),
            nn.ReLU(),
            nn.Linear(self.d // 2, n_domains),
        )

    def forward(self, x, domain_labels=None, alpha=1.0, **kwargs):
        pred_3d, weights, log_var, nll = super().forward(x, **kwargs)
        # feat_pooled would be exposed from the parent forward or cached
        # For the sketch, assume we expose it via _forward_features
        feat = self._forward_features(x, **kwargs)  # (B*T, J, d)
        feat_flat = feat.mean(dim=1)  # (B*T, d)
        reversed_feat = GradientReversalFn.apply(feat_flat, alpha)
        domain_logits = self.domain_head(reversed_feat)
        return pred_3d, weights, log_var, nll, domain_logits
```

Training loop snippet:

```python
for xb, yb, K, R, t, domain_ids in mixed_loader:
    optimizer.zero_grad()
    pred, weights, log_var, nll, domain_logits = model(
        xb, K=K, R=R, t=t, domain_labels=domain_ids, alpha=alpha
    )
    pose_loss = mse(pred, yb)
    domain_loss = ce(domain_logits, domain_ids)
    loss = pose_loss + reproj_weight * reproj_loss + nll + lambda_domain * domain_loss
    loss.backward()
    optimizer.step()
```

## 6. Next Steps

1. Implement the domain-adaptive wrapper and expose `feat_pooled` from the advanced model.
2. Create `experiments/train_cross_dataset.py` that consumes a YAML list of source/target `.npz` files and builds mixed batches.
3. Add camera-normalization utilities to `motionflow_mv/data/` and joint canonicalization maps.
4. Run the experiment ladder (Sec. 3) and populate `docs/swarm_iter11/` with per-experiment results.
5. Update `experiments/eval_cross_dataset_generalization.py` to load the advanced model and report the metrics in Sec. 3.
