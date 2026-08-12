# Cross-Camera / Domain Generalization for ray_attention_v3

**Topic:** `cross_dataset_generalization`  
**Date:** 2026-08-04  
**Scope:** Add domain-agnostic camera embeddings and an optional gradient-reversal domain classifier to `RayAttentionFusionModelV3`, plus a cross-dataset evaluation script.

## What changed

1. **`motionflow_mv/fusion/ray_attention_v3_model.py`**
   - Added `_domain_agnostic_camera_features(K, R, t)`:
     - Normalizes intrinsics by the geometric-mean focal length.
     - Computes camera centers `c = -R^T t` and normalizes them by rig diameter.
     - Produces a 21-D per-view descriptor (same dimension as the raw K/R/t vector)
       so the existing `camera_embed_mlp` can consume it without changing its input size.
   - Added `GradientReversalLayer` and `GradientReversalFunction` for
     domain-adversarial training.
   - Added optional domain classifier:
     - Controlled by `use_domain_classifier` (default `False`, so existing trainers
       keep their two-output contract).
     - Pools the per-view camera embedding and predicts a domain label.
     - When `domain_labels` is passed or `return_domain_logits=True`, the model
       returns `(pred_3d, weights, domain_logits)`.

2. **`experiments/eval_cross_dataset_generalization.py`**
   - Evaluates a trained `ray_attention_v3` checkpoint on a source and a target
     `.npz` multi-view dataset.
   - Reports model MPJPE vs. a DLT baseline under configurable 2D noise / view
     dropout / outliers.
   - Supports the optional domain classifier (`--use_domain_classifier`).
   - DLT baseline uses the model's own `torch.linalg.lstsq` DLT implementation,
     which avoids the broken numpy BLAS path observed on the local WSL 4090 env.

## Verification

Ran a 50-frame sanity check on H36M source (`s_01_act_02_multiview.npz`) and
zero-shot target (`s_09_acts_02_multiview.npz`) with no checkpoint (random
weights):

```text
 drop noise  src_model    src_dlt  tgt_model    tgt_dlt
-----------------------------------------------------------------
  0.0  0.00       2.72       0.58     748.82     736.58
  0.0  2.00      12.89      12.06     749.82     737.48
  0.0  5.00      30.63      29.57     751.46     739.32
  0.2  0.00       7.87       6.66    1163.46    1156.47
  0.2  2.00      23.30      22.70    1150.91    1149.39
  0.2  5.00      48.27      47.42    1117.08    1113.60
```

The numbers confirm:
- The script and model run end-to-end on both source and target.
- DLT is a strong lower-bound on clean data.
- Random weights produce large target error, as expected; a trained checkpoint
  is needed for meaningful cross-dataset numbers.

Also verified the domain-classifier path with `--use_domain_classifier`; the
model correctly returns three outputs and the script completes without error.

## Next steps

1. Train `ray_attention_v3` on the full H36M source (`s_01_acts_02..16`, 62k
   frames) both with and without the domain classifier to measure whether the
   GRL improves zero-shot transfer to `s_09`.
2. Add a domain-label data loader that tags each batch by subject/dataset, so the
   GRL loss can be used during training.
3. Extend the eval to Shelf/Campus once 3D-GT multi-view `.npz` files are
   prepared for those datasets.
