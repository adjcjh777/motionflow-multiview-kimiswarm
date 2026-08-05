# Direction 6: Multi-scale / multi-resolution spatial features

## Problem statement

The current best model, `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, processes each joint at a single resolution. Distal joints (wrists, ankles) are harder to localize and may benefit from a coarse-to-fine representation over the joint dimension. A reusable `SpatialFeaturePyramid` already exists in `motionflow_mv/models/spatial_feature_pyramid.py`, but it is not yet connected to the principal-point (PP) model that holds the best MPI-INF-3DHP result (9.32 mm). The next step is therefore to insert the pyramid between the per-frame feature extractor and the spatio-temporal transformer and verify that it preserves the forward and backward passes.

## Simplest concrete next step

1. Add a `SpatialFeaturePyramid(d, d, num_scales=3)` submodule to the PP model.
2. After `_extract_frame_features` returns `(B*T, V, J, d)`, reshape to `(B, T, V, J, d)`, run the pyramid, and reshape back.
3. Run a CPU-only smoke test that confirms shapes and gradients are correct before any GPU training is queued.

## Files to touch

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`
  - Import `SpatialFeaturePyramid`.
  - In `__init__`, add `self.sfp = SpatialFeaturePyramid(in_channels=d, out_channels=d, num_scales=3)`.
  - In `forward`, after `feat = self._extract_frame_features(...)`, insert:

```python
feat = feat.view(B, T, V, J, self.d)
feat = self.sfp(feat)
feat = feat.view(B * T, V, J, self.d)
```

- `motionflow_mv/fusion/__init__.py` (if the model is exported) — no breaking change, only a new optional flag.
- `scripts/run_crossview_residual_spatial_pyramid_wsl.sh` — GPU launcher skeleton (created separately if needed).
- `docs/swarm_iter7/ablate_multiscale_spatial_pp.py` — CPU smoke script (already created).

## Rough diff / sketch

```diff
--- a/motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py
+++ b/motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py
@@ -10,6 +10,7 @@ import torch.nn as nn
 
 from .principal_point_correction import PrincipalPointCorrection
+from motionflow_mv.models.spatial_feature_pyramid import SpatialFeaturePyramid
 from .ray_attention_temporal_crossview_residual_model import (
     RayAttentionFusionModelTemporalCrossviewResidual,
 )
@@ -55,6 +56,7 @@ class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(...):
         self.principal_point_correction = PrincipalPointCorrection(
             d=d, hidden=principal_point_hidden, max_offset=principal_point_max_offset, max_focal_scale=focal_max_scale,
         )
+        self.sfp = SpatialFeaturePyramid(in_channels=d, out_channels=d, num_scales=3)
 
     def forward(self, x, cameras=None, K=None, R=None, t=None):
         ...
@@ -112,6 +114,9 @@ class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(...):
         # Per-frame v3 features (uses corrected intrinsics).
         feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)
+        feat = feat.view(B, T, V, J, self.d)
+        feat = self.sfp(feat)
+        feat = feat.view(B * T, V, J, self.d)
 
         # Spatio-temporal (time + view) attention.
         feat = feat.view(B, T, V, J, self.d)
```

## CPU smoke test

The script `docs/swarm_iter7/ablate_multiscale_spatial_pp.py` performs two checks:

1. Verifies that `SpatialFeaturePyramid` can consume the per-frame features of the PP model and that gradients reach its parameters.
2. Provides a complete subclass, `SpatialPyramidPPModel`, that wires the pyramid into the full forward/backward pass.

### Command

```bash
KMP_DUPLICATE_LIB_OK=TRUE python docs/swarm_iter7/ablate_multiscale_spatial_pp.py
```

### Result

```text
SpatialFeaturePyramid insertion sanity test passed
  Input x shape:          (2, 1, 4, 17, 3)
  Per-frame features:     (2, 4, 17, 64)
  After pyramid:          (2, 4, 17, 64)

Subclassed SpatialPyramidPPModel forward/backward test passed
  Prediction shape: (2, 1, 17, 3)
  Weight shape:     (2, 1, 4, 17)
```

Both checks passed, confirming that the pyramid can be inserted at the `(B, T, V, J, d)` representation without altering downstream shapes and that gradients flow into the new module.

## Expected success metric

- **CPU smoke:** forward and backward pass succeed; output shapes match baseline. ✅ Done.
- **GPU smoke (10 epochs, when GPU is free):** clean MPJPE does not regress beyond the baseline ( 9.32 mm on MPI-INF-3DHP). This can be evaluated with a tiny `--max_batches` smoke.
- **Full run:** clean MPJPE ≤ 9.3 mm with ≥ 2% relative improvement on distal-joint MPJPE (wrist/ankle) versus the current best PP checkpoint.

## Resource requirement

- **This step:** CPU-only, no training, < 5 s runtime.
- **Next step:** GPU required for the 10-epoch smoke and the full MPI-INF-3DHP run. Do not run until the RTX 4090 is free.

## GPU launcher skeleton (do not run yet)

```bash
#!/usr/bin/env bash
# scripts/run_crossview_residual_spatial_pyramid_wsl.sh
set -e
python -m experiments.train_crossview_residual_principal_point \
    --config configs/train_ray_attention_reproducible.yaml \
    --spatial_pyramid True \
    --num_scales 3 \
    --warm_start checkpoints/ray_attention_temporal_crossview_residual_principal_point_best.pt \
    --epochs 50 \
    --batch_size 4 \
    --device cuda
```

The launcher is intentionally a skeleton; actual flags should match the existing `experiments/train_crossview_residual_principal_point.py` interface.
