# Design: Camera Positional Encoding v2 for the Main Residual Model

## 1. Motivation

The main residual model (`RayAttentionFusionModelTemporalResidual`) currently learns camera embeddings from a concatenation of raw intrinsics and extrinsics (`K, R, t`).  This embedding is dataset-specific: it fixes the number of views, assumes a fixed view ordering, and carries no explicit geometric meaning.  The first CamPE variant (`ray_attention_temporal_residual_campe_model.py`) replaced this MLP with `CameraPositionalEncoding`, but it zeroed out the camera-center part of the ray embedding, leaving geometry on the table.

**CamPE v2** integrates the geometry-based camera positional encoding more cleanly into the main residual model while keeping the full ray embedding and removing an unused, view-count-dependent layer.

## 2. Design decisions

### 2.1 Keep the full ray embedding

The base residual model builds ray embeddings from both camera centres and ray directions:

```
ray_input = [c_v, r_v]  in R^6
```

CamPE v2 preserves this (the v1 variant used a zero centre placeholder for compatibility reasons).  Keeping the centres gives the per-frame encoder stronger geometric context.

### 2.2 Replace camera embedding with `CameraPositionalEncoding`

The constructor now instantiates:

```python
self.camera_embed_mlp = CameraPositionalEncoding(d=d, n_bands=n_bands)
```

`CameraPositionalEncoding` derives per-view tokens from:

```
c_v = -R_v^T t_v                     # camera centre in world
r_v = R_v^T [0, 0, 1]^T              # principal ray direction in world
f_v = (f_x + f_y) / 2                # mean focal length
```

It normalises camera centres by rig diameter, normalises focal length by the mean focal length, applies Fourier sinusoidal bands, and projects the result to `d` dimensions.  This makes the encoding invariant to absolute scene scale and image resolution, and lets the same checkpoint run on rigs with different numbers of views.

### 2.3 Remove the unused `fusion_mlp`

The base residual model inherits a `fusion_mlp` that depends on `n_views` via `d * n_views`.  It is never used in the residual forward pass.  CamPE v2 deletes it, so the model is genuinely variable-view.

### 2.4 Re-use the base temporal+residual forward

Only `_extract_frame_features` is overridden.  The temporal transformer, weight head, DLT triangulation, and residual refinement head are reused unchanged from `RayAttentionFusionModelTemporalResidual`.  This keeps the change minimal and preserves checkpoint compatibility for every parameter except the old camera-embedding MLP.

## 3. File paths

* `motionflow_mv/fusion/ray_attention_temporal_residual_campe_v2_model.py` — new model class.
* `motionflow_mv/fusion/camera_positional_encoding.py` — reusable CamPE module (already existed).
* `tests/test_ray_attention_temporal_residual_campe_v2.py` — forward/backward/variable-view smoke tests.

## 4. Validation

Run the smoke tests:

```bash
python -m pytest tests/test_ray_attention_temporal_residual_campe_v2.py -v
```

The tests check:

* forward/backward pass on 4-view clips;
* single-frame (4D input) compatibility;
* variable-view inference: the same model runs on 4-view and 14-view rigs;
* iterative refinement (`n_iter=3`).

All four tests pass on CPU in a few seconds.

## 5. Expected impact

* **Variable view count:** the model can be evaluated on any number of views without retraining or shape errors.
* **Cross-dataset transfer:** CamPE is invariant to scene scale and focal length, so the same checkpoint can be reused across MPI-INF-3DHP, Human3.6M, and AIST++.
* **Stronger geometry:** restoring camera centres in the ray embedding gives the encoder a better inductive bias than the zero-placeholder v1.
* **Minimal diff:** only the camera-embedding parameters change; all other weights remain compatible with the main residual model.

## 6. Blockers / next steps

* The new model needs a short training run on MPI-INF-3DHP to confirm it matches or beats the 10.46 mm baseline.  No long training was run here.
* A small training script mirroring `train_ray_attention_temporal_residual_campe_mpiinf3dhp.py` for the v2 class would make experiments easier.
