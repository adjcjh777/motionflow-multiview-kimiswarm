# Temporal Ray-Attention Fusion

## What was built

A new model `RayAttentionFusionModelTemporal` lives in
`motionflow_mv/fusion/ray_attention_temporal_model.py`.  It extends
`RayAttentionFusionModelV3` with a temporal transformer over frames, while
keeping the same geometric inductive bias: the network still predicts
per-view-per-joint weights and feeds them into a differentiable weighted DLT
layer.

## Architecture

The model expects input of shape `(B, T, V, J, 3)` (or `(B, V, J, 3)` for
single-frame inference) and produces:

- `pred_3d`: `(B, T, J, 3)` (or `(B, J, 3)`)
- `weights`: `(B, T, V, J)` (or `(B, V, J)`)

The processing flow is:

1. **Per-frame v3 encoder** — observation + ray embeddings, camera-conditioned
   embeddings, view-level self-attention, joint-level self-attention.  This is
   identical to `ray_attention_v3_model.py` and is applied independently to each
   frame.
2. **Temporal transformer** — the per-frame feature grid `(V, J, d)` is treated
   as a sequence of `V*J` tokens; self-attention is applied over the `T`
   frames.  A learned temporal position embedding encodes frame order.
3. **Per-frame weighted DLT** — the temporally-refined features are used to
   predict per-view weights and triangulate 3D joints for each frame.

All changes are local to the new file; no existing code was modified.

## Verification

A synthetic forward/backward sanity check is in `tests/test_ray_attention_temporal.py`:

```bash
D:/anaconda3/envs/jz_py310/python -m pytest tests/test_ray_attention_temporal.py -v
```

On the local WSL 4090 setup this printed:

```text
============================== 2 passed in 3.06s ==============================
```

The model:
- Produces the expected output shapes for both 5-frame clips and single-frame
  inputs.
- Allows gradients to flow through the temporal transformer.
- Falls back gracefully when a 4D input is passed.

## Open items / next steps

- The model has not been trained yet; a training script analogous to
  `experiments/train_ray_attention_v3_h36m.py` should be added that samples
  `(B, T, V, J, 3)` clips from the existing Human3.6M multi-view sequences.
- A `RayAttentionTemporalFusionModule` plugin wrapper can be added once the
  training checkpoint is available.
- A lightweight temporal baseline (e.g. averaging v3 predictions over a sliding
  window) would help quantify the gain of the learned temporal transformer.
