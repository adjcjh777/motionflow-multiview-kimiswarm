# GVHMR / ScoreHMR integration with ray-attention v3

## Task

Extend `experiments/demo_gvhmr_multiview_projection.py` to feed GVHMR
per-view SMPL joints into `RayAttentionFusionModelV3` and compare the
fused multi-view output with the original single-view GVHMR output.

## What changed

- `experiments/demo_gvhmr_multiview_projection.py`
  - Imports and instantiates `RayAttentionFusionModelV3` directly.
  - New CLI flags:
    - `--ray_v3_checkpoint`: optional trained checkpoint for v3.
    - `--max_frames`: quick-test mode that processes only the first N frames.
  - After running the existing plugin registry loop, the script feeds the
    same per-view 2D projections + confidences into `ray_attention_v3`
    (camera-conditioned embeddings + view/joint attention + weighted DLT).
  - Reports MPJPE for every registered plugin plus `ray_attention_v3`,
    and prints an explicit single-view vs multi-view summary.
  - The original GVHMR single-view world joints are kept as the reference,
    so the single-view baseline MPJPE is zero by construction and the
    multi-view methods are measured against it.

## Verification

- `python -m py_compile` on the script succeeds.
- The script imports cleanly in the target environment and exposes the
  expected helpers.
- A targeted test confirmed that `RayAttentionFusionModelV3` forward
  works on the local 4090 with synthetic input.

## Blocker / finding

- Full end-to-end execution of `demo_gvhmr_multiview_projection.py` on the
  local Windows/Anaconda environment exits with code 127 as soon as it hits
  a NumPy matrix multiply (`K @ Rt`, `np.dot`, etc.). This is a
  pre-existing environment/BLAS issue, not a script bug: the original
  script from `HEAD` exhibits the same failure.
- As a result, the end-to-end MPJPE numbers for the new v3 integration were
  not produced on this machine. They can be generated once the NumPy/MKL
  environment is fixed (e.g., in WSL or on the A800-D read-only machine).

## Next steps

- Re-run the extended demo in a healthy environment.
- Train or load a `ray_attention_v3` checkpoint and compare its MPJPE
  against the v1/v2 baselines and the single-view GVHMR reference.
- If available, plug in real ScoreHMR per-view SMPL predictions instead of
  projecting the single global GVHMR output.
