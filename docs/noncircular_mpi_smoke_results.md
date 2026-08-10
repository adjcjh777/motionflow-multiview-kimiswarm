# Non-Circular MPI-INF-3DHP Smoke Results

**Date:** 2026-08-10  
**Protocol:** MPI-INF-3DHP only (true 3D GT, no H36M circular labels)  
**Manifest:** `configs/splits/mpi_only_noncircular_smoke.yaml`  
**Smoke setting:** 3 train files, 128 samples, 3 epochs, batch_size 4 (2 for v25), RTX 4090

## Results

| Model | Best val MPJPE (mm) | Notes |
|---|---|---|
| DLT baseline | **23.79** | Geometric lower bound; non-learned triangulation |
| v25 geometry fusion | **26.15** | Closest to DLT, consistent with old circular-label dominance |
| v57 DC-PSC (128 samples, 3 ep) | **33.26** | Domain-conditional physical-space calibration |
| v57 DC-PSC (512 samples, 5 ep) | **33.96** | More training did not close the gap to DLT/v25 |
| v46 SVG | **34.94** | Sparse-view generalization |
| v80 VRBT (new) | **35.22** | Learned view-reliability before triangulation; needs more tuning |

## Interpretation

1. **Old H36M leaderboard was measuring DLT-proximity, not pose accuracy.** On true 3D GT, the geometric DLT baseline is the strongest single number; v25 is only ~2.4 mm behind because it contains an explicit weighted DLT layer.
2. **Complex modules do not automatically win.** v57/v46/v80 all underperform v25 and DLT on this tiny smoke. This does not mean they are useless—it means they need longer training, more data, and a sparse-view/cross-domain evaluation to show their value.
3. **The paper story must be robustness, not absolute MPJPE.** The new contribution should be MPJPE@k for small k, cross-dataset transfer, and resilience to noisy/occluded views.

## Next steps

- Run larger non-circular smokes (full small configs) to see if the gap closes.
- Evaluate all models on `eval_variable_views.py` with `MPJPE@k` for k={2,3,4,14}.
- Generate true detected 2D for MPI-INF-3DHP or obtain H36M true 3D GT.
- Tune v80 identity initialization and loss weight before deciding keep/drop.
