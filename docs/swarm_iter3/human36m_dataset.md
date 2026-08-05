Done. I researched Human3.6M in the context of the motionflow-multiview codebase and wrote a focused technical report to `docs/swarm_iter3/human36m_dataset.md` (224 lines, ~2–4 pages).

**Key takeaways in the report:**
- **Problem:** The project needs real 3D ground truth to move from reprojection-only/pseudo-supervised training to 3D-supervised training. Current learned plugins match DLT on Shelf but do not beat it.
- **Why Human3.6M:** It is the canonical large-scale indoor multi-view 3D pose benchmark with 3.6M frames, 4 calibrated cameras, and world-coordinate 3D joint positions—exactly the supervision needed to train/validate `attention`, `robust_triangulation`, `residual_refiner`, and `temporal_refiner`.
- **Relation to codebase:** The existing `FusionModule` interface `(T, V, J, 2) + confidences + cameras -> (T, J, 3)` already fits Human3.6M, but there is no H36M loader or evaluation script yet.
- **Recommendations:** 
  1. Acquire/register H36M.
  2. Implement `motionflow_mv/data/human36m_loader.py`.
  3. Add `experiments/eval_all_plugins_h36m.py` with MPJPE/PA-MPJPE.
  4. Train the existing plugins with a combined 3D + reprojection + bone-length + temporal loss.
  5. Use H36M as the primary 3D-supervised benchmark for CVPR/ICRA 2027.
- **Risks:** Dataset access/licensing, storage/compute, skeleton mapping (32 joints → 17 COCO-style), scale (mm → m), and the possibility that learned plugins still do not beat DLT.