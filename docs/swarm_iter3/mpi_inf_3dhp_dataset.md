Done. Wrote a technical report to `docs/swarm_iter3/mpi_inf_3dhp_dataset.md` (141 lines).

**Summary of findings:**

- **Why MPI-INF-3DHP matters for the project:** It supplies the large-scale, in-the-wild, calibrated multi-view 3D ground truth that `docs/design_v3.md` identifies as missing. The current pipeline is trained/evaluated only on synthetic data and the small Shelf/Campus lab scenes, which limits the learned `attention` / `residual_refiner` plugins.
- **Fit with existing code:** The dataset maps cleanly onto `Camera`/`projection_matrix`, `VoxelPoseLoader`, `AttentionFusionV2Module` (already consumes projection matrices), and `HumanMotionIR`. `metrics.py` just needs an AUC/PCK@150 helper.
- **Key recommendations:**
  1. Download MPI-INF-3DHP and implement `MPI3DHPLoader` in `motionflow_mv/data/`.
  2. Add PCK@150 mm and AUC metrics to `motionflow_mv/eval/metrics.py`.
  3. Train `attention_v2` / `residual_refiner` with real 3D GT (MPJPE loss) and scale-aware augmentation.
  4. Run cross-dataset validation (train 3DHP, validate Shelf/Campus) to address the generalization gap.
  5. Use it for the GVHMR multi-view projection demo and for SMPL-world evaluation.
- **Main risks:** license/research-use terms, coordinate-frame (m vs. mm, y-up vs. z-up), 14 vs. 17 joints, outdoor occlusion, and storage/bandwidth.

The report also cites five key works (Mehta et al. 3DV 2017, Iskakov et al. ICCV 2019, Tu et al. ECCV 2020, recent multi-view transformers, and GVHMR).