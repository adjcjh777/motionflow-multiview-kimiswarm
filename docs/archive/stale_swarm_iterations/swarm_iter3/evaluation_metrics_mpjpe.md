Done. Wrote the research report to:

`docs/swarm_iter3/evaluation_metrics_mpjpe.md`

Key takeaways:

- **Problem:** The project currently relies on 2D reprojection error, which cannot validate true 3D accuracy. MPJPE/PA-MPJPE are the missing publication-quality metrics.
- **Codebase state:** `motionflow_mv/eval/metrics.py` already has basic MPJPE, PA-MPJPE, and PCK, but real-dataset scripts only report reprojection. Unit handling, MRPE, MPVE, AUC, and per-joint breakdowns are absent.
- **Recommendations:** Extend `metrics.py` with root-centering, MRPE, MPVE, AUC, per-joint error, and bone-length/temporal consistency; create an H36M/Shelf/Campus 3D evaluation harness; switch training to real 3D GT loss; lead paper claims with MPJPE/PA-MPJPE on H36M plus MRPE for the ICRA robotics angle.
- **Risks:** Ground-truth data access, Procrustes convention, root-joint mapping across skeletons, and meter/millimetre unit confusion.

No code files were modified; only the research report was added.