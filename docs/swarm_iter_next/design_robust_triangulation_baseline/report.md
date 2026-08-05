# Design: IRLS/Charbonnier Robust Triangulation Baseline

## 1. Motivation

The repository already contains a learned `RobustTriangulationModel` (`motionflow_mv/fusion/robust_triangulation.py`) that predicts per-view weights with a small transformer and solves a differentiable weighted DLT system. While powerful, that model has trainable parameters and needs data/optimization. There is no parameter-free geometric baseline that explicitly models robustness to outliers (occluded or badly localized views).

This task fills that gap: implement a classical iteratively reweighted least-squares (IRLS) triangulation baseline using a Charbonnier robust loss, wrapped as a `FusionModule` so it plugs into the same evaluation harness as DLT and the learned attention/ray-attention models.

## 2. Algorithm

For each joint independently:

1. **Initialize** view weights as the supplied detector confidences:
   \[
   w_v^{(0)} = c_v
   \]
2. **Solve** a confidence/robust weighted DLT system to obtain a 3D point
   \(\mathbf{X}\):
   \[
   \mathbf{A}(\mathbf{X}) = 0, \qquad \mathbf{A} = \{ w_v \, (x_v \mathbf{P}_{v,2} - \mathbf{P}_{v,0}),\; w_v \, (y_v \mathbf{P}_{v,2} - \mathbf{P}_{v,1}) \}_v
   \]
3. **Reproject** \(\mathbf{X}\) into every view and compute residuals
   \[
   r_v = \| \text{proj}_v(\mathbf{X}) - (x_v, y_v) \|_2
   \]
4. **Reweight** with the inverse Charbonnier loss
   \[
   w_v^{(t+1)} = \frac{c_v}{\sqrt{r_v^2 + \varepsilon^2}}
   \]
5. Repeat steps 2-4 for a fixed number of IRLS iterations (default 5).

The final triangulated point is returned. The algorithm is deterministic, has no learned parameters, and naturally down-weights views with large reprojection residuals.

## 3. Files created

| File | Purpose |
|------|---------|
| `motionflow_mv/fusion/robust_triangulation_baseline.py` | Core `triangulate_irls(...)` routine and private helpers. |
| `motionflow_mv/fusion/robust_triangulation_baseline_module.py` | `RobustTriangulationBaselineFusion` `FusionModule` wrapper and registration helper. |
| `experiments/eval_robust_triangulation_baseline.py` | Smoke-evaluation script on synthetic multi-view data with/without injected outliers. |
| `docs/swarm_iter_next/design_robust_triangulation_baseline/report.md` | This design report. |

## 4. Implementation notes

- The wrapper obeys the standard `FusionModule` contract (`fuse(points_2d, confidences, cameras)`), so it can be dropped into `MultiViewAdapter` and the global `FUSION_REGISTRY`.
- The linear-algebra backend uses `torch.linalg.lstsq` instead of `numpy.linalg.lstsq` because the Windows numpy/MKL stack in the development environment raises fatal exceptions for `np.linalg` matrix operations. The public API still accepts and returns NumPy arrays, keeping the plugin compatible with the rest of the pipeline.
- Projection matrices are computed from camera intrinsics/extrinsics with PyTorch (`K [R|t]`) to avoid the same numpy BLAS issue.

## 5. Smoke-test validation

Run:

```bash
/d/anaconda3/envs/mf/python -u experiments/eval_robust_triangulation_baseline.py
```

Result on a 5-view, 17-joint, 30-frame synthetic sequence:

| Setting | DLT MPJPE (mm) | IRLS MPJPE (mm) | IRLS PA-MPJPE (mm) |
|---------|----------------|-----------------|--------------------|
| Clean (0% outliers) | 1.7 | 1.7 | 1.8 |
| 20% outliers | 5.6 | 2.3 | 2.7 |

IRLS recovers close to the clean error even when 20% of the 2D observations are corrupted by large Gaussian outliers.

## 6. Expected impact

- **Strong geometric baseline:** gives the team a parameter-free lower bound against which to compare learned robust-triangulation models.
- **Outlier robustness:** explicitly down-weights badly projected/occluded views, which should help on real-world datasets with missing keypoints.
- **Plug-and-play:** as a `FusionModule`, it slots into the same evaluation and training harness as the current best RayAttentionFusionModelTemporalResidual.

## 7. Known limitations / blockers

- The built-in `DLTFusion` (`motionflow_mv/fusion/fusion_module.py`) calls `Camera.projection_matrix`, which uses `numpy` matrix multiplication. On the current Windows development environment, `numpy` BLAS/LAPACK operations (`@`, `np.linalg.lstsq`, `np.linalg.svd`) raise fatal Windows exceptions. This prevents the existing `DLTFusion` from running here, but the new IRLS plugin itself works because it routes all linear algebra through PyTorch.
- Long-dataset evaluation (MPI-INF-3DHP, H36M, AIST) has not been run; the smoke test uses only synthetic data. MPI-INF-3DHP metrics require a data loader that is not exercised in this quick prototype.

## 8. Next steps / follow-up

- Plug `RobustTriangulationBaselineFusion` into the MPI-INF-3DHP/H36M evaluation harness to obtain a full MPJPE/PCK/AUC comparison against `DLTFusion` and `RobustTriangulationModel`.
- Explore adaptive choices of `eps` (e.g., based on image resolution or per-joint uncertainty).
- Consider a hybrid learned+IRLS model where the neural network predicts only the initial outlier mask/weights, and IRLS refines them geometrically.
