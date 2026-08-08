---
name: v25 Geometry Fusion Round
about: Track the v25 multi-view geometry fusion experiment round (geometry-aware cross-view attention, learned depth triangulation, confidence-weighted reprojection).
title: "EXP: v25 geometry fusion round — <run-id>"
labels: ["experiment", "P1-next"]
assignees: ''

---

## Experiment Summary

- **Round / Run ID:** <!-- e.g. v25_small, v25_full_a800_gpu5 -->
- **Based on baseline:** v18 (`RayAttentionFusionModelTemporalResidual`) — **20.24 mm val_MPJPE**
- **Goal:** Validate that `MultiViewGeometryFusionV25` improves cross-view fusion through geometry-aware attention, learned depth triangulation, and confidence-weighted reprojection loss.
- **Key hypothesis:** Explicit ray-based geometry attention + bounded GeoBA refines 3D pose without the v21-style camera regression.

## Model Configuration

- **Module file:** `motionflow_mv/fusion/multiview_geometry_fusion_v25.py`
- **Integration:** `motionflow_mv/fusion/omniview_fusion_v5.py`
- **Training script:** `experiments/train_omniview_fusion_v5_webbridge_multi.py`
- **Small-run script:** `scripts/run_v25_geometry_fusion_a800_small.sh`
- **Full-run script:** `scripts/run_v25_geometry_fusion_a800_full.sh`
- **Test file:** `tests/test_multiview_geometry_fusion_v25.py`

### Toggles

| Toggle | Value | Rationale |
|--------|-------|-----------|
| `use_multiview_geometry_fusion_v25` | `true` | Enable the v25 block |
| `v25_use_geometry_attention` | `true` | Geometry-aware cross-view attention |
| `v25_use_learned_depth_triangulation` | `true` | Learned depth-proposal triangulation head |
| `v25_use_geometry_bundle_adjustment` | `true` / `false` | Enable bounded GeoBA (smoke first) |
| `v25_use_camera_joint_graph` | `false` | Keep off for first round |
| `v25_geom_loss_weight` | `0.1` | Auxiliary geometry loss weight |

## Dataset & Resources

- **Datasets:** WebBridge + H36M + MPI-INF-3DHP mixed split
- **Manifest:** `configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml`
- **GPU target:** A800-D (smoke on RTX 4090 / CPU)
- **tmux session naming:** `v25_<size>_gpu<g>` (e.g. `v25_small_gpu5`)

## Metrics

- **Primary metric:** `val_MPJPE` (mm)
- **Baseline to beat / preserve:** v18 — **20.24 mm**
- **Secondary metrics:**
  - `val_PA_MPJPE` (mm)
  - Reprojection error (px)
  - Epipolar / cheirality / depth-consistency losses
  - Per-view outlier robustness (outlier_view_prob=0.3)
- **Success criterion:** `val_MPJPE` ≤ v18 baseline with no regression in PA-MPJPE; GeoBA block is stable (no camera divergence > 5 mm).

## Checklist

- [ ] Configuration and manifest double-checked
- [ ] Smoke test passed locally: `pytest tests/test_multiview_geometry_fusion_v25.py -q`
- [ ] v25 small launched on A800-D (or local RTX 4090 if A800 busy)
- [ ] First-epoch `val_MPJPE` observed and logged
- [ ] GeoBA identity-at-init verified
- [ ] Full v25 run launched when GPU is free
- [ ] Comparison table vs. v18 / v23b / v24b updated
- [ ] Failure modes documented (camera divergence, geometry loss explosion, etc.)

## Related Issues / Runs

- v23b: v18 + KAP, no neural BA
- v24b: v18 + fixed BA + KAP
- v21: neural BA (stopped; regressed to 128.27 mm)

## Notes / Observations

<!-- Log intermediate results, blockers, and decisions here. -->
