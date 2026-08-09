# MotionFlow-MultiView v49 Paper Story

## One-sentence claim

> MotionFlow-MultiView is a self-evolving multi-view 3D human pose system that uses geometric triangulation as a foundation, learns per-view reliability from its own reprojection residuals, and then generalizes the resulting pose estimator across sparse views, time, domains, and real-time streaming constraints.

## Four-act story arc

| Act | Technical content | Claim | ICRA/CVPR 2027 angle |
|---|---|---|---|
| 1. Geometry foundation | v25 multi-view geometry fusion + v45 adaptive triangulation weights | Triangulation with learned adaptive weights is the strongest core signal | Geometry is not dead: projective cues still beat pure transformer baselines |
| 2. Self-critique reliability | v37 self-critique view reliability, v39 reliability-coupled refinement, v43 adaptive residual | The model learns to trust its own views from reprojection residuals | A differentiable feedback loop replaces hand-tuned confidence thresholds |
| 3. Robustness in the wild | v46 sparse-view generalization, v47 temporal aggregation, v48 domain generalization | The same model handles missing views, motion, and domain shifts | From studio rigs to in-the-wild multi-view video |
| 4. Deployment | v49 real-time streaming, dynamic view budget, causal temporal smoother | Low-latency inference on constrained hardware | Accuracy-latency trade-offs for practical multi-camera systems |

## Pipeline figure (text)

```text
2D keypoints + cameras (any V >= 2)
        |
        v
[v25/v45 geometry fusion + v37/v39 self-critique reliability]
        |    <-- self-evolution: residual -> reliability gate -> refined 3D
        v
[v46 sparse-view generalization]
        |
        v
[v47 temporal aggregation]
        |
        v
[v48 domain-invariant refinement]
        |
        v
[v49 real-time streaming head]
        |
        v
3D pose, per-joint uncertainty, and runtime budget
```

## Main experimental results (target / to be filled)

| Variant | val_MPJPE@full (mm) | MPJPE@2 (mm) | MPJPE@3 (mm) | MPJPE@4 (mm) | note |
|---|---|---|---|---|---|
| v25 geometry fusion (baseline) | ~17 (A800) | — | — | — | strongest known baseline |
| v45 adaptive geometry fusion | ? | ? | ? | ? | local smoke in progress |
| v46 sparse-view generalization | ? | ? | ? | ? | local smoke in progress |
| v47 temporal aggregation | ? | ? | ? | ? | local smoke in progress |
| v48 domain generalization | ? | ? | ? | ? | local smoke in progress |

## Key ablations (planned)

1. **No v46 dropout** – verify sparse-view gain is real.
2. **No v47 temporal** – verify temporal smoothing does not regress full-view accuracy.
3. **No v48 domain adapter** – verify 3DPW actual-mode gap reduction.
4. **No v37 reliability feedback** – verify self-evolution loop contribution.

## Related-work positioning

| Ours | Gap in literature | Representative prior work |
|---|---|---|
| v25/v45 geometry fusion | End-to-end transformers ignore projective geometry | Liao & Zhu 2023; Moliner & Huang 2024 |
| v37/v39 self-critique reliability | Hand-tuned confidence or hard outlier rejection | Bragagnolo et al. 2024; Davoodnia et al. 2024 |
| v46/v47/v48 robustness | Fixed camera rigs and studio data only | Ghasemzadeh & Alahi 2024; Choudhury & Kitani 2023 |
| v49 real-time streaming | Accurate lifters are not latency-aware | MV-SSM (Chharia et al. 2025); RUMPL (2025) |

## Next milestones

- [ ] v46/v47/v48 local smoke passes with finite val_MPJPE.
- [ ] v46/v47/v48 A800 full runs complete.
- [ ] MPJPE@k benchmark on H36M / MPI / 3DPW actual-mode.
- [ ] v49-Lite RTX 4090 smoke (real-time variant).
- [ ] Full paper draft with figures and tables.
