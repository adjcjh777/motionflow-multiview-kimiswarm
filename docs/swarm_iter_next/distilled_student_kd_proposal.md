# Real-Time Model Compression: Knowledge-Distilled Lightweight Student

## Motivation

Current best models on MPI-INF-3DHP are heavy. The anchor to beat is 8.75 mm. This proposal distills a lightweight student from the Bayesian triangulation teacher for real-time deployment.

## Method

- Teacher: RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri (261k params).
- Student: DistilledStudentPrincipalPointModel (~65k params).
- Loss: L = (1-alpha)*MSE(student, gt) + alpha*MSE(student, teacher) + beta*(1 - cos_sim(weights)).

## Files

- motionflow_mv/models/distilled_student_principal_point_model.py
- experiments/train_distilled_student_pp_mpiinf3dhp.py
- scripts/run_distilled_student_pp_smoke_wsl.sh
- tests/test_distilled_student_pp.py
- docs/swarm_iter_next/distilled_student_kd_proposal.md

## How to Run

```bash
bash scripts/run_distilled_student_pp_smoke_wsl.sh
```

## References

- Issue #23
- Issue #25
