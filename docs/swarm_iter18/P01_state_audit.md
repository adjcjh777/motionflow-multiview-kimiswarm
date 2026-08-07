# P01 – State Audit (MotionFlow-MultiView, swarm-iter18-omniview)

> Branch: `feat/swarm-iter18-omniview`  
> Audit time: 2026-08-07 ~10:55 CST (session time 2026-08-07T02:52:08Z)  
> Sources: `docs/results_iter16.md`, `docs/experiment_log_icra_cvpr_2027.md`, `outputs/*.log`, WSL process list, `git branch -a`.

---

## 1. Current Best Results

| Dataset | Model / ensemble | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | AUC | Source |
|---|---|---:|---:|---:|---:|---|
| MPI-INF-3DHP S2/Seq1 | Bayesian Tri v2 ensemble (stabilized + aug) | **8.35** | 5.29 | 1.000 | **0.9444** | `outputs/bayesian_tri_v2_ensemble_2_eval.json` |
| MPI-INF-3DHP S2/Seq1 | Bayesian Tri v2 ensemble (earlier stabilized + aug snapshot) | 8.61 | 5.38 | 1.000 | 0.9426 | `outputs/bayesian_tri_v2_ensemble_eval.json` |
| MPI-INF-3DHP S2/Seq1 | `bayesian_tri_v2_stabilized_mpiinf3dhp` (single) | 9.03 | 5.69 | 1.000 | 0.9398 | `outputs/bayesian_tri_v2_stabilized_mpiinf3dhp_eval.json` |
| MPI-INF-3DHP S2/Seq1 | Iter16 anchor (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`) | 8.75 | 4.95 | — | — | `docs/results_iter16.md` |
| Human3.6M S5/Act2 | CamPE+GraphJR (d=64, h=128) | 0.62 | 0.70 | 0.9993 | 0.9936 | `docs/experiment_log_icra_cvpr_2027.md` |

**Key takeaway:** The current best clean MPI-INF-3DHP S2/Seq1 result **8.35 mm** already satisfies the ICRA/CVPR 2027 publishable-accuracy target of `< 8.75 mm`.

### Robustness snapshot (single stabilized model)
From `outputs/robustness_matrix_bayesian_tri_v2_stabilized_run1.log` → `outputs/extended_robustness_matrix_bayesian_tri_v2_stabilized/robustness_matrix.*`.

| Condition | MPJPE (mm) | PA-MPJPE (mm) | AUC |
|---|---:|---:|---:|
| clean | 9.03 | 5.69 | 0.940 |
| noise 0.5 px | 9.05 | 5.73 | 0.940 |
| noise 1.0 px | 9.10 | 5.83 | 0.939 |
| noise 2.0 px | 9.31 | 6.19 | 0.938 |
| joint occlusion 10 % | 11.43 | 9.20 | 0.924 |
| joint occlusion 20 % | 14.56 | 12.90 | 0.903 |
| joint occlusion 30 % | 16.99 | 15.98 | 0.887 |
| view dropout 10 % | 12.03 | 6.06 | 0.920 |
| view dropout 30 % | 18.15 | 7.02 | 0.879 |
| view dropout 50 % | 23.89 | 8.86 | 0.841 |

---

## 2. Running GPU Experiments (WSL)

Identified via `wsl ps aux` and recent log writes. NVIDIA RTX 4090 is at ~13.1 GB / 24 GB VRAM, ~85 % GPU util (Windows `nvidia-smi` does not enumerate WSL compute processes).

| PID | Start | Script / command | Output checkpoint | Latest status |
|---:|---|---|---|---|
| 31119 | 09:43 | `scripts/run_bayesian_tri_v2_aug_wsl.sh` | `outputs/bayesian_tri_v2_aug_mpiinf3dhp.pth` | epoch 33, best val 9.17 mm at epoch 22 |
| 31673 | 09:53 | `scripts/run_epipolar_bias_v2_lite_pp_full_data_wsl.sh` | `outputs/epipolar_bias_v2_lite_pp_full_data_mpiinf3dhp.pth` | PP-head pre-training done; end-to-end just starting |
| 32724 | 10:15 | `scripts/run_graph_joint_relation_full_data_wsl.sh` | `outputs/graph_joint_relation_full_data_mpiinf3dhp.pth` | PP-head pre-training in progress |
| 33006 | 10:30 | `scripts/run_bayesian_tri_v2_visibility_wsl.sh` | `outputs/bayesian_tri_v2_visibility_mpiinf3dhp.pth` | epoch 2, val 13.90 mm (warm-started from stabilized) |
| 33671 | 10:49 | `scripts/run_bayesian_tri_v2_attention_entropy_wsl.sh` | `outputs/bayesian_tri_v2_attention_entropy_mpiinf3dhp.pth` | just started (warm-started from stabilized) |
| 33618 | 10:48 | `experiments/eval_variable_views_bayesian_tri_v2.py` | — | variable-view MPJPE@k evaluation running |

All training runs use the same base architecture: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, `d=128`, `residual_hidden=256`, `n_st_layers=3`, `clip_len=13`, `epochs=30–50`, full MPI-INF-3DHP training set (`s_01/s_03–s_08`), validation on `s_02_seq_01`.

---

## 3. Recently Completed / Stopped Runs

| Run | Status | Notes |
|---|---|---|
| `bayesian_tri_v2_large_scale_mpiinf3dhp` | **Finished** | Eval 9.71 mm / PA 5.79 mm. Training diverged after epoch 8 (~40 mm val). |
| `bayesian_tri_v2_full_data_mpiinf3dhp` | **Incomplete / stopped** | Only 6 epochs logged, val 13.94 mm at epoch 6. No active PID. |
| `kinematic_chain_pp_full_mpiinf3dhp` | **Finished** | Eval 13.63 mm / PA 7.29 mm. |
| `hierarchical_attention_entropy_reg_full_mpiinf3dhp` | **Finished** | Full checkpoint saved; no clean eval JSON in `outputs/` yet. |

---

## 4. Available Checkpoints (selected)

Recent `.pth` files under `outputs/` (most recent first, by mtime):

- `bayesian_tri_v2_attention_entropy_smoke.pth` (Aug 7 10:41)
- `bayesian_tri_v2_visibility_mpiinf3dhp.pth` (Aug 7 10:41)
- `bayesian_tri_v2_visibility_smoke.pth` (Aug 7 10:27)
- `bayesian_tri_v2_aug_mpiinf3dhp.pth` (Aug 7 10:22) — **in-progress**
- `bayesian_tri_v2_full_data_mpiinf3dhp.pth` (Aug 7 10:12) — incomplete
- `bayesian_tri_v2_stabilized_mpiinf3dhp.pth` (Aug 7 08:56) — **current single best**
- `bayesian_tri_v2_large_scale_mpiinf3dhp.pth` (Aug 6 18:06) — diverged run
- `kinematic_chain_pp_full_mpiinf3dhp.pth` (Aug 6 17:54)
- `hierarchical_attention_entropy_reg_full_mpiinf3dhp.pth` (Aug 6 17:24)
- `epipolar_bias_v2_pp_full_mpiinf3dhp.pth` (Aug 6 12:08)
- `splat_pp_full_mpiinf3dhp.pth` (Aug 6 11:22)
- `bayesian_tri_pp_full_mpiinf3dhp.pth` (Aug 6 11:41)
- `ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth` (Aug 6 06:00) — iter16 anchor

A full machine-readable list is in the working tree via `ls -lt outputs/*.pth`.

---

## 5. Open Branches

Current branch: **`feat/swarm-iter18-omniview`**.

Local branches (all local branches from `git branch -a`):

```
  attention-entropy-interpretability
  attention-entropy-interpretability-final
  attention-entropy-interpretability-v2
  camera-conditioned-pp-module-registration
  clean-data-aug
  data-aug-clean
  design-v2-paper-direction
  domain_adaptation_shelf_campus
  domain_adaptation_shelf_campus_v2
  feat/adaptive-scale-spatial-pyramid-fusion
  feat/bayesian-tri-v2-batched-lstsq
  feat/crossview-visibility-uncertainty-v1
  feat/fast-epipolar-bias-v2-pp
  feat/iter-next-ablation-csv-plotting
  feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability
  feat/iter-next-audit-webbridge-mpi-inf-3dhp-data-availability-wt
  feat/iter-next-bayesian-tri-v2-batched-dlt-tests
  feat/iter-next-confidence-aware-view-dropout
  feat/iter-next-cross-view-graph-attention
  feat/iter-next-draft-icra-cvpr-paper-story
  feat/iter-next-ema-checkpoint-save-load-support
  feat/iter-next-ensemble-inference-multi-checkpoint
  feat/iter-next-extend-camera-perturbation-ranges-and-intrinsics-curriculum
  feat/iter-next-extend-robustness-matrix
  feat/iter-next-hp-search-large
  feat/iter-next-integration
  feat/iter-next-learned-per-joint-precision-and-refinement
  feat/iter-next-prototype-deeper-st-attention
  feat/iter-next-roadmap
  feat/iter-next-synchronized-multiview-2d-augmentation
  feat/iter-next-synthesize-swarm-outputs
  feat/iter-next-synthetic-joint-occlusion
  feat/iter-next-temporal-velocity-acceleration-consistency-loss
  feat/iter-next-trainer-cosine-warmup-clip-amp
  feat/iter-next-update-gh-issue-25
  feat/iter17-adaptive-scale-spatial-pyramid
  feat/iter17-attention-entropy-regularization
  feat/iter17-bayesian-tri-v3
  feat/iter17-camera-conditioned-pp
  feat/iter17-confidence-aware-view-dropout
  feat/iter17-cross-view-graph-attention
  feat/iter17-crossview-contrast-ssl
  feat/iter17-deeper-st-attention
  feat/iter17-ema-checkpoint-save-load
  feat/iter17-epipolar-bias-v2-lite
  feat/iter17-extended-camera-perturbation-curriculum
  feat/iter17-graph-joint-relation
  feat/iter17-kinematic-chain-constraints
  feat/iter17-mixed-dataset-balanced-sampling
  feat/iter17-physics-motion-prior
  feat/iter17-realtime-kd-student
  feat/iter17-semi-supervised-pseudo-labeling
  feat/iter17-splatv2-view-dependent-covariance
  feat/iter17-temporal-velocity-acceleration-loss
  feat/iter17-visibility-uncertainty-v1
  feat/kinematic-chain-constraints-aux
  feat/multiscale-temporal-residual
  feat/semi-supervised-pseudo-labeling
  feat/set-transformer-crossview
* feat/swarm-iter18-omniview
  feat/temporal-ray-attention-deeper
  feat/temporal-skeleton-consistency-loss
  feat/unified-results-csv
  feature/data-augmentation-multiview-vstj
  feature/data-augmentation-multiview-vstj-clean
  feature/epipolar-bias-v2-lite
  feature/graph-joint-relation-full-run
  feature/mixed-dataset-balanced-sampling
  feature/physics-motion-prior
  feature/robustness-matrix-multi-model
  feature/splatv2-view-dependent-covariance
  feature/splatv2-view-dependent-covariance-clean
  feature/splatv2-view-dependent-covariance-clean2
  feature/splatv2-view-dependent-covariance-final
  feature/webbridge-mixed-17joint
  feature/webbridge-mixed-17joint-v3
  fix/h36m-corrected-track
  iter5-robust-triangulation
  iter6-residual-refiner
  iter7-temporal-refiner
  iter8-synthetic-pretrain
  main
  multiview-residual-exploration
  my-attention-entropy
  phase0-literature-audit
  phase1-humanmotion-ir
  realtime-kd-student-iter16
  ssl-view-contrast
  wt-proto-14460
```

Remote tracking mirrors the same pattern (`origin/...`).

---

## 6. Blockers & Follow-ups

1. **Ensemble loader dimension mismatch**  
   `outputs/bayesian_tri_v2_ensemble_eval.log` fails with a `size mismatch` error because `experiments/prototypes/eval_ensemble_checkpoints.py` instantiates the model with default `d=64` while the saved checkpoints are `d=128`. The 8.61 mm and 8.35 mm ensemble results were produced by scripts that passed the correct `--d 128`, but the standard `scripts/eval_ensemble_wsl.sh` may still be brittle if called without the right flags.

2. **GPU saturation / scheduling**  
   Six concurrent processes are sharing the single RTX 4090. Several runs are still in epoch 1–2; the queue should be monitored to avoid OOM or training stalls when larger batches start.

3. **Divergent `large_scale` run**  
   `bayesian_tri_v2_large_scale` jumped to ~40 mm after epoch 8. Needs diagnosis before re-running.

4. **Incomplete `full_data` run**  
   `bayesian_tri_v2_full_data_mpiinf3dhp` stopped at epoch 6 (val 13.94 mm). Decide whether to resume or drop.

5. **Missing clean evals**  
   `hierarchical_attention_entropy_reg_full_mpiinf3dhp` and a few other completed checkpoints lack an `outputs/*_eval.json`. Add them to the evaluation queue.

---

*End of audit. Next recommended action: wait for the five in-flight GPU runs to reach at least epoch 5–10, then run a fresh ensemble evaluation with `--d 128` and update this file.*

*Audit committed from `feat/swarm-iter18-omniview` at 2026-08-07 11:05 CST.*
