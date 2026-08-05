# MotionFlow-MultiView Experiment Log

This log records the chronological exploration toward ICRA / CVPR 2027.

## Self-evolution loop

Inspired by the iterative self-improvement idea in the Qwen3.8 release notes, the project is driven by a closed loop:

1. **Design** – use a swarm of planning agents to propose the next highest-ROI architecture / augmentation / loss change.
2. **Implement** – make the smallest code change that tests the idea.
3. **Train** – run on the local RTX 4090 (or read-only A800 data).
4. **Evaluate** – report clean accuracy and the same calibration-robustness / 2D-perturbation protocol.
5. **Critique** – compare to the current best and decide whether to keep, stack, or reject the change.
6. **Loop** – feed the new results back to the swarm and go to step 1.

The goal is not to build a bigger model but to converge on a compact, robust, and reproducible multi-view fusion module that reaches ICRA / CVPR 2027 publishable quality.

## Best results so far

| Dataset | Model | MPJPE (mm) | PA-MPJPE (mm) | PCK@50 | AUC |
|---|---|---:|---:|---:|---:|
| MPI-INF-3DHP S2/Seq1 | Temporal residual (d=64, h=128, 20 ep) | **10.46** | 8.93 | 1.000 | 0.9303 |
| Human3.6M S5/Act2 | CamPE+GraphJR (d=64, h=128) | **0.62** | 0.70 | 0.9993 | 0.9936 |

## MPI-INF-3DHP variants tried

| Model | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| Raw DLT | 25.21 | — | no learning |
| Temporal ray-attention (no residual) | 25.21 | 24.14 | base encoder only |
| Residual small (d=32, h=64) | 13.22 | 11.77 | 66 k params |
| Residual 3-epoch | 14.17 | 12.99 | intermediate |
| Residual 5-epoch | 11.17 | 8.24 | intermediate |
| **Residual 20-epoch** | **10.46** | **8.93** | best MPI result |
| CamPE 20-epoch | 11.25 | 9.14 | variable views, slightly worse |
| CamPE+Adaptive hard | 12.73 | 9.14 | Gumbel top-k gate |
| CamPE+Adaptive soft-gate | 12.84 | 11.57 | continuous gate, similar gap |
| CamPE+GraphJR cross-view only | 12.81 | 11.05 | no skeleton on MPI |
| CamPE+GraphJR full skeleton | 13.98 | 13.03 | real 28-joint graph |
| Cross-view residual (d=128, n_st=3, h=256) | 13.90 | 10.90 | 1.06 M params |
| Factorised cross-view/temporal | — | — | slow, stopped before convergence |
| Baseline + reprojection loss (w=0.01) | — | — | very slow, stopped |
| Larger baseline (d=96, h=192, n_t=3) | 18.79 | 12.30 | overfits, worse than baseline |

## H36M variants tried

| Model | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|---|---:|---:|---|
| Residual h=128 (WebBridge S1→S5) | 0.94 | 0.92 | strong baseline |
| CamPE h=128 | 1.39 | 1.14 | variable views |
| CamPE+GraphJR h=128 | 0.62 | 0.70 | graph helps on H36M |

## Key observations

- The **simple residual head on top of weighted DLT** is surprisingly strong and hard to beat on MPI-INF-3DHP.
- **CamPE** trades a small accuracy gap for variable camera rigs (important for practical deployment).
- **GraphJR** helps on Human3.6M (consistent 4-view rig, 17-joint skeleton) but not on MPI-INF-3DHP (14 views, 28 joints), possibly because the dense attention already captures relationships or the MPI skeleton graph adds noise.
- **Adaptive view selection** (hard and soft) does not help, suggesting the model already learns robust per-view confidences.
- **Reprojection loss** is computationally expensive and did not reach a checkpoint quickly.
- The **factorised cross-view/temporal model** is also slow; first epoch did not finish in time.

## Completed side experiments

- **Camera calibration robustness** (200-clip subset):
  - Clean: 7.45 mm / 10.19 mm PA
  - Rotation ±0.5°: 17.43 mm; ±1.0°: 32.44 mm
  - Translation ±5 mm: 7.91 mm; ±10 mm: 9.46 mm
  - Focal length ±1%: 9.90 mm; ±2%: 13.06 mm
  - Principal point ±3 px / ±5 px: catastrophic (>2 m) due to ray misalignment
- **GVHMR real-world projection demo** (H36M-trained residual checkpoint):
  - Clean 0.5 px noise: **3.13 mm**
  - 2 px noise + 10% view dropout: **8.73 mm**

## Ongoing experiments

- **Training-time camera calibration perturbation** (MPI-INF-3DHP):
  - Perturb per-clip cameras with ±0.5° rotation, ±5 mm translation, ±1% focal length, ±2 px principal point.
  - Goal: preserve clean accuracy while drastically improving robustness to rotation / principal-point errors.
  - Full 20-epoch run was very slow (~20 min/epoch); switched to a fast ablation: small model (d=32, h=64), 1 000 random clips/sequence, val_stride=10 (`outputs/ray_attention_temporal_residual_perturb_small_mpiinf3dhp.pth`).
  - Fast ablation results (small model, best checkpoint):
    - Perturbed model: Clean **15.67 mm** / 14.99 mm PA
    - rot ±0.5°: 21.39 mm (vs baseline 22.11); rot ±1.0°: 32.45 mm (vs baseline 35.51)
    - trans/focal: nearly unchanged vs baseline
    - principal point: still catastrophic (>1.8 m)
    - Fair small baseline: Clean **14.97 mm** / 15.21 mm PA
  - Perturbed + bone-length loss (w=0.1):
    - Clean **16.59 mm** / 16.44 mm PA (worse than perturbation-only)
    - rot ±0.5°: 21.87 mm; rot ±1.0°: 32.21 mm
    - Conclusion: bone-length loss does not help on this setup.
- **Bone-length auxiliary loss**:
  - Added `motionflow_mv/losses/bone_length.py` and wired it into the MPI-INF-3DHP training script via `--bone_weight`.
  - Can be stacked with camera perturbation and reprojection loss in the next run.
- **Combined evaluation script**:
  - Added `experiments/eval_perturb_model_mpiinf3dhp.py` to run clean + calibration-robustness in one pass and emit a JSON/ markdown summary.

## Full-model (d=64, h=128) camera-perturbation fast ablation

A full-size model with the same perturbation schedule (±0.5° rotation, ±5 mm translation, ±1% focal length, ±2 px principal point) was trained for 10 epochs on 1 000 random clips per train sequence. Evaluation on MPI-INF-3DHP S2/Seq1:

| Condition | No perturbation (full) | With perturbation (full) |
|---|---:|---:|
| Clean | **11.78** | 14.15 |
| Rotation ±0.5° | 20.15 | **19.47** |
| Rotation ±1.0° | 33.67 | **30.22** |
| Translation ±5 mm | 12.27 | 14.35 |
| Translation ±10 mm | 13.41 | 15.32 |
| Focal length ±1% | 12.57 | 14.42 |
| Focal length ±2% | 13.90 | 15.53 |
| Principal point ±3 px | 1656.68 | 1929.25 |
| Principal point ±5 px | 1941.93 | 2132.83 |

The full perturbed model improves over the small perturbed model (clean 15.67 mm → 14.15 mm; rot ±0.5° 21.39 mm → 19.47 mm; rot ±1.0° 32.45 mm → 30.22 mm), confirming that capacity helps absorb rotation errors. As expected, training with perturbation trades clean accuracy for rotation robustness: the no-perturbation full model is 2.4 mm cleaner but 0.7–3.5 mm worse under rotation. Translation and focal-length robustness remain strong. Principal-point errors are still catastrophic in both cases, which motivated the next experiment: a learned principal-point correction layer.

## Principal-point correction experiment (in progress)

- Implemented `motionflow_mv/fusion/principal_point_correction.py` and `motionflow_mv/fusion/ray_attention_temporal_residual_principal_point_model.py`.
- Added training script `experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py`.
- Added evaluation script `experiments/eval_principal_point_model_mpiinf3dhp.py`.
- A fast ablation (10 epochs, 1 000 clips/sequence, ±5 px principal-point perturbation) is now running on the local RTX 4090.

## 2026-08-05 — 20-agent swarm review of next directions

A second self-evolution loop was run with 20 planning agents, each reviewing one open direction against the current best model (cross-view temporal residual + principal-point correction, MPI-INF-3DHP clean **9.32 mm**). The full synthesis is in [`docs/next_iteration_plan_swarm.md`](./next_iteration_plan_swarm.md).

Top-ranked next actions are:

1. **Calibration perturbation curriculum** — increase rot/trans augmentation on the best PP model to close the rot_0.5° (16.89 mm) and rot_1.0° (27.45 mm) gaps while keeping clean ≤ 9.6 mm.
2. **Variable-view inference + view-dropout training** — benchmark MPJPE@k for k=2..14 and re-train with random view dropout.
3. **Unified WebBridge benchmark harness** — a single script to evaluate the best checkpoint across MPI/H36M/AIST/Shelf/Campus `.npz` files and produce paper tables.
4. **MotionFlow plugin integration** — wrap the best model as a `FusionModule` so the existing multi-view plugin can load it.
5. **Visibility-gated fusion v2** — add an explicit occlusion head to the best PP model.

GPU work is gated by the running calibration curriculum training; non-GPU items can proceed in parallel.

### 2026-08-05 (cont.) Implementation status

- Calibration curriculum + view-dropout training is running in WSL (`scripts/run_crossview_pp_curriculum_wsl.sh`).
- The training scripts for the best PP model and the visibility-gated model both support `--warm_start` for faster convergence.
- Variable-view evaluation now works for `crossview_residual_pp` and `crossview_residual_pp_visibility`.
- Next GPU queue after curriculum: visibility-gated fusion v2 training.

## 2026-08-06 — 20-agent swarm synthesis and non-GPU tooling

A third 20-agent swarm reviewed 20 open directions and produced a new prioritized plan in [`docs/next_iteration_plan_swarm.md`](./next_iteration_plan_swarm.md). Top P0 actions are now:

1. **Camera calibration robustness** — evaluate the ongoing curriculum and extend to focal/stronger extrinsic perturbation.
2. **Visibility-aware adaptive fusion** — run `scripts/run_crossview_pp_visibility_wsl.sh` once GPU is free.
3. **Variable-view inference & view-dropout** — benchmark MPJPE@k for k=2..14.
4. **Interpretability & failure analysis** — per-joint/per-view failure profiles and PP-correction visualizations.
5. **Evaluation protocol, metrics & reproducibility** — unify benchmark protocol, add metrics, run repeated seeds.

Implemented non-GPU components:

- `motionflow_mv/data/ssl_dataset.py` + `experiments/pretrain_ray_attention_ssl.py` + runners for MPI and H36M: self-supervised masked-view pretraining skeleton.
- `experiments/analyze_failures_crossview_pp.py` + runner: failure analysis for the PP model.
- `experiments/run_repeated_seeds.py`: multi-seed repeated training harness.
- `motionflow_mv/eval/metrics.py`: added `root_rel_mpjpe`, `velocity_mpjpe`, and `bone_length_error`.
- Started CPU background jobs: WebBridge cross-dataset benchmark and variable-view MPJPE@k curve on the crossview-residual baseline.

## 2026-08-06 (continued) — Variable-view, WebBridge v2, and swarm tooling landed

Completed background runs:

- **Variable-view inference (smoke)**: `outputs/variable_views_crossview_residual_smoke.log`
  - MPJPE@k (mm): k=2:101.2, 3:84.8, 4:73.9, 5:60.8, 6:50.2, 7:41.4, 8:34.2, 9:30.4, 10:30.7, 11:31.2, 12:33.9, 13:38.0, 14:14.0.
  - Plot saved to `docs/figures/variable_views_crossview_residual_smoke.png`.
- **WebBridge benchmark v2**: `outputs/webbridge_benchmark_crossview_residual_smoke_v2.json`
  - mpi_s2_seq1_v14: MPJPE 14.71 / PA 13.86 mm
  - mpi_s3_seq1_v14: MPJPE 14.70 / PA 11.41 mm
  - mpi_s1_seq1_v4: MPJPE 27.95 / PA 19.10 mm
- **Commit/push**: `2683a17` on `multiview-residual-exploration`.
- **Issue/PR comments**: Updated GitHub issue #21 and PR #17.

## 2026-08-06 (continued) — Interim robustness + visibility v2 trainer

- **Robustness eval on current (mid-training) curriculum checkpoint** (MPI S2 full):
  - clean: 10.69 / 7.01 mm (MPJPE/PA)
  - rot_0.5°: 26.78 / 11.09 mm
  - trans_5mm: 12.42 / 7.13 mm
  - focal_1%: 11.07 / 7.32 mm
  - pp_10px: 2023.42 / 459.74 mm (still breaking; final checkpoint evaluation pending)
- **New visibility v2 full MPI trainer**: `experiments/train_crossview_residual_visibility_v2_mpiinf3dhp.py` — CPU smoke passed (20.17 mm val on smoke data). Ready for GPU once the curriculum run finishes.

## Next directions

1. Evaluate the final calibration curriculum checkpoint (clean + robustness matrix) once GPU training finishes.
2. Start `train_crossview_residual_visibility_v2_mpiinf3dhp.py` on GPU after curriculum.
3. Run SSL pretraining on MPI/H36M once GPU is free.
4. Scale to mixed-dataset training (MPI + H36M + AIST++) once robustness is fixed.
5. Generate final paper figures and tables.
