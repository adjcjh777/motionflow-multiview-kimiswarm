# Semi-Supervised Pseudo-Labeling for MotionFlow-MultiView

**Direction:** semi_supervised_pseudo_labeling  
**Swarm anchor:** 8.75 mm MPJPE on MPI-INF-3DHP / Human3.6M / WebBridge  
**Author:** Kimi Code CLI (semi-supervised direction agent)

## Motivation

The current best supervised runs are plateauing around 9.8–10.2 mm clean MPJPE. 3D pose labels are expensive, but unlabeled multi-view video is abundant in WebBridge and similar datasets. Semi-supervised pseudo-labeling can extend the effective training set by using a teacher model to generate 3-D pseudo-ground-truth for unlabeled clips and training a student on a mixture of labeled and pseudo-labeled data.

## Method

We implement a two-stage teacher-student pipeline:

1. **Teacher**  
   - Load an existing supervised checkpoint *or* train a fresh teacher for a few epochs on the labeled split.  
   - The teacher generates 3-D pseudo-labels for unlabeled multi-view clips.

2. **Pseudo-label generation**  
   - Unlabeled `.npz` files follow the standard layout: `points_2d`, `confidences`, `camera_K`, `camera_R`, `camera_t`, but no `joints_3d`.  
   - For each clip, the teacher predicts `pred_3d`.  
   - A per-clip confidence weight is computed from the teacher's mean reprojection error on the visible 2-D keypoints: low reprojection error → weight ≈ 1; high error → weight ≈ 0.  
   - This weight down-weights unreliable pseudo-labels during student training.

3. **Student training**  
   - The student is trained on alternating labeled and pseudo-labeled mini-batches.  
   - Loss = `MSE(student, labeled_GT) + λ_pseudo * weighted_MSE(student, pseudo_GT * confidence)`.  
   - Both labeled and pseudo-labeled clips receive the same camera-perturbation / augmentation pipeline as the supervised baseline.

## Files added

- `experiments/train_pseudo_label_ray_attention_mpiinf3dhp.py` — main trainer  
- `tests/test_pseudo_label_training.py` — CPU smoke test  
- `scripts/run_pseudo_label_smoke_wsl.sh` — WSL CPU smoke wrapper  
- `docs/swarm_iter_next/design_semi_supervised_pseudo_labeling/design_semi_supervised_pseudo_labeling.md` — this document

## How to run

### CPU smoke test

```bash
bash scripts/run_pseudo_label_smoke_wsl.sh
```

This generates tiny synthetic labeled/unlabeled `.npz` files, trains a teacher for 1 epoch and a student for 1 epoch on the CPU, and checks that a checkpoint is produced.

### Full run (requires WebBridge MPI-INF-3DHP `.npz` files)

```bash
python experiments/train_pseudo_label_ray_attention_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --unlabeled data/webbridge/mpi_inf_3dhp/s_04_seq_01_v14_multiview_m.npz \
    --teacher outputs/ray_attention_pp_teacher.pth \
    --clip_len 13 --d 64 --residual_hidden 128 \
    --epochs 20 --lambda_pseudo 0.5 --pseudo_conf_thresh 5.0 \
    --output outputs/ray_attention_pp_pseudo_student.pth
```

If no teacher checkpoint is available, set `--teacher_epochs 5` to train the teacher inside the same run.

## Expected outcome

- A reproducible semi-supervised training loop that does not crash and produces a valid checkpoint.  
- By adding unlabeled MPI-INF-3DHP / WebBridge sequences, the student should close part of the gap toward the 8.75 mm anchor without requiring new 3-D annotations.  
- The confidence-weighted pseudo-label loss provides a knob to trade off pseudo-label quantity vs. quality.

## Next validation steps

1. Confirm the CPU smoke test passes.  
2. Run a short GPU smoke (2–3 epochs) on real MPI-INF-3DHP data to verify no OOM and stable loss.  
3. Compare the student against the supervised-only baseline on the same labeled split.  
4. Tune `λ_pseudo` and `pseudo_conf_thresh` with a small grid search.  
5. Optionally extend to Human3.6M and cross-dataset WebBridge mixes once MPI-INF-3DHP results are promising.
