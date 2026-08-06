# Real-Time Model Compression: Knowledge-Distilled Lightweight Student

## Motivation

Current best models on MPI-INF-3DHP are heavy:

- **Bayesian Triangulation** (`ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`): 261,930 params, full run MPJPE ~9.81 mm.
- **Hierarchical Attention** (`ray_attention_hierarchical_view_temporal_joint_residual_principal_point_model.py`): ~10.23 mm best so far.\nThe anchor to beat is **8.75 mm**. A smaller, faster model that can be deployed in real-time settings (e.g., streaming 30–60 Hz on an RTX 4090) is valuable both for the accuracy target and for practical deployment.

This proposal distills a lightweight student from the Bayesian triangulation teacher. The student reuses the proven principal-point cross-view residual architecture but at a much smaller capacity: `d=32`, `n_st_layers=1`, `residual_hidden=64` (~65k params, ~4x smaller than the teacher).

## Method

### Architecture

- **Teacher**: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri` (d=64, n_st_layers=2, residual_hidden=128).
- **Student**: `DistilledStudentPrincipalPointModel` in `motionflow_mv/models/distilled_student_principal_point_model.py` — a thin subclass of `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` with configurable reduced dimensions.

### Training

New trainer: `experiments/train_distilled_student_pp_mpiinf3dhp.py`

- Loads the frozen teacher from `outputs/bayesian_tri_pp_full_mpiinf3dhp.pth`.
- Runs the standard data loader/augmentation pipeline used by the PP baseline.
- Mixed loss:

  ```
  L = (1 - α) * MSE(student, gt) + α * MSE(student, teacher) + β * (1 - cos_sim(student_weights, teacher_weights))
  ```

  - `α = distill_alpha` (default 0.5): balances ground-truth supervision and teacher distillation.
  - `β = weight_align_beta` (default 0.1): aligns per-view weight distributions between teacher and student.
- Optional PP-head pre-training (`pp_pretrain_epochs`) to match the teacher's intrinsic curriculum.

### Smoke / Full Runs

- **Smoke**: `scripts/run_distilled_student_pp_smoke_wsl.sh` — CPU-only, 2 epochs on smoke subsets.
- **Full**: same trainer with full MPI-INF-3DHP sequences, 30 epochs, `d=32`, `n_st_layers=1`, `residual_hidden=64`.

## Expected Outcome

- **Size**: ~65k params vs. ~262k teacher params (~4x compression).
- **Speed**: significantly faster inference due to smaller hidden dim and single ST layer; exact FPS to be measured with `experiments/benchmark_runtime.py`.
- **Accuracy**: target is to approach the teacher's 9.81 mm while moving toward the 8.75 mm anchor. The distillation loss should help the small student recover teacher-level triangulation quality.

## How to Run

```bash
# CPU smoke test (no GPU)
bash scripts/run_distilled_student_pp_smoke_wsl.sh

# Full GPU run on MPI-INF-3DHP
python experiments/train_distilled_student_pp_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --teacher outputs/bayesian_tri_pp_full_mpiinf3dhp.pth \
  --clip_len 13 --d 32 --n_st_layers 1 --residual_hidden 64 \
  --distill_alpha 0.5 --weight_align_beta 0.1 \
  --batch_size 8 --train_samples 4000 --epochs 30 --val_stride 50 \
  --pp_loss_weight 0.2 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --pp_pretrain_epochs 3 \
  --output outputs/distilled_student_pp_mpiinf3dhp.pth
```

## Next Validation Steps

1. Run the CPU smoke test to confirm the trainer completes without errors.
2. Run the full GPU training on MPI-INF-3DHP.
3. Evaluate the resulting checkpoint with `scripts/eval_crossview_pp_full_wsl.sh` or equivalent.
4. Measure inference latency with `experiments/benchmark_runtime.py` to quantify real-time speedup.

## References

- Issue #23 (real-time / compression direction)
- Issue #25 (anchor < 8.75 mm)
- Existing teacher: `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model.py`
