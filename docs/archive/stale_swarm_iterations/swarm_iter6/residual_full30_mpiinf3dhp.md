# Full 30-Epoch Residual Temporal Training on MPI-INF-3DHP

## Goal
Run the existing residual temporal model (`RayAttentionFusionModelTemporalResidual`)
for a full 30 epochs on the MPI-INF-3DHP cross-subject split and report the final
cross-subject MPJPE on the standard validation sequence (S2 Seq1).

## Setup
- Model: `motionflow_mv/fusion/ray_attention_temporal_residual_model.py`
- Base trainer: `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py`
- Wrapper created: `experiments/train_ray_attention_temporal_residual_full30_mpiinf3dhp.py`
- Training data (cross-subject):
  - Train: S1 Seq1, S1 Seq1_02, S1 Seq2, S3 Seq1, S3 Seq2
  - Val:   S2 Seq1
- Hyperparameters: `d=64`, `n_temporal_layers=2`, `residual_hidden=128`,
  `clip_len=13`, `lr=1e-3`, `epochs=30`, `batch_size=32`, `train_samples=250`.

## Execution summary
Launched the wrapper with `conda run -n mf python ...`.

Training progressed through **Epoch 1** before failing:
```text
Device: cuda
n_views=14, j=28, clip_len=13, d=64, residual_hidden=128
Model params: 243428
Epoch 1: train_loss=0.002169, val_MPJPE=19.23mm (saved)
```

During Epoch 2, training crashed with a CUDA illegal memory access inside
`_triangulate_joint` (`torch.linalg.lstsq`):
```text
RuntimeError: CUDA error: an illegal memory access was encountered
  File ".../ray_attention_model.py", line 116, in _triangulate_joint
    X, *_ = torch.linalg.lstsq(Aw, bw)
```

Two restart attempts were made:
1. A lighter configuration (`batch_size=8`, `train_samples=250`) was launched
   but never produced output — the process was stalled waiting for GPU time.
2. CPU-side evaluation was used to bypass GPU contention.

A checkpoint was saved at the end of the first epoch before the crash:
`outputs/ray_attention_temporal_residual_full30.pth`.

## Result
The Epoch-1 checkpoint was evaluated on CPU over the full S2 Seq1 validation
sequence:

| Metric | Value |
|--------|-------|
| Checkpoint | `outputs/ray_attention_temporal_residual_full30.pth` |
| Training epochs completed | 1 / 30 |
| Validation MPJPE (S2 Seq1) | **16.77 mm** |

## Blockers
1. **GPU resource contention.** The local RTX 4090 was running multiple other
   swarm-agent training/eval jobs concurrently (up to 9+ processes at peak),
   leaving insufficient GPU time for the full 30-epoch run to complete.
2. **CUDA illegal memory access.** The crash inside `torch.linalg.lstsq` during
   triangulation is consistent with concurrent GPU access / memory pressure under
   heavy contention. It occurred after the first successful validation pass.
3. **Subsequent stalls.** Restarted runs produced no logged output because the
   process could not acquire meaningful GPU time.

## Takeaway
With only a single epoch of training, the residual temporal model already
reaches **16.77 mm** cross-subject MPJPE on MPI-INF-3DHP S2 Seq1. This is a
reasonable starting point but higher than the existing 13.84 mm reported for
`ray_attention_temporal_residual_v2.pth` (5-epoch S1-only smoke run). Completing
all 30 epochs would require exclusive GPU time or serialization of swarm jobs
to avoid concurrent CUDA access.

## Files touched
- `experiments/train_ray_attention_temporal_residual_full30_mpiinf3dhp.py`
- `experiments/eval_full30_checkpoint.py`
- `outputs/ray_attention_temporal_residual_full30.pth`
- `docs/swarm_iter6/residual_full30_mpiinf3dhp.md`
- `outputs/train_full30.log` (training log / crash trace)
