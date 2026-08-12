# Bone-Length & Skeleton Consistency Losses

## 1. Current state

Skeleton-aware losses are already partially implemented, but **not yet used by the current best model**.

- `experiments/train_utils.py` (lines 24–196) provides reusable helpers:
  - `bone_length_loss(pred, target, parents/parent_pairs, weight)` — supervised L1 on per-bone lengths.
  - `temporal_bone_length_consistency_loss(...)` — penalises bone-length variance across a batch/sequence.
  - `bone_symmetry_loss(...)` — L1 between mirrored left/right bone lengths.
  - `skeleton_consistency_loss(...)` — combined temporal + symmetry wrapper.
  - Presets: `H36M_17_PARENTS`, `COCO_17_PARENTS`, `SMPL17_PARENTS`.
- `experiments/train_ray_attention_v3_h36m.py` wires the helpers in with `--bone_weight` / `--consistency_weight` / `--skeleton_layout`, but it trains the older `RayAttentionFusionModelV3` on Human3.6M, not the temporal-residual model.
- `experiments/train_ray_attention_temporal_mpiinf3dhp_aux_v1.py` uses `bone_length_loss` for the 28-joint MPI-INF-3DHP skeleton with a supervised L1 target, but only on the non-residual `RayAttentionFusionModelTemporal`.
- `experiments/train_ray_attention_temporal_residual_v3_mpiinf3dhp.py` re-implements a *crude* unsupervised bone-length regulariser (`bone_length_loss`, lines 130–142) inside the trainer, but the paper draft explicitly states: **"No auxiliary reprojection or bone-length losses are used."**
- Best current checkpoint `outputs/ray_attention_temporal_residual_final5.pth` (MPJPE 11.17 mm on MPI-INF-3DHP) is the **temporal ray-attention + residual refinement** model and was trained without any bone-length/skeleton consistency loss.

## 2. Gap / opportunity

The current best model (`RayAttentionFusionModelTemporalResidual`) and its training script (`train_ray_attention_temporal_residual_v3_mpiinf3dhp.py`) do not exploit the skeleton-aware losses. Adding a properly weighted supervised bone-length and an unsupervised temporal bone-length consistency term could:

- Reduce the residual head’s freedom to distort bone lengths.
- Improve robustness to occlusion and noisy views by enforcing anatomical plausibility.
- Provide a stronger regulariser than the current crude variance-based loss.

This is a low-risk, low-effort regularisation experiment that may push the MPI-INF-3DHP / H36M numbers further down and give a cleaner ablation story for the paper.

## 3. Concrete next step

Create a new trainer `experiments/train_ray_attention_temporal_residual_aux_mpiinf3dhp.py` (a copy of `train_ray_attention_temporal_residual_v3_mpiinf3dhp.py`) with the following changes:

1. Import `bone_length_loss` from `experiments/train_utils.py` and the MPI-INF-3DHP 28-joint parent array from `train_ray_attention_temporal_mpiinf3dhp_aux_v1.py`.
2. Add CLI flags: `--bone_weight` (default `0.01`), `--bone_temporal_weight` (default `0.005`).
3. In the training loop, after the base MSE loss, add:
   - Supervised bone-length L1: `bone_length_loss(pred, yb, parents=MPI_INF_3DHP_28_PARENTS, weight=args.bone_weight)`.
   - Unsupervised temporal bone-length consistency: `temporal_bone_length_consistency_loss(pred, parents=MPI_INF_3DHP_28_PARENTS, weight=args.bone_temporal_weight)`.
4. Train a smoke run (2 epochs, S1 Seq1+Seq2 → S2 Seq1, clip_len=13) using the existing final5 hyperparameters (`d=64`, `residual_hidden=128`).
5. If the smoke shows no regression, run the full 5-epoch protocol and compare to `outputs/ray_attention_temporal_residual_final5.pth`.

## 4. Expected success metric

- Primary: cross-subject MPI-INF-3DHP MPJPE < 11.17 mm (current best).
- Secondary: improved PA-MPJPE and robustness under 50% joint occlusion / 20% outliers (reported via `experiments/eval_all_datasets.py` or the existing robustness scripts).
- Sanity check: training remains stable (no NaNs, no divergence) and the auxiliary loss terms are ≤10–20% of the MSE magnitude.

## 5. Risks / blockers

- **A800-D / Docker read-only**: cannot train there; use the local RTX 4090 / WSL environment.
- **Data**: WebBridge MPI-INF-3DHP and H36M `.npz` files are already in `data/webbridge/`, so no download is needed for this experiment.
- **Weight sensitivity**: previous aux-loss smoke runs (`mpiinf3dhp_aux_losses_report.md`) found `weight=0.1` too dominant; start at `0.01` for bone and `0.005` for temporal consistency.
- **Skeleton layout mismatch**: the 28-joint MPI-INF-3DHP parent array in `train_ray_attention_temporal_mpiinf3dhp_aux_v1.py` should be double-checked against the actual joint order in `data/webbridge/mpi_inf_3dhp/*_multiview_m.npz`.
- **No file commits**: do not commit checkpoints or large `.npz` files; only the new trainer script and this report should enter git.
