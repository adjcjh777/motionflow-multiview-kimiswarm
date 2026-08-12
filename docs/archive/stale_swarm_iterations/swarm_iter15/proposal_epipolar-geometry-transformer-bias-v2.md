# Proposal: Epipolar-Geometry Transformer Bias v2

## One-sentence hypothesis

Injecting calibrated multi-view epipolar geometry as a *relative-position bias*
inside the spatio-temporal transformer will make cross-view feature fusion more
robust to noisy/dropped views and improve 3D skeleton accuracy while preserving
the existing triangulation-and-residual-refinement pipeline.

## Related existing files/modules

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`
  – current iter14 anchor (`RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`,
  clean MPJPE 9.32 mm on MPI-INF-3DHP S2/Seq1).
- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_model.py`
  – v1 baseline that applies epipolar bias only to the final per-view weight head.
- `motionflow_mv/fusion/epipolar_attention_bias.py` – differentiable epipolar
  distance utilities used by v1.
- `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py` – base
  spatio-temporal transformer architecture (time + view tokens).
- `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`
  – training script for the anchor / v1 variants.

## Proposed code changes

1. **New helper module** `motionflow_mv/fusion/epipolar_transformer_bias.py`
   - `_aggregate_pairwise_epipolar_distance(K, R, t, points_2d) -> (N, V, V, J)`
   - `compute_per_frame_epipolar_bias(K, R, t, points_2d, temperature) -> (N, V, V)`
   - `build_temporal_bias_from_frames(per_frame_bias, n_heads, n_joints)`
     builds an `(B*J*n_heads, T*V, T*V)` attention mask with block-diagonal
     time structure.
   - `EpipolarBiasedTransformerEncoderLayer` – mirrors
     `nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=d*2,
     batch_first=True, norm_first=True)` but accepts an `epipolar_bias`
     argument and forwards it to `MultiheadAttention(attn_mask=...)`.

2. **New model file** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_epipolar_bias_v2_model.py`
   - Class: `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2`
   - Inherits from `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
   - Adds:
     - `epipolar_temperature: float = 10.0` (smaller = sharper geometry bias)
     - `gate_init: float = 2.0` (initial logit for sigmoid-blended bias gate)
     - `self.epipolar_gate = nn.Parameter(torch.full((1,), gate_init))`
   - Replaces `self.st_transformer` with a `ModuleList` of
     `EpipolarBiasedTransformerEncoderLayer`.
   - In `forward`, after principal-point correction, computes the symmetric
     per-frame epipolar bias, expands it to `(B*J*n_heads, T*V, T*V)`, and
     passes it into every ST transformer layer modulated by `sigmoid(epipolar_gate)`.
   - All remaining logic (weight head, DLT triangulation, residual MLP,
     return flags) is inherited/identical to the anchor, keeping the change
     minimal and ablatable.

3. **Registration in training/eval factory** (to be done after smoke test)
   - Add a new `model_type="epipolar_bias_v2"` branch in
     `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`.
   - Add to the model dictionary in `experiments/eval_full_metrics.py` under
     key `epipolar_bias_v2_pp` for benchmark parity.

## Training/smoke plan

- **Dataset**: MPI-INF-3DHP S1/Seq01+Seq02 train, S2/Seq01 val (canonical
  WebBridge multiview `.npz` files already in `data/webbridge/mpi_inf_3dhp/`).
- **Clip**: 13 frames, batch size 8, d=64, n_st_layers=2, residual_hidden=128.
- **Command** (≤5 epochs smoke):
  ```bash
  python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
      --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
             data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
      --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
      --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 \
      --epochs 5 --batch_size 8 --lr 1e-3 --pp_loss_weight 0.1 \
      --cam_aug_pp 5.0 --model_type epipolar_bias_v2 \
      --output outputs/ray_attention_temporal_crossview_residual_pp_epipolar_bias_v2_smoke.pth
  ```
- **Runtime estimate**: ~45–60 minutes on the local RTX 4090 for 5 epochs
  (anchor smoke is ~40 min under the same settings).  No background GPU jobs
  are to be launched now; this command is provided for the next experimental
  step.

## Success metrics

- **Primary**: Clean MPJPE on MPI-INF-3DHP S2/Seq1 ≤ 9.20 mm (beating the 9.32 mm
  anchor; v1 epipolar reports ~9.28 mm in prior runs).
- **Robustness axis**: MPJPE under synthetic noise/outliers (noise_std=2.0 px,
  outlier_rate=0.05, view_dropout_rate=0.25) should improve by ≥3 % over the
  anchor, measured with the existing `eval_occlusion_robustness.py` protocol.
- **Geometry usage**: `epipolar_gate` should converge to a value > 0.5
  (sigmoid output), confirming that the model actually uses the epipolar bias.
- **Training stability**: no NaNs/inf and validation loss monotonically
  decreases across the ≤5 epoch smoke.

## Risk and fallback

- **Risk**: `EpipolarBiasedTransformerEncoderLayer` changes the attention mask
  shape requirements; a shape mismatch can silently break the ST transformer.
  **Mitigation**: the attached helper expands the bias to the exact shape
  `(B*J*n_heads, T*V, T*V)` expected by `MultiheadAttention`, and a CPU forward
  pass is performed before any smoke run.
- **Risk**: Strong epipolar bias may over-regularize attention and hurt
  performance when camera calibration is already accurate.  **Mitigation**: the
  learned `epipolar_gate` can suppress the bias (sigmoid → 0); if the gate
  collapses to near-zero, the model reduces to the anchor.
- **Fallback**: if the ST-attention bias yields no improvement, fall back to the
  v1 epipolar weight-head bias or to the plain PP anchor, both of which share
  the same downstream triangulation path and require only a model-class swap.
