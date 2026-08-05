# Implementation Plan: Camera Positional Encoding

## Goal
Replace the learned dataset-specific `view_pos_embed` in the temporal/cross-view ray-attention models with a geometry-based camera positional encoding, and demonstrate variable-view and cross-dataset capability without retraining view IDs.

## Step 1: Add a reusable camera-positional-encoding module

- **Create file:** `motionflow_mv/fusion/camera_positional_encoding.py`
- **Implement:**
  - `fourier_features(x, L)` — sinusoidal positional encoding for a tensor of scalars.
  - `CameraPositionalEncoding(nn.Module)` — takes `(B, V, 3, 3)` `K`, `R`, `t`; returns `(B, V, d)` camera position tokens.
  - Internal normalization of camera centers by rig diameter and focal length by mean focal length.

## Step 2: Build the new fusion model

- **Create file:** `motionflow_mv/fusion/ray_attention_temporal_crossview_camera_pos_model.py`
- **Base:** start from `RayAttentionFusionModelTemporalCrossview` in `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py`.
- **Changes:**
  - Remove `self.view_pos_embed`.
  - Instantiate `self.camera_pos_enc = CameraPositionalEncoding(d=d)`.
  - In `forward`, after per-frame feature extraction, compute `view_emb = self.camera_pos_enc(K, R, t)` and add it to the `(B, T, V, J, d)` feature grid as `view_emb[:, None, :, None, :]`.
  - Keep `time_pos_embed` unchanged.
  - Add `view_mask: Optional[torch.Tensor] = None` argument and pass it to `st_transformer` layers as `src_key_padding_mask` to support variable view counts.

## Step 3: Unit tests

- **Create file:** `tests/test_camera_positional_encoding.py`
- **Verify:**
  - Output shape `(B, V, d)` for variable `V`.
  - Scale invariance: doubling all camera translations leaves the output unchanged.
  - Permutation equivariance: reordering views reorders the output tokens correspondingly.

## Step 4: Training script

- **Create file:** `experiments/train_ray_attention_temporal_crossview_camera_pos_mpiinf3dhp.py`
- Mirror `experiments/train_ray_attention_temporal_crossview_mpiinf3dhp_v1.py` but import the new model class.
- Run a smoke test: 3 epochs on MPI-INF-3DHP subject 1, validate on subject 2 sequence 1, `clip_len=13`, `d=64`.

## Step 5: Evaluation and ablation

- Run the new model and record MPJPE, PA-MPJPE, PCK, and AUC.
- Compare against the baseline with learned `view_pos_embed` (`docs/paper_draft_icra_cvpr_2027.md` table).
- Evaluate with `V=2,3,4` on the same rig to test variable-view degradation.

## Success metrics

- MPI-INF-3DHP S2/Seq1 MPJPE within 1 mm of the learned-embedding baseline (≤12 mm).
- Variable-view evaluation: graceful MPJPE increase as views are dropped.
- Cross-dataset: the same checkpoint runs on H36M (4 views) and MPI-INF-3DHP (14 views) without shape errors.
