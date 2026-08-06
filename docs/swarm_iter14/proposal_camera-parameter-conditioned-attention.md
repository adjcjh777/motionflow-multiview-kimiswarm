# Camera-Parameter-Conditioned Attention: Inject K/R/t into Attention as Embeddings

## 1. Problem

The current ray-attention fusion modules compute cross-view affinity from ray directions and 2-D features alone, so attention weights are agnostic to the actual camera intrinsics and extrinsics that generated those rays, which limits geometric consistency under calibration noise and view dropout.

## 2. Hypothesis

If we encode intrinsics K, rotation R, and translation t as per-view embedding vectors and inject them into multi-head attention as key/value/bias terms, the model will learn geometrically grounded affinities that improve both clean accuracy and robustness to focal/PP/extrinsic perturbations without increasing inference latency measurably.

## 3. Method

### 3.1 Architecture changes

Create a new model: `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_campe_model.py` (camera-parameter-conditioned attention, or CAMPE) that subclasses the existing `RayAttentionTemporalCrossviewResidualPrincipalPointModel` and replaces the raw ray-attention block with a camera-conditioned variant.

Create a new module: `motionflow_mv/fusion/camera_parameter_embedding.py` containing:

- `CameraParameterEmbedding(nn.Module)`:
  - Inputs: `K` (B, V, 3, 3), `R` (B, V, 3, 3), `t` (B, V, 3).
  - Outputs: `cam_emb` (B, V, D) where D = 64.
  - Implementation: flatten the upper-left 2×2 of K, focal length (fx, fy), principal point (cx, cy), the flattened rotation matrix R, and translation t; project through a 2-layer MLP with LayerNorm and ReLU.

- `CameraConditionedAttention(nn.Module)`:
  - Wraps the existing multi-head attention in `motionflow_mv/fusion/ray_attention_module.py`.
  - Adds `cam_emb` to keys and values before the attention dot product and as a per-head bias.
  - Formula: attention = softmax((Q · (K + cam_emb).T) / sqrt(d_k) + bias), where bias = MLP(cam_emb).

### 3.2 Loss changes

No new loss. Keep the standard MPJPE + PA-MPJPE + optional visibility BCE from the visibility-v2 branch. The camera conditioning is a pure architecture change, so the existing `train_crossview_residual_visibility_v2_mpiinf3dhp.py` trainer can be reused with a one-line model swap.

### 3.3 Data changes

No data loader change. The existing `MPIINF3DHP` dataset already returns `K`, `R`, `t` dictionaries; we just need to thread them into the model forward call.

### 3.4 Exact files to create or modify

- **Create** `motionflow_mv/fusion/camera_parameter_embedding.py`:
  - `CameraParameterEmbedding`
  - `CameraConditionedAttention`
- **Create** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_campe_model.py`:
  - Model class `RayAttentionTemporalCrossviewResidualCamPEModel`.
  - Inherits from `RayAttentionTemporalCrossviewResidualPrincipalPointModel`.
  - Overrides the attention block to use `CameraConditionedAttention`.
- **Create** `experiments/train_camera_parameter_conditioned_attention_smoke_mpiinf3dhp.py`:
  - Smoke trainer, 3–5 epochs, 500 samples, d=32, residual_hidden=64.
  - Mirrors `experiments/train_factorized_pp_smoke_mpiinf3dhp.py` but with the new model.
- **Create** `experiments/eval_camera_parameter_conditioned_attention_mpiinf3dhp.py`:
  - Evaluate clean + robustness matrix on the smoke checkpoint.
- **Create** this proposal file (done).
- **Modify** `motionflow_mv/fusion/__init__.py`:
  - Register `RayAttentionTemporalCrossviewResidualCamPEModel` in the model registry.
- **Modify** `experiments/benchmark_runtime.py` (if exists):
  - Add the new model to the latency sweep.

## 4. Smoke-Test Plan

Run a 3–5 epoch smoke on MPI-INF-3DHP with 500 training samples and d=32.

- Pass criteria:
  - No NaNs or crashes.
  - Val MPJPE ≤ 60 mm and PA-MPJPE ≤ 40 mm (smoke-level sanity only; full model target is ≤ 9.8 mm after 20 epochs).
  - Attention weights are non-uniform and depend on camera parameters (verified by ablating `cam_emb` to zero and observing changed MPJPE).
- Fail criteria:
  - Any NaN/Inf in losses or outputs.
  - Smoke MPJPE > 80 mm, indicating the camera embedding destabilizes training.
  - Runtime > 2× baseline per epoch.

## 5. Evaluation Plan

- Metrics: MPJPE, PA-MPJPE, PCK@50/100/150, AUC on MPI-INF-3DHP clean val/test split.
- Scripts:
  - `experiments/eval_camera_parameter_conditioned_attention_mpiinf3dhp.py` for clean metrics.
  - `experiments/eval_robustness_matrix_pp_mpiinf3dhp.py` extended with `--model campe` for the 6-axis robustness matrix (focal, PP, rotation, translation, view dropout, joint dropout).
- Acceptance: clean MPJPE ≤ 9.8 mm (within 0.5 mm of the 9.32 mm anchor) and no robustness axis degrades > 30% relative to the PP baseline under matched corruption.

## 6. Estimated GPU/CPU Cost on RTX 4090

- Smoke (3–5 epochs, 500 samples): ~8–12 minutes on RTX 4090, < 2 GB extra memory.
- Full 20-epoch run (queued later): ~2.5–3.5 hours on RTX 4090.
- Robustness matrix eval: ~10–15 minutes CPU/GPU.
- The camera embedding MLP and attention bias add < 5% compute; total parameter count remains under ~100 k.

## 7. Risks & Fallback

- **Risk:** Camera parameters are already implicitly present in ray directions, so explicit conditioning may add no new information and just increase parameters.
  - *Fallback:* Ablate by zeroing `cam_emb`; if it matches the base model, abandon the full experiment and report negative result.
- **Risk:** Injecting K/R/t as biases could destabilize attention gradients during early training.
  - *Fallback:* Scale `cam_emb` by a learned temperature or initialize its output projection to near zero.
- **Risk:** Extra MLP increases memory/latency on already CPU-bound visibility-v2 training.
  - *Fallback:* Reduce `D` from 64 to 32 or fuse the embedding with the existing positional encoding in `camera_positional_encoding.py`.
- **Risk:** Smoke runtime exceeds the 2× budget.
  - *Fallback:* Use the factorized attention backbone instead of the full `(time × view)` attention to keep conditioning cheap.
