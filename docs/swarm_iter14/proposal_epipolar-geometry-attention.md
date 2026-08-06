# Epipolar Geometry Attention for Cross-View Ray Matching

**Date:** 2026-08-06
**Author:** MotionFlow-MultiView iter14 swarm
**Parent anchor:** `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` — 9.32 mm MPJPE / 5.37 mm PA-MPJPE on MPI-INF-3DHP clean

## 1. Problem (1 sentence)

The current cross-view attention treats every view-to-view token pair equally, ignoring the fact that a 2-D joint in one view can only project to a 1-D epipolar line in any other calibrated view, so geometrically implausible matches receive the same weight as plausible ones.

## 2. Hypothesis (1 sentence)

If we bias the cross-view attention scores with the distance of each candidate keypoint from the epipolar line induced by its paired view, the model will attend to geometrically consistent rays and lower triangulation error without adding new learnable parameters or changing the loss.

## 3. Method

### 3.1 Architecture change

Create a new model that subclasses the current anchor and injects an epipolar-line distance bias into the cross-view attention logits.

- **New file:** `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_epipolar_model.py`
  - Class `RayAttentionFusionModelTemporalCrossviewResidualEpipolar`
  - Inherits from `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`
  - Forward pass:
    1. After intrinsic correction (`K_corrected`, `R`, `t`) and ray-feature extraction, keep per-view 2-D keypoints in normalized image coordinates.
    2. For each pair of views `(v_src, v_dst)` and each joint `j`, compute the epipolar line `l` in `v_dst` induced by the ray/keypoint in `v_src` using the essential/fundamental matrix derived from the two corrected cameras.
    3. Compute the perpendicular image-space distance from the joint in `v_dst` to that epipolar line.
    4. Convert distance to a bias term `b_{v_src,v_dst}` via a fixed exponential or learned scalar, then add to the cross-view attention logits (not the temporal self-attention).
    5. Keep the temporal self-attention and residual MLP unchanged.

- **New file:** `motionflow_mv/fusion/epipolar_attention_bias.py`
  - Function `compute_epipolar_line_distance_2d(K_src, R_src, t_src, K_dst, R_dst, t_dst, pts_src, pts_dst)`
  - Returns `(B*T, V, V, J)` distance tensor `D[v_src, v_dst, j]` = signed/unsigned distance from the joint in `v_dst` to the epipolar line induced by the joint in `v_src`.
  - Function `epipolar_bias_from_distance(D, temperature=1.0, clip_max=100.0)` returns `-D / temperature` as an additive attention bias.
  - Pure geometry + one scalar temperature; no extra trainable weights by default. Optionally expose a single scalar `temperature` as a learnable parameter to keep risk minimal.

### 3.2 Files to create / modify

| File | Action | Purpose |
|------|--------|---------|
| `motionflow_mv/fusion/epipolar_attention_bias.py` | Create | Epipolar line + distance utilities, differentiable w.r.t. cameras |
| `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_epipolar_model.py` | Create | New model injecting the epipolar bias into cross-view attention layers |
| `motionflow_mv/fusion/__init__.py` | Modify | Register the new model class |
| `experiments/train_epipolar_crossview_residual_mpiinf3dhp.py` | Create | Smoke trainer (copy from `train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py` and point to new model) |
| `scripts/run_epipolar_crossview_smoke_wsl.sh` | Create | 5-epoch smoke shell script |

### 3.3 Loss / data / optimizer changes

- **Loss:** unchanged (MPJPE + optional PP/focal direct loss if available).
- **Data:** unchanged; uses the same MPI-INF-3DHP loader as the PP anchor.
- **Optimizer:** unchanged Adam/AdamW schedule.
- **Augmentation:** unchanged; the epipolar bias is recomputed on the fly from augmented cameras, so it is robust to camera perturbation curriculum.

### 3.4 Exact insertion point

In `ray_attention_temporal_crossview_residual_epipolar_model.py`, after line 122 of the parent model:

```python
feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
```

the new forward will:
1. Reshape `feat` back to `(B, T, V, J, d)` temporarily.
2. Compute `epi_dist` of shape `(B*T, V, V, J)`.
3. Reshape `epi_dist` to `(B*J, T*V, T*V)` so entries between views within each frame are populated and all cross-frame entries are zero (temporal attention remains unbiased).
4. Add `epi_bias` to the attention logits inside each transformer layer using the layer's attention mask hook or by overriding the layer's `forward`.

If hooking each layer is too invasive, we can instead implement a custom `TransformerEncoderLayer` in a new module `motionflow_mv/fusion/epipolar_transformer_layer.py` and replace `self.st_transformer` with layers that accept an additive bias.

## 4. Smoke-Test Plan

Run a 5-epoch smoke on a 500-sample subset of MPI-INF-3DHP using the new model and the same hyperparameters as the current factorized PP smoke.

**Command:**

```bash
bash scripts/run_epipolar_crossview_smoke_wsl.sh
```

**The script executes:**

```bash
python experiments/train_epipolar_crossview_residual_mpiinf3dhp.py \
  --d 32 \
  --residual_hidden 64 \
  --n_epochs 5 \
  --max_samples 500 \
  --batch_size 2 \
  --cam_aug_schedule intrinsics_curriculum \
  --out_dir outputs/epipolar_crossview_smoke
```

**Pass / fail criteria:**

- **Pass:** training finishes in < 30 minutes, no NaNs, and validation MPJPE ≤ 60 mm on the 500-sample smoke (the factorized PP smoke lands ~57.7 mm; this is a sanity bound, not a target).
- **Pass:** the epipolar bias tensor is non-zero and finite for at least 90% of joint/view triplets.
- **Pass:** epipolar distance computation runs without degenerate-camera crashes (determinant / rank checks pass).
- **Fail:** training crashes, NaN loss, or validation MPJPE > 80 mm (clear regression against the factorized PP smoke).

## 5. Evaluation Plan

After a successful smoke, run the same evaluation pipeline as the PP anchor:

1. **Clean metrics:** `python experiments/eval_full_metrics.py --model epipolar_crossview --checkpoint outputs/epipolar_crossview_smoke/best.pth`
   - Report MPJPE, PA-MPJPE, PCK@50/100/150, AUC.
   - Target: within 5% of the 9.32 mm anchor on the full clean val split.

2. **Robustness matrix:** `python experiments/eval_robustness_matrix_pp_mpiinf3dhp.py --model epipolar_crossview --checkpoint outputs/epipolar_crossview_smoke/best.pth --n_clips 20`
   - Compare 6-axis corruption MPJPE against the PP anchor.
   - Hypothesized win axes: `focal_*`, `cxcy_*`, `rot_*` (any corruption that moves the epipolar geometry).

3. **Diagnostic ablation:** re-run the smoke with `temperature = inf` (zero bias) to isolate the contribution of the epipolar term from architectural noise.

4. **Latency check:** `python experiments/benchmark_runtime.py --model epipolar_crossview`
   - Ensure epipolar distance computation adds < 10% latency; if larger, move distance computation to a cached pre-pass.

## 6. Estimated GPU/CPU Cost on RTX 4090

- **Smoke (5 epochs, 500 samples):** ~25–35 minutes on RTX 4090, including the extra `(V, V)` pairwise distance loop. The geometry is pure PyTorch and runs on the same GPU; no CPU bottleneck added.
- **Full eval on clean val split (~10k frames):** ~2 minutes GPU inference + ~1 minute CPU metrics.
- **Robustness matrix (20 clips):** ~5 minutes GPU + ~2 minutes CPU.
- **Memory:** the epipolar bias adds a `(B*J, T*V, T*V)` float tensor during attention, which for `B=2, T=9, V=4, J=17` is small (~0.5 MB). No OOM risk for the smoke batch.

## 7. Risks & Fallback

| Risk | Impact | Mitigation / Fallback |
|------|--------|----------------------|
| Degenerate or near-degenerate camera pairs (same viewpoint, baseline ~0) cause infinite epipolar lines | NaN loss | Clamp distance, skip bias for pairs with baseline < ε, fallback to unbiased attention |
| Epipolar bias is too strong and suppresses useful cross-view attention | Clean accuracy drops > 5% | Treat `temperature` as a learnable scalar initialized large, or add a residual gate `α * bias + (1-α) * 0` with `α` learned |
| Distance computation is slower than expected | Latency budget exceeded | Pre-compute fundamental matrices per frame and cache; vectorize over joints |
| No improvement on clean or robust metrics | Negative result | Archive the model but do not integrate; use the geometric insight to instead bias the **weight head** (`w_logits`) rather than attention, which is a one-line change in the parent model |
| Integration into the existing transformer is invasive | Risk of breaking the 9.32 mm anchor | Keep the parent model untouched; implement via subclass and only override the transformer construction in the new file |

## 8. Self-Evolution Mapping

- **Reflect:** Cross-view attention is geometry-agnostic; epipolar constraints are a free inductive bias in calibrated multi-view settings.
- **Hypothesize:** Adding epipolar-line distance as an attention bias improves geometric consistency and robustness to intrinsic/extrinsic perturbation.
- **Smoke-validate:** 5-epoch smoke checks training stability and clean sanity metrics.
- **Integrate:** If clean accuracy holds and robustness improves on any axis, promote the bias as a default in the cross-view attention branch and backport to the factorized variant.
