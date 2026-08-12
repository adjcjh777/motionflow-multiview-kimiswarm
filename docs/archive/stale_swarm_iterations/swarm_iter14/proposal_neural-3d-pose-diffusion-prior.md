# Neural 3D Pose Diffusion Prior: Refine 3D Pose with a Lightweight Diffusion Model

## 1. Problem

The anchor `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` reaches 9.32 mm MPJPE on MPI-INF-3DHP clean, but its output 3-D skeletons still contain structured, non-rigid errors (e.g., implausible bone lengths, jittery joints, occasional flipped limbs) that a small generative prior over plausible human poses could suppress without retraining the full multi-view encoder.

## 2. Hypothesis

A lightweight diffusion model trained as a pose refiner on top of the anchor’s triangulated 3-D joints can reduce residual pose errors and improve robustness under mild camera perturbations, with negligible added inference cost after a small number of deterministic denoising steps.

## 3. Method

### 3.1 Architecture

Add a post-hoc **3-D pose diffusion refiner** (`PoseDiffusionRefiner`) that operates only on the 3-D joint coordinates produced by the existing anchor model.

**New files:**

- `motionflow_mv/models/pose_diffusion_refiner.py`
  - `PoseDiffusionRefiner(nn.Module)`
    - Input: `J x 3` predicted 3-D pose (anchor output), optionally with per-joint confidence.
    - Backbone: small transformer encoder (2 layers, 4 heads, `d_model=128`) over joints.
    - Output: denoised residual `ΔP` added to the input pose, plus optional per-joint uncertainty.
    - Diffusion: standard DDPM-style forward noising of the residual, with the network predicting the noise `ε_θ(P_t, t)`.
  - `DiffusionSchedule`: cosine beta schedule, `T=50` train steps, inference with 5 deterministic DDIM steps.
- `experiments/train_pose_diffusion_refiner_mpiinf3dhp.py`
  - Loads frozen anchor checkpoint, generates initial 3-D poses from the train/val split, and trains the refiner on pose residuals with respect to ground truth.
- `experiments/eval_pose_diffusion_refiner_mpiinf3dhp.py`
  - Applies the refiner to val/test poses and reports standard 3-D pose metrics.

**Files to modify (exact edits, not committed without smoke pass):**

- `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_model.py`: add optional `pose_refiner: Optional[PoseDiffusionRefiner] = None` argument and call it after the final 3-D pose head when provided.
- `experiments/eval_full_metrics.py`: add `--refiner_checkpoint` flag to load and optionally apply the refiner during evaluation.

### 3.2 Loss

Training loss (per sample):

```
L_noise = E_{t, ε ~ N(0,1)} || ε_θ(P_anchor + ε·sqrt(α_t), t) - ε ||²
L_pose  = || (P_anchor + ΔP_0) - P_gt ||_1
L       = L_noise + λ L_pose          (λ = 0.1)
```

Add an small **bone-length consistency** loss as an auxiliary signal (no new annotations needed; skeleton graph is available from `motionflow_mv/data/skeleton.py`):

```
L_bone = mean( || bone_lengths(P_anchor + ΔP_0) - bone_lengths(P_gt) ||² )
```

### 3.3 Data

- Use the existing MPI-INF-3DHP train split.
- For each sample, the anchor model produces the initial 3-D pose; ground truth is the canonical 3-D pose.
- No new datasets; no modification of the multi-view loader.

### 3.4 Integration

At inference, the anchor model runs once; the refiner runs 5 DDIM steps. The refiner is independent and can be disabled by passing `pose_refiner=None`, making it a safe, additive module.

## 4. Smoke-Test Plan

Run a 5-epoch smoke on a small subset (500 samples) of MPI-INF-3DHP.

- **Command:**
  ```bash
  python experiments/train_pose_diffusion_refiner_mpiinf3dhp.py \
      --anchor_checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point/best.ckpt \
      --dataset mpiinf3dhp \
      --max_samples 500 \
      --epochs 5 \
      --batch_size 16 \
      --output_dir tmp/pose_diffusion_refiner_smoke
  ```

- **Pass criteria:**
  - Training completes without NaNs/Inf.
  - Val MPJPE of the refined pose is ≤ 9.4 mm (within ~1% of the 9.32 mm anchor; the diffusion refiner should not degrade accuracy).
  - Per-joint residual magnitude is visually plausible (no large spikes).

- **Fail criteria:**
  - Val MPJPE > 9.6 mm or any NaN/Inf.
  - Refiner output diverges (joints far outside the capture volume).

## 5. Evaluation Plan

Evaluate on the same MPI-INF-3DHP clean val/test split used for the anchor.

- **Metrics:** MPJPE, PA-MPJPE, PCK@50/100/150, AUC.
- **Scripts:**
  - `experiments/eval_pose_diffusion_refiner_mpiinf3dhp.py --refiner_checkpoint <...>`
  - Compare against the anchor alone using `experiments/eval_full_metrics.py --model ray_attention_temporal_crossview_residual_principal_point`.
- **Robustness smoke:** run the refiner on the 6-axis robustness matrix (20 clips) to check if it reduces degradation under `cxcy_3px` and `focal_2pct` conditions.
- **Success criterion:** ≥ 0.2 mm MPJPE improvement on clean and no regression > 0.3 mm on any corruption axis versus the anchor alone.

## 6. Estimated GPU/CPU Cost on RTX 4090

- **Training:** ~5–10 minutes for the 5-epoch smoke on 500 samples (lightweight transformer, small batches; GPU memory < 2 GB).
- **Full training:** ~30–60 minutes on the full MPI-INF-3DHP train split.
- **Inference:** ~5 ms per sample for 5 DDIM steps (negligible vs. the multi-view anchor).
- **CPU:** data loading and metric computation are negligible; no CPU bottleneck expected.

## 7. Risks & Fallback

| Risk | Mitigation / Fallback |
|------|----------------------|
| Diffusion refiner overfits to anchor biases and does not improve on val. | Keep the refiner small (90 k params) and train on residuals; fall back to the raw anchor output if val improvement < 0.1 mm. |
| Added denoising steps hurt runtime or introduce latency. | Use only 5 DDIM steps; fallback to 1-step deterministic estimate (T=0) if latency is a concern. |
| Pose-space prior conflicts with camera-geometry corrections already learned by the anchor. | Make the refiner residual and optional; disable it without changing the anchor weights. |
| Smoke shows no gain because anchor errors are already below the diffusion model’s resolution. | Abandon the diffusion branch and instead use the same transformer as a deterministic pose-MLP refiner (same architecture, no diffusion). |
