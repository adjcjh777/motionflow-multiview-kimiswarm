# v51 Self-Supervised Multi-View Pose Pretraining (SSMVP)

## Focus-area alignment

v50 closes the self-evolution loop inside the supervised training graph. v51 asks whether the same backbone can learn useful multi-view geometry before seeing 3-D ground truth. Self-supervised pretraining targets the remaining weak points — sparse-view generalization and cross-domain transfer — without adding inference cost. The idea is to pretrain the v46/v50 encoder on unlabeled multi-view video (WebBridge raw, unlabeled H36M/MPI, and 3DPW actual) using view reconstruction and temporal/epipolar consistency as free supervision, then fine-tune on the supervised task.

## Architecture

**Module**: `SelfSupervisedMultiViewPretrainerV51` → `motionflow_mv/training/self_supervised_pretrainer_v51.py`

The pretrainer wraps the v50 backbone and appends three lightweight heads:

1. **Masked-view 2-D keypoint reconstruction head** — predicts the 2-D keypoints of randomly masked camera views from the visible views, reusing the existing cross-view attention and geometry-fusion layers.
2. **Temporal consistency head** — enforces smoothness between adjacent frames' predicted 3-D poses.
3. **Epipolar consistency head** — projects the predicted 3-D pose back to all visible views and penalizes epipolar-line distance violations.

The downstream 3-D pose regression head is detached during pretraining. After pretraining, the backbone is loaded into the supervised trainer and the pose head is trained from scratch. All pretraining heads are zero-initialized so that the module can be skipped without affecting the v50 warm-start path.

## New config flags with defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_self_supervised_pretraining` | bool | `False` |
| `v51_ss_pretraining_epochs` | int | `5` |
| `v51_ss_mask_view_prob` | float | `0.3` |
| `v51_ss_mask_joint_prob` | float | `0.1` |
| `v51_ss_temporal_window` | int | `3` |
| `v51_ss_reconstruction_weight` | float | `1.0` |
| `v51_ss_temporal_consistency_weight` | float | `0.5` |
| `v51_ss_epipolar_weight` | float | `0.3` |
| `v51_ss_finetune_lr` | float | `1e-4` |
| `v51_ss_unlabeled_data_sources` | list | `["webbridge_raw", "h36m_unlabeled", "mpi_unlabeled", "3dpw_actual"]` |

## Loss term

```
L_ss = λ_recon * L_2d_recon + λ_temp * L_temporal + λ_epi * L_epipolar

L_2d_recon = mean over masked (view, joint) of ||K_vj − K_vj||²
L_temporal = mean over adjacent frames of ||P_t − P_{t+1}||²
L_epipolar = mean epipolar-line distance between predicted 3-D point and reprojection
```

`L_2d_recon` dominates; the other terms regularize geometry. No 3-D ground truth is used during pretraining.

## Evaluation metric

- **Pretraining stage**: masked 2-D keypoint reconstruction error (pixels), temporal consistency error (mm), and epipolar distance (pixels).
- **After fine-tuning**: `MPJPE@k` for `k = 2, 3, 4, full`; per-domain `MPJPE@k`; 3DPW actual `MPJPE@2` and `MPJPE@3`.

## Expected MPJPE impact

| Metric | Expected change |
|---|---|
| `MPJPE@2` | −3 to −5 mm |
| `MPJPE@3` | −2 to −4 mm |
| `MPJPE@full` | −0.5 to −1.0 mm |
| 3DPW actual `MPJPE@2` | −4 to −7 mm |

Largest gains are expected on 3DPW actual, where unlabeled multi-view video can be exploited.

## Main risk

**Risk**: The pretraining task may not transfer to supervised pose estimation if the masking distribution or unlabeled domain differs too much from the downstream task, or if the reconstruction loss dominates early gradients.

**Mitigation**: (1) match view-masking to v46's `v46_svg_view_dropout_prob`; (2) keep the pose head frozen during pretraining; (3) warm up with `λ_recon = 1.0` and other weights at `0.0`, then gradually increase them; (4) run a 1-epoch local RTX 4090 smoke before committing to a full A800 run.

## Smoke-test plan

- **Config**: `configs/benchmark_v51_ssmvp_smoke.yaml`
- **Command**: `bash scripts/run_v51_ssmvp_smoke_local_4090.sh`
- **Acceptance**: pretraining loss decreases for 1 epoch, and fine-tuned `MPJPE@full` is within 1.5 mm of the v50-only baseline.
