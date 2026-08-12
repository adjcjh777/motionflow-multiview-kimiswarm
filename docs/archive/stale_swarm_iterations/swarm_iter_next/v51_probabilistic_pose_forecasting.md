# v51 Probabilistic Pose Forecaster

## Module proposal

**`ProbabilisticPoseForecasterV51`** → `motionflow_mv/fusion/probabilistic_pose_forecaster_v51.py`

### Architecture

A lightweight causal temporal head that predicts a distribution over future 3-D poses given the recent pose sequence. It consumes the last `v51_ppf_history_frames` pose estimates, encodes them with a small causal Transformer or masked MLP, and outputs Gaussian parameters (mean `μ` and log-variance `log σ²`) for the next `v51_ppf_future_frames` frames. The mean is initialized to a zero-velocity forecast (repeat last known pose), so the module is identity-at-start and cannot regress the strong v46/v47 baseline. When `v51_ppf_use_aleatoric=True`, the predicted `σ` is used as a per-joint uncertainty scale that gates the supervised loss on the current frame. For sparse-view scenarios, the forecast prior is precision-fused with the next-frame triangulation evidence, giving missing or noisy views a temporal prior to fall back on.

### Config flags and defaults

| Flag | Type | Default |
|---|---|---|
| `use_v51_probabilistic_pose_forecasting` | bool | `False` |
| `v51_ppf_history_frames` | int | `4` |
| `v51_ppf_future_frames` | int | `2` |
| `v51_ppf_hidden` | int | `64` |
| `v51_ppf_num_layers` | int | `2` |
| `v51_ppf_n_heads` | int | `4` |
| `v51_ppf_dropout` | float | `0.1` |
| `v51_ppf_loss_weight` | float | `0.01` |
| `v51_ppf_use_aleatoric` | bool | `True` |
| `v51_ppf_use_future_reproj_consistency` | bool | `False` |
| `v51_ppf_identity_init` | bool | `True` |
| `v51_ppf_reproj_weight` | float | `0.5` |

### Loss

`L_ppf = v51_ppf_loss_weight * ( L_nll(future_pose; μ, σ) + λ * L_smooth + γ * L_reproj )`

- `L_nll`: negative log-likelihood of the ground-truth future pose under the predicted Gaussian.
- `L_smooth`: L2 penalty on second-order temporal differences of `μ` to reduce jitter.
- `L_reproj` (optional, when `v51_ppf_use_future_reproj_consistency=True`): reprojection of the forecast mean onto available next-frame 2-D keypoints.

### Evaluation metric

Primary: `val_MPJPE@k` for `k = 2, 3, 4, full` on the current frame. Secondary (tracked in logs): `MPJPE_future@1` and `NLL_future` on the next `v51_ppf_future_frames` frames, plus Spearman correlation between predicted `σ` and actual per-joint error. The v49 real-time streaming head may also report `MPJPE_streaming@1` with the forecaster providing the prior.

### Expected MPJPE impact

Full-view baseline should move by less than ±0.5 mm because the forecaster is identity-at-init and low-weighted. Sparse-view gains are expected to be larger: **MPJPE@2 −1 to −2 mm**, **MPJPE@3 −0.5 to −1 mm**, driven by the temporal prior regularizing noisy triangulation. The largest payoff is downstream in v49 streaming, where a probabilistic forecast lets the system hold a low-latency prior when cameras drop or run late.

### Main risk

The forecaster can overfit to short-term motion and dominate the gradient budget, destabilizing the already-good v46/v47 baseline. Mitigation: identity-at-init, loss weight starting at `0.001`, freeze base weights for the first epoch, and clamp `log σ²` to `[-3, 3]`. A second risk is the need for contiguous clip data; the existing clip loader already satisfies this, but future-frame labels require extending the temporal window by `v51_ppf_future_frames` frames.

### Why this fits v51

v50 closes the self-evolution loop on *current-frame* reliability. v51 extends that loop into the *temporal* dimension: the model learns not only to critique its present estimate, but to forecast a distribution over where the body will be. That directly serves the ICRA/CVPR 2027 story of sparse-view, cross-domain, real-time pose estimation and provides a principled bridge to the v49 streaming head.
