# v54 Multi-View Temporal Sync (MVTS)

## Module Title
`multi_view_temporal_sync_v54` — cross-view temporal synchronization for v52/v53 pose refinement.

## Motivation

The v52 Uncertainty-Weighted Triangulation (UWT) and v53 Physical-Space Calibration (PSC) modules improve per-frame 3-D pose by re-weighting camera views and enforcing floor/bone constraints. However, both operate largely frame-by-frame: a noisy or occluded view at frame *t* is not repaired by information from *t±1*, and brief view dropouts create jitter in the final sequence.

v54 closes that temporal loop. It treats the per-view, per-joint UWT weights as a **temporal synchronization signal**: a high-confidence view at a neighboring frame can guide the current frame, and a low-confidence view can be temporally inpainted. The module is placed **after v53 in `OmniMultiViewFusionV5`**, so it refines the already-calibrated physical-space pose while preserving the warm-start/identity-at-init property of the v52/v53 chain.

## Architecture

v54 is a lightweight transformer that attends over `(time, view)` tokens for each joint, conditioned on the current 3-D pose and the UWT reliability weights. It predicts a small per-joint 3-D residual that is added to the input pose.

```
Input
  pred_3d  : (B, T, J, 3)   <- output of v53 PSC
  weights  : (B, T, V, J)   <- v52 UWT weights (or ones if v52 disabled)
  view_mask: (B, T, V)      <- valid view mask
  K, R, t  : camera params  <- for geometry-aware position encoding

Temporal reliability encoder
  -> per-(t, v, j) token (B, T, V, J, d)

Cross-view temporal attention (per joint)
  Q : (B, J, T, d)  from current frame/joint
  K : (B, J, T, d)  from all frames
  V : (B, J, T, d)  from all frames
  -> temporal sync features (B, T, J, d)

Gated residual MLP
  -> offset : (B, T, J, 3)
  -> pred_sync = pred_3d + sigmoid(gate) * offset
```

The attention is performed **per joint**, not across joints, to keep the head count low and preserve interpretability. A small camera-ray position encoding is added to each token so that the module respects epipolar geometry when propagating information across views.

## Key Equations

Given the input pose `X_t \in \mathbb{R}^{J \times 3}` and UWT weights `w_{t,v,j} \in [0,1]`:

1. Reliability-normalized feature:
   ```
   r_{t,v,j} = w_{t,v,j} / (\sum_v w_{t,v,j} + \epsilon)
   ```

2. Per-joint temporal query/key/value:
   ```
   Q_j = W_q [X_t^j ; r_{t,:,j} ; \phi_{ray}(K,R,t)]
   K_j = W_k [X_\tau^j ; r_{\tau,:,j} ; \phi_{ray}(K,R,t)]
   V_j = W_v [X_\tau^j ; r_{\tau,:,j}]
   ```
   where `\tau` indexes all temporal positions (or a fixed window) and `\phi_{ray}` is the ray-direction embedding.

3. Sync feature and residual:
   ```
   F_j = softmax(Q_j K_j^T / \sqrt{d}) V_j
   \Delta X_t = MLP(F_j)
   \hat{X}_t = X_t + \sigma(g) \cdot \Delta X_t
   ```
   `g` is a learned gate initialized so `\sigma(g) \approx 0` at init.

4. Auxiliary loss:
   ```
   L_sync =  L_temporal + L_reproj + L_identity
   L_temporal = \sum_{t,j} \bar{w}_{t,j} || \hat{X}_{t+1}^j - \hat{X}_t^j ||^2
   L_reproj   = \sum_{v,j} w_{t,v,j} || \Pi_v(\hat{X}_t^j) - \hat{p}_{t,v}^j ||^2
   L_identity = ||\Delta X_t||^2
   ```
   `\bar{w}_{t,j}` averages the per-view weights and acts as a robustness weight for the smoothness term.

## Inputs / Outputs

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pred_3d` | `(B, T, J, 3)` | Pose from v53 PSC (or v52 if PSC disabled). |
| `uwt_weights` | `(B, T, V, J)` | v52 per-view triangulation weights; ones if v52 off. |
| `view_mask` | `(B, T, V)` | Boolean valid-view mask. |
| `K, R, t` | `(B, T, V, 3, 3)`, `(B, T, V, 3)` | Calibrated camera parameters. |
| **Output** `pred_3d_sync` | `(B, T, J, 3)` | Temporally synchronized pose. |
| **Output** `sync_loss` | scalar | Auxiliary temporal/reproj consistency loss. |

## Config Flags

```yaml
use_v54_multi_view_temporal_sync: false
v54_mvts_hidden: 64
v54_mvts_n_layers: 2
v54_mvts_n_heads: 4
v54_mvts_window: null              # null = full sequence; else causal window size
v54_mvts_identity_init: true
v54_mvts_residual_gate_init: -6.0  # sigmoid(-6) ≈ 0.0025 -> identity at init
v54_mvts_use_uwt_weights: true
v54_mvts_temporal_loss_weight: 0.01
v54_mvts_reproj_loss_weight: 0.1
v54_mvts_identity_loss_weight: 0.001
v54_mvts_warmup_epochs: 0
```

## Expected MPJPE Impact

- **Smoke test (RTX 4090):** within `0.1 mm` of the v53 baseline at epoch 0 / identity init; within `1 mm` after a short smoke run.
- **Full A800:** expected **1–3 mm MPJPE reduction** on sequences with motion blur, partial occlusion, or variable view counts, because the module temporally borrows clean views.
- **No regression:** because of the gated residual and zero-initialized output layer, disabling the loss or loading a v53 checkpoint into a v54-enabled model should change MPJPE by less than `0.1 mm`.

## Risks and Mitigations

(See `docs/swarm_iter28/reports/agent_multi_view_temporal_sync_risks.md`.)

## 5-Step Implementation Plan

1. **Module implementation.** Create `motionflow_mv/fusion/multi_view_temporal_sync_v54.py` with `MultiViewTemporalSyncV54(nn.Module)`. Implement the reliability encoder, per-joint temporal attention, and gated residual MLP. Zero-initialize the final residual layer and set `residual_gate_init=-6.0` for identity-at-init.

2. **Integration into `OmniMultiViewFusionV5`.** In `motionflow_mv/fusion/omniview_fusion_v5.py`, add the constructor flags, instantiate the module after the v53 PSC block, and call it in the forward pass. Re-use the existing `view_mask_flat` and `domain_id` plumbing. Store the auxiliary loss in `self._v54_mvts_loss` and add it to `epi_loss` with `v54_mvts_warmup_epochs` gating.

3. **YAML config and smoke script.** Add the flags to `configs/benchmark_v53_physical_space_calibration_smoke.yaml` (or a new `benchmark_v54_mvts_smoke.yaml`) and create `scripts/run_v54_mvts_smoke_local_4090.sh`.

4. **Identity-at-init verification.** Run a smoke forward pass with v54 enabled but untrained; assert `|MPJPE_v54 - MPJPE_v53| < 0.1 mm`. Then run a 1-epoch smoke and confirm no NaN/Inf and a finite loss.

5. **Ablation run on A800.** Queue a full A800 run (`v54_mvts_on_v53` in `scripts/launch_v33_a800_queue.py`) and compare against the v53 baseline. Report `MPJPE@full` and `MPJPE@k` for sparse views, and update `docs/swarm_iter28/status.md`.
