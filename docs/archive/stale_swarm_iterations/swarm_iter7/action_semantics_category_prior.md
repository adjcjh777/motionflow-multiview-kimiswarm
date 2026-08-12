# Direction 17: Action Semantics & Category Prior

## Problem statement

The current best model, `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`, is action-agnostic: a walking clip and a sitting clip receive exactly the same triangulation and residual refinement weights. On Human3.6M this ignores strong action-specific kinematic priors—e.g., “SittingDown” has very different hip/knee configurations from “Walking”—which should let a small action embedding specialize the network per category and reduce per-action MPJPE. The risk is that 16 small action classes may not provide enough signal to beat a strong geometric baseline, so the next step must be the smallest possible architecture change with an fast CPU sanity check before any GPU training.

## Simplest concrete next experiment

Add a learned `nn.Embedding(num_actions, d)` to the principal-point model and broadcast it over the spatio-temporal features before the cross-view transformer. Then smoke-test the modified model on the existing single-action H36M `.npz` on CPU. The full H36M training comparison (action-aware vs. action-agnostic baseline) must wait for the RTX 4090 queue; this iteration only produces the model skeleton, a CPU smoke script, and a ready-to-run training plan.

## Files to touch / rough diff

### New: `motionflow_mv/fusion/action_aware_principal_point_model.py`

Subclass the PP model, add an per-action embedding, and inject it before the spatio-temporal transformer:

```python
class ActionAwareRayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    def __init__(self, num_actions: int = 16, action_embed_dim: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.action_embed_dim = action_embed_dim or self.d
        self.action_embed = nn.Embedding(num_actions + 1, self.action_embed_dim)

    def forward(self, x, action_id=None, cameras=None, K=None, R=None, t=None):
        # ... same setup as parent up to _extract_frame_features ...
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Inject action embedding
        feat = feat.view(B, T, V, J, self.d)
        action_emb = self.action_embed(action_id)  # (B, d)
        action_emb = action_emb.view(B, 1, 1, 1, self.d).expand(B, T, V, J, self.d)
        feat = (feat + action_emb).reshape(B * T, V, J, self.d)

        # ... remainder of parent forward ...
```

The embedding is added, not concatenated, so the rest of the model (attention, weight head, DLT, residual MLP) is unchanged.

### New: `experiments/smoke_action_aware_pp_h36m.py`

CPU-only smoke test that:
1. Loads `data/h36m_hf/s_01_act_02_multiview.npz` via the existing `ActionAwareRandomClipDataset`.
2. Builds the action-aware PP model with `d=32`, single ST layer.
3. Runs two forward/backward epochs on CPU and reports loss/MPJPE plus the parsed action distribution.

### Future (GPU queue)

- `experiments/train_action_aware_pp_h36m.py` / `scripts/run_action_aware_pp_h36m_wsl.sh`: train the action-aware model vs. the baseline on H36M S1 actions 02–16, holding out one action for validation, mirroring the existing `train_ray_attention_v4_h36m.py` setup.

## Expected success metric

For the **CPU smoke**: a clean forward/backward pass with non-exploding loss, and the model accepting `action_id` without touching existing runners.

For the **future GPU run**: on H36M subject 01, the action-aware model should reduce mean per-action MPJPE by ≥ 5% over the action-agnostic baseline, with larger relative gains on kinematically distinct actions (e.g., `SittingDown`, `WalkDog`). The baseline can be the current `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint` trained under the same data split.

## Compute requirement

- **CPU-only smoke**: completed below.
- **Full training**: requires GPU; do **not** run until the RTX 4090 queue is free.

## Command and result

Run the CPU smoke (no GPU, < 1 minute):

```bash
python experiments/smoke_action_aware_pp_h36m.py \
    --npz data/h36m_hf/s_01_act_02_multiview.npz \
    --n_samples 10 --batch_size 2 --epochs 2 --d 32
```

Output:

```text
Data: n_views=4, joints=17, clip_len=9, action_id=2
Action vocabulary size (max id): 16
Model parameters: 50,774
Action embedding parameters: 544
Epoch 1: loss=3292.546021, MPJPE=96.827312 m
Epoch 2: loss=1475.986499, MPJPE=64.361202 m

--- Per-file action distribution (frames) ---
  action_id= 2:   2995 frames

CPU smoke test completed successfully.
```

The loss decreases, the model consumes the action label, and no existing experiment runner was modified.

## Notes / follow-up

- The current H36M `.npz` files only carry a single action per file, so `ActionAwareRandomClipDataset` parses the action id from the filename. For the multi-action concatenated file (`s_01_acts_02_..._16_multiview.npz`), the loader will label every frame as action 2; future work should either split that file into per-action files or add a per-frame `action_id` array to the NPZ.
- GPU training must be queued after the currently running cross-view PP curriculum completes.
