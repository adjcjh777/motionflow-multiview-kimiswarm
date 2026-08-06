# Implementation Plan: Adaptive View Selection

## Phase 1 — Core Selector Module (1 day)

**File:** `motionflow_mv/fusion/adaptive_view_selector.py`

Implement a standalone, reusable `AdaptiveViewSelector`:

- `__init__(d, n_views, k=4, tau=0.5, hard_inference=True, geo_features=True)`
- `_geometry_features(points_2d, K, R, t) -> (N, V, J, 3)` — ray angles, baselines, and triangulation stability proxy.
- `forward(feat, points_2d, K, R, t) -> (soft_mask, hard_mask, scores)`
  - During training: Gumbel-softmax over `V` views per joint.
  - During inference: hard top-`k` mask, straight-through gradient.

## Phase 2 — Model Integration (1 day)

**File:** `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_adaptive_v1_model.py`

Subclass `RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1`:

1. After the spatio-temporal transformer (`feat` shaped `(B*T, V, J, d)`):
   - Pass `feat` to `AdaptiveViewSelector`.
   - Multiply uncertainty weights by the selected hard mask before DLT.
2. Return the selection mask alongside `pred_3d`, `weights`, `log_var`, and `nll_loss`.
3. Add `selection_loss` composed of:
   - Budget loss: `((hard_mask.sum(dim=1) - k).pow(2)).mean()`
   - Entropy regularizer: `-(p * log(p + 1e-8)).sum(dim=-1).mean()` to keep decisions sharp but not prematurely peaked.

## Phase 3 — Training Script (1 day)

**File:** `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_adaptive_v1_mpiinf3dhp.py`

Mirror `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_mpiinf3dhp.py` with these additions:

- CLI flags: `--k 4`, `--selection_loss_weight 0.1`, `--tau 0.5`, `--geo_features`.
- Loss: `loss = mse + 0.1 * nll_loss + λ_sel * selection_loss`.
- Resume/start-epoch support already exists in recent commits; keep it.

## Phase 4 — Evaluation (1–2 days)

**File:** `experiments/eval_adaptive_view_selection_mpiinf3dhp_v1.py`

Extend `eval_residual_robustness_mpiinf3dhp_v1.py`:

- Run clean validation at `k ∈ {2, 3, 4}`.
- Inject 0%, 10%, 30%, 50% occlusion and compare adaptive selector vs. fixed `k`.
- Log per-joint selected view counts and mask accuracy vs. injected occlusion.

## Deliverables & Exact Paths

| Deliverable | Path |
|-------------|------|
| Selector module | `motionflow_mv/fusion/adaptive_view_selector.py` |
| Model | `motionflow_mv/fusion/ray_attention_temporal_crossview_uncertainty_residual_adaptive_v1_model.py` |
| Trainer | `experiments/train_ray_attention_temporal_crossview_uncertainty_residual_adaptive_v1_mpiinf3dhp.py` |
| Evaluator | `experiments/eval_adaptive_view_selection_mpiinf3dhp_v1.py` |
| Test/sanity | `tests/test_adaptive_view_selector.py` |
| Report (this) | `docs/swarm_iter_next/design_adaptive_view_selection/` |

## Timeline

- Day 1: selector module + model integration.
- Day 2: training script + smoke test on 1–2 epochs.
- Day 3: evaluation script + robustness sweep.
- Day 4: analysis and decision to scale to H36M/AIST++ if MPI-INF-3DHP gains materialize.
