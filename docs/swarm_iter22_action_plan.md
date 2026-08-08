# Swarm Iteration 22 — Kinematic Anthropometric Prior (KAP)

**Goal:** Add a small, SMPL-free kinematic/anthropometric prior to `OmniMultiViewFusionV5` and validate it on the A800.  
**Status:** plan  
**Tracking issue:** #88  

## v22 scope

A single new optional head, `KinematicAnthropometricPrior`, that refines the per-frame 3D pose using a learned bone-length prior and a soft joint-angle limit. It is gated by `use_kinematic_anthropometric_prior_v22` inside `OmniMultiViewFusionV5` and stacks with the existing v18/v19/v21 modules.

## Implementation checklist

### T01 — Implement the KAP head

*Owner:* coder  
*File:* `motionflow_mv/fusion/kinematic_anthropometric_prior_v22.py`  
*Task:* Implement `KinematicAnthropometricPrior`.
  - Support `J=17` (H36M) and `J=28` (MPI-INF-3DHP) via `H36M_17_PARENTS` / `MPI_INF_3DHP_28_PARENTS`.
  - Learn per-bone `bone_mu` and `bone_logvar`.
  - Compute bone-length NLL loss.
  - Optional `joint_limit_loss` from `motionflow_mv/losses/kinematic_v15.py`.
  - Predict per-joint residual `delta` and confidence `conf`; refine pose as `x + conf * delta`.
*Success:* CPU smoke test runs with `B=2, T=9, V=4, J=17` and backpropagates.

### T02 — Wire the flag into `OmniMultiViewFusionV5`

*Owner:* coder  
*File:* `motionflow_mv/fusion/omniview_fusion_v5.py`  
*Task:*
  - Add `use_kinematic_anthropometric_prior_v22: bool = False` and `kap_loss_weight: float = 0.01`.
  - Instantiate the head when the flag is on.
  - Insert the call after the residual/diffusion branch and after v21, before the final v4 kinematic refiner and before v19 reshape.
  - Add `kap_loss * kap_loss_weight` to `epi_loss`.
*Success:* `python motionflow_mv/fusion/omniview_fusion_v5.py` runs with the flag both on and off.

### T03 — Update the v5 WebBridge trainer

*Owner:* coder  
*File:* `experiments/train_omniview_fusion_v5_webbridge_multi.py`  
*Task:*
  - Add CLI flags:
    - `--use_kinematic_anthropometric_prior_v22`
    - `--kap_loss_weight`
    - `--kap_angle_limit_weight`
  - Pass them through `build_model_from_args`.
  - (Optional) extract and log `kap_loss` separately if the model returns it.
*Success:* `--smoke` run completes with `--use_kinematic_anthropometric_prior_v22`.

### T04 — Unit and integration tests

*Owner:* coder  
*Files:*
  - `tests/test_kinematic_anthropometric_prior_v22.py` (new)
  - `tests/test_omniview_fusion_v5.py` (extend)
*Task:*
  - Test forward/backward, shape, identity at init, confidence range, J=17 and J=28.
  - Add a full v5 toggle-on case to `test_omniview_fusion_v5.py`.
*Success:* `pytest tests/test_kinematic_anthropometric_prior_v22.py tests/test_omniview_fusion_v5.py -q` passes.

### T05 — Add a smoke config

*Owner:* coder  
*File:* `configs/benchmark_webbridge_kap_v22_smoke.yaml`  
*Task:* Copy the existing smoke config and add:
  ```yaml
  use_kinematic_anthropometric_prior_v22: true
  kap_loss_weight: 0.01
  kap_angle_limit_weight: 0.1
  ```
*Success:* Config loads without errors.

### T06 — A800 CPU smoke

*Owner:* coder  
*Command:*  
```bash
python motionflow_mv/fusion/omniview_fusion_v5.py
pytest tests/test_kinematic_anthropometric_prior_v22.py tests/test_omniview_fusion_v5.py -q
python experiments/train_omniview_fusion_v5_webbridge_multi.py --smoke --use_kinematic_anthropometric_prior_v22
```
*Success:* All three complete without NaNs or shape errors.

### T07 — A800 GPU smoke run

*Owner:* coder  
*Command:*  
```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --manifest configs/benchmark_webbridge_kap_v22_smoke.yaml \
    --use_kinematic_anthropometric_prior_v22 \
    --epochs 2 \
    --batch_size 4 \
    --d 64 \
    --residual_hidden 128
```
*Success:* Completes 2 epochs; training and validation MPJPE are logged; no OOM.

### T08 — Full A800 launch

*Owner:* coder  
*Command:*  
```bash
python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_deformable_cross_view_attention_v18 \
    --use_temporal_perceiver_v19 \
    --use_neural_bundle_adjustment_v21 \
    --use_kinematic_anthropometric_prior_v22 \
    --d 128 \
    --residual_hidden 256 \
    --epochs 30
```
*Success:* Training starts on an A800 GPU; checkpoint saved; MPJPE trend is tracked in issue #88.

### T09 — Evaluation and ablation

*Owner:* coder  
*File:* `experiments/eval_omniview_fusion_v5_webbridge_multi.py` (extend)  
*Task:* Evaluate the v18+v19+v21+KAP model against the v18+v19+v21 baseline. Report:
  - Clean MPJPE / PA-MPJPE
  - Variable-view MPJPE@k
  - Per-joint error
*Success:* A results table is posted to issue #88.

### T10 — Documentation

*Owner:* coder  
*Files:*
  - `docs/proposals/v22_kinematic_anthropometric_prior.md`
  - `docs/swarm_iter22_action_plan.md`
*Task:* Keep the proposal and action plan up to date as implementation details settle.

## Cross-package references

- `motionflow_mv/fusion/omniview_fusion_v5.py` — integration point.
- `motionflow_mv/fusion/deformable_cross_view_attention.py` — v18.
- `motionflow_mv/fusion/temporal_perceiver_v19.py` — v19.
- `motionflow_mv/fusion/neural_bundle_adjustment_v21.py` — v21.
- `motionflow_mv/fusion/graph_joint_relation.py` — parent lists.
- `motionflow_mv/losses/kinematic_v15.py` — `joint_limit_loss`.
- `experiments/train_omniview_fusion_v5_webbridge_multi.py` — trainer.

## Definition of done

- [ ] T01–T05 are implemented and committed.
- [ ] CPU smoke tests in T06 pass.
- [ ] A800 GPU smoke in T07 completes without OOM.
- [ ] At least one full training run in T08 is queued or running on A800.
- [ ] Evaluation table from T09 is posted to issue #88.
- [ ] This action plan and the proposal are merged to `main`.
