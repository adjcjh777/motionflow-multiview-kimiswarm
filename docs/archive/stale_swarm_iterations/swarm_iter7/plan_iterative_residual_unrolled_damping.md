# Iterative Residual Refinement with Unrolled Training and Learned Damping

**Goal:** Push the MPI-INF-3DHP cross-subject val MPJPE below the current best 11.17 mm by making the residual refinement head stable for multiple iterations, instead of the single-pass used today.

**Why the previous iterative attempt failed:**  
In `RayAttentionFusionModelTemporalResidual`, running `n_iter > 1` at inference caused rapid divergence (13.12 mm → 40.99 mm → 71.15 mm). The residual MLP was trained only on the raw DLT pose; feeding it an already-refined pose shifts the input distribution, so it over-corrects.

**Idea:** Train the model with the iterative loop *unrolled* so the residual MLP learns to correct its own output, and add a learned per-joint damping scalar that scales each correction to prevent overshoot.

## What to change

1. **New model:** `motionflow_mv/fusion/ray_attention_temporal_residual_iterative_model.py`
   - Subclass `RayAttentionFusionModelTemporalResidual`.
   - In `forward`, replace the single residual pass with a loop of `n_iter` steps.
   - Share the same residual MLP across all steps.
   - Add a tiny gating branch that predicts a per-joint damping factor:
     ```python
     alpha = torch.sigmoid(gate([feat_pooled, current_pose]))  # (B*T, J, 1)
     delta = alpha * residual_mlp([feat_pooled, current_pose])
     current_pose = current_pose + delta
     ```
   - Initialize the gate bias so `alpha ≈ 0.3` at the start, making early iterations conservative.

2. **Training loss:** Unroll `n_iter` steps and supervise every step:
   ```python
   loss = MSE(pred_final, gt)
   for pred_k in intermediate_preds:
       loss += args.iter_loss_weight * MSE(pred_k, gt)
   ```
   Default: `n_iter=3`, `iter_loss_weight=0.5`.

3. **New training script:** `experiments/train_ray_attention_temporal_residual_iterative_mpiinf3dhp.py`
   - Mirrors `train_ray_attention_temporal_residual_mpiinf3dhp.py`.
   - Adds `--n_iter`, `--iter_loss_weight`, and `--residual_hidden`.

4. **Evaluation:** Extend `experiments/eval_ray_attention_temporal_residual_iterative.py` or add a `--model_module` flag so it can load the new class and evaluate `n_iter=1,2,3`.

## Smoke-test command

```bash
conda run -n mf python experiments/train_ray_attention_temporal_residual_iterative_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m_smoke.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 9 --d 32 --residual_hidden 64 \
    --n_iter 3 --iter_loss_weight 0.5 \
    --epochs 3 --batch_size 4 --train_samples 200 \
    --output outputs/ray_attention_temporal_residual_iterative_smoke.pth
```

## Full training command

```bash
conda run -n mf python experiments/train_ray_attention_temporal_residual_iterative_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --residual_hidden 128 \
    --n_iter 3 --iter_loss_weight 0.5 \
    --epochs 5 --batch_size 8 --train_samples 4000 \
    --output outputs/ray_attention_temporal_residual_iterative.pth
```

## Evaluation command

```bash
conda run -n mf python experiments/eval_ray_attention_temporal_residual_iterative.py \
    --checkpoint outputs/ray_attention_temporal_residual_iterative.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --batch_size 8 \
    --max_n_iter 3 \
    --out docs/swarm_iter7/iterative_residual_unrolled_eval.json
```

## Expected metrics

| setting | MPJPE (mm) | PA-MPJPE (mm) | PCK@50mm | AUC |
|---|---:|---:|---:|---:|
| Baseline (current best) | 11.17 | 8.24 | 1.0000 | 0.9256 |
| Target after unrolled iterative refinement | **< 10.5** | **< 8.0** | **≥ 0.9995** | **≥ 0.9300** |

Smoke success criterion: `n_iter=3` does not diverge and stays within ~10 % of `n_iter=1`.

## Risks and mitigations

1. **No improvement over single-step residual.** The single-step head is already strong; iterative refinement may saturate. *Mitigation:* keep the change small and directly compare with the same d/h/config.
2. **Training instability from unrolled loop.** Back-propagating through repeated corrections can explode. *Mitigation:* conservative gate initialization and gradient clipping (already supported in v3 trainer).
3. **Overfitting.** Adding the gate and multi-step loss gives more capacity. *Mitigation:* match the base model size; only the gate adds a few hundred parameters.
4. **Hyperparameter sensitivity.** `iter_loss_weight` and `n_iter` matter. *Mitigation:* smoke-test `iter_loss_weight ∈ {0.25, 0.5, 1.0}` before committing to full runs.
