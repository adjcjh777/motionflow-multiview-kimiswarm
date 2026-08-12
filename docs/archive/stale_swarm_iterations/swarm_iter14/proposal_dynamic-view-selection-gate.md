# Dynamic View-Selection Gate: Learn to Drop/Weight Noisy Views Per Joint

## 1. Problem

The current 9.32 mm anchor fuses every view with learned attention weights, so a single noisy or occluded view can still bias the DLT triangulation because there is no explicit, per-joint mechanism that learns to ignore it.

## 2. Hypothesis

A lightweight, differentiable per-view/per-joint selection gate, trained with a pose loss plus a sparsity-entropy regularizer, can learn to down-weight or drop noisy views per joint without degrading clean accuracy and while improving robustness under view dropout and joint-level occlusion.

## 3. Method

### 3.1 New module — `DynamicViewSelectionGate`

Create `motionflow_mv/fusion/dynamic_view_selection_gate.py` with a module that:

- Consumes the post-attention tokens `feat: (B*T, V, J, d)` from the cross-view spatio-temporal transformer.
- Optionally concatenates two ray-geometry features (baseline length and ray angle) already used by `AdaptiveViewSelector`.
- Predicts a per-view/per-joint gate `g_vj = sigmoid(logit)` in `[0, 1]`.
- Returns `gate_weights = g_vj`, `gate_mask = (g_vj > 0.5)`, and `gate_logits` for loss computation.

The gate is implemented as a 2-layer per-joint MLP shared across views:

```python
self.gate_mlp = nn.Sequential(
    nn.Linear(d + 2, d // 2),
    nn.ReLU(),
    nn.Linear(d // 2, 1),
)
```

### 3.2 New model — `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointDynamicGate`

Create `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_principal_point_dynamic_gate_model.py`:

- Subclass `RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`.
- Insert `DynamicViewSelectionGate` after the ST attention and before the DLT weight computation.
- Multiply the predicted triangulation weights by the gate:

```python
weights = weights * confidences * gate_weights
```

- Expose `return_gate=True` to output `gate_weights` alongside `pred_3d` and `weights`.

### 3.3 New loss — `view_selection_loss`

Create `motionflow_mv/losses/view_selection_loss.py`:

```python
class ViewSelectionLoss(nn.Module):
    def __init__(self, sparsity_weight: float = 0.01, entropy_weight: float = 0.001):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.entropy_weight = entropy_weight

    def forward(self, gate_weights):
        sparsity = gate_weights.mean()
        entropy = -(gate_weights * torch.log(gate_weights + 1e-6) +
                    (1 - gate_weights) * torch.log(1 - gate_weights + 1e-6)).mean()
        return self.sparsity_weight * sparsity, self.entropy_weight * entropy
```

Combined training loss:

```python
loss = mse(pred_3d, gt_3d) + sparsity_loss + entropy_loss
```

### 3.4 Training script

Create `experiments/train_crossview_residual_dynamic_view_gate_mpiinf3dhp.py` by extending `experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`:

- Add arguments:
  - `--gate_sparsity_weight` (default `0.01`)
  - `--gate_entropy_weight` (default `0.001`)
  - `--view_noise_std` (default `2.0`, pixels)
  - `--joint_dropout_rate` (default `0.15`)
- Augment each training clip with:
  - Per-view 2D Gaussian noise (std `view_noise_std`).
  - Per-joint random dropout of `confidences` at rate `joint_dropout_rate`.
- Instantiate the dynamic-gate model variant and add `ViewSelectionLoss` to the optimizer.

### 3.5 Data changes

No dataset change; use existing `data/webbridge/mpi_inf_3dhp/*.npz` files. Synthetic noise/dropout is applied online in the trainer.

## 4. Smoke-Test Plan

Run a 5-epoch smoke with a small subset:

```bash
python experiments/train_crossview_residual_dynamic_view_gate_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 9 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --batch_size 8 --train_samples 500 --epochs 5 \
  --gate_sparsity_weight 0.01 --gate_entropy_weight 0.001 \
  --view_noise_std 2.0 --joint_dropout_rate 0.15 \
  --output outputs/dynamic_view_gate_smoke.pth
```

**Pass criteria:**
- No NaNs or crashes.
- Validation MPJPE after 5 epochs ≤ 11 mm.
- Mean gate value across all joints/views ≤ 0.95 (i.e., at least 5% average down-weighting).
- Training loss monotonically decreases for the last 3 epochs.

**Fail criteria:**
- Validation MPJPE > 15 mm.
- Gate mean > 0.98 (gate is not learning to drop anything).
- Any NaN or crash.

## 5. Evaluation Plan

1. **Clean accuracy on MPI-INF-3DHP S2/Seq1**
   - Run `python experiments/eval_full_metrics.py --model dynamic_gate_pp --checkpoint outputs/dynamic_view_gate_smoke.pth`.
   - Report MPJPE, PA-MPJPE, PCK@50/100/150, AUC.
   - Target: within 0.5 mm of the 9.32 mm anchor (smoke) and within 0.3 mm after a full 20-epoch run.

2. **Robustness matrix**
   - Run `python experiments/eval_robustness_matrix_pp_mpiinf3dhp.py --model dynamic_gate_pp`.
   - Compare view_dropout_0.3 and joint_dropout_0.2 vs. the PP baseline.
   - Pass: ≥ 10% relative improvement under dropout with clean MPJPE ≤ 9.6 mm.

3. **Gate interpretability**
   - Dump per-corruption mean gates to `outputs/dynamic_view_gate_stats.json`.
   - Verify that the gate average is lower for corrupted/occluded views than clean views.

## 6. Estimated GPU/CPU Cost on RTX 4090

- **Smoke (5 epochs, 500 samples):** ~20–30 minutes on RTX 4090; the gate adds ~3k parameters, so throughput is essentially identical to the PP baseline.
- **Full run (20 epochs, 4k samples):** ~2.5–3.5 hours on RTX 4090.
- **Evaluation:** CPU-only robustness matrix in < 10 minutes; clean eval in < 5 minutes.

## 7. Risks & Fallback

| Risk | Mitigation / Fallback |
|------|-----------------------|
| Gate collapses and keeps all weights ≈ 1 | Increase `sparsity_weight` by 5× or switch to a learned *dropout probability* with a fixed budget (top-k hard selection). |
| Gate drops useful views and hurts clean accuracy | Start from the 9.32 mm anchor as a warm start and freeze the backbone for 2 epochs; lower `entropy_weight`. |
| Hard/straight-through gate causes NaNs | Use the soft gate only for triangulation; keep the loss continuous. |
| Online synthetic noise slows the CPU pipeline | Cache augmented clips in `tmp/dynamic_gate_aug/` or reduce workers to 0 and pre-generate a small corrupted split. |
| No improvement on real occlusion | Fall back to the visibility-v2 path (`visibility_gated_fusion.py`) or combine gate with explicit reprojection-error supervision. |
