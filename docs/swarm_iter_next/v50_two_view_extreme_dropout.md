# v50: Two-View Extreme Dropout (ExtremeSparseViewV50)

## 1. Architecture

ExtremeSparseViewV50 pushes the v46 sparse-view generalization objective to its practical limit: **reliable 3-D pose from only two views**. The module sits between the v25/v45 geometry-fusion backbone and the v46 sparse-view head. During training it periodically samples all (or a random subset of) 2-view subsets, forces the model to produce a 3-D pose from each pair, and compares that pair-wise prediction to a stable full-view teacher pose. A lightweight **pairwise view-synergy scorer** (a small MLP over view-pair geometry features and v37 reliability) estimates how much each 2-view pair should be trusted; this score is then used to weight the consistency loss. At inference the module is disabled by default: it only refines the learned reliability/geometry fusion weights, so the full-view pipeline remains unchanged and the model incurs no extra latency.

Key components:
- **2-view subset sampler**: deterministic round-robin over view pairs during training, with `min_views=2` enforced.
- **Pairwise synergy scorer**: inputs are relative camera pose, ray intersection angle, and v37 per-view reliability; output is a scalar weight in `(0,1)` for each view pair.
- **Teacher-aware consistency head**: the 2-view prediction is aligned to the full-view teacher with Procrustes-free `L2`, then weighted by the synergy score.

The scorer is identity-at-init (starts near uniform) so the module does not perturb an already warm v46 checkpoint.

## 2. New Config Flags

```yaml
use_v50_two_view_extreme_dropout: false        # master switch
v50_tv_dropout_prob: 0.5                       # probability of a 2-view training step
v50_tv_pair_loss_weight: 0.01                  # weight of the 2-view consistency loss
v50_tv_min_views: 2                            # hard floor (kept equal to 2)
v50_tv_use_pairwise_reliability: true          # use the synergy scorer
v50_tv_use_epipolar_teacher: true              # teacher = full-view triangulated pose
v50_tv_max_pairs_per_step: 4                   # avoid quadratic blow-up when V is large
v50_tv_warmup_epochs: 1                        # freeze scorer for first epoch
```

## 3. Loss Term

```text
L_v50 = v50_tv_pair_loss_weight * mean(
    synergy(pair) * ||P_2view(pair) - P_teacher||_2
)
```

`P_2view(pair)` is the 3-D pose predicted from the selected two views only. `P_teacher` is the full-view output (or ground-truth when available). When `v50_tv_use_epipolar_teacher=false`, ground truth is used, which is preferred for WebBridge/H36M but unavailable for 3DPW actual-mode validation. The synergy score is supervised indirectly: good pairs (large baseline, non-degenerate rays, high v37 reliability) must produce lower `||P_2view - P_teacher||`.

## 4. Evaluation Metric

Primary: `MPJPE@2` (mean per-joint position error when exactly two views are used). Secondary: `MPJPE@full` to confirm no regression. Also report mean synergy score per dataset as a sanity check.

## 5. Expected MPJPE Impact

Based on the v46-SVG smoke `val_MPJPE@full ≈ 33 mm`, the hardest remaining gap is the 2-view case. ExtremeSparseViewV50 is expected to improve `MPJPE@2` by **3–5 mm** while keeping `MPJPE@full` within **±1 mm** of the baseline. On 3DPW actual-mode (if available), the target is a finite `MPJPE@2` below 80 mm.

## 6. Main Risk / Mitigations

| Risk | Mitigation |
|------|------------|
| Full-view teacher is noisy and biases the 2-view branch | Use ground-truth teacher wherever labels exist; otherwise require `v37` reliability > 0.5 for the teacher sample. |
| Aggressive 2-view mining destabilizes training | Cap `v50_tv_dropout_prob` at 0.5; freeze scorer for `v50_tv_warmup_epochs`; gradually raise pair count. |
| Combinatorial explosion when V is large | Limit `v50_tv_max_pairs_per_step`; sample pairs rather than enumerate. |
| Overfitting to the two easiest camera pairs | Round-robin pair sampling and domain-conditional pair selection. |

## 7. Dependencies

Depends on: v45-AGF, v46-SVG, v37 self-critique reliability. Should be evaluated only after v49-Lite and the adaptive v49 view-dropout are stable, since it extends the same sparse-view story.
