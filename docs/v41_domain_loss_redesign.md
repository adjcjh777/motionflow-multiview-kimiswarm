# v41 Domain-Aware Loss Redesign

**Slug:** `v41_domain_loss_redesign`  
**Scope:** MotionFlow-MultiView v41  
**Target downstream:** ICRA/CVPR 2027 multi-view pose pipeline  

## 1. Current v41 implementation (baseline)

The existing v41 weighted-domain loss is a single scalar weight per domain applied to the 3-D MSE term in `experiments/train_omniview_fusion_v5_webbridge_multi.py` (lines 1096–1102):

```python
mse_per_sample = F.mse_loss(pred_3d, y, reduction="none").mean(dim=(1, 2, 3))
if dataset_id is not None and domain_loss_weights is not None:
    domain_sample_weights = domain_loss_weights.to(device)[dataset_id.squeeze(-1).long()]
    loss = (mse_per_sample * domain_sample_weights).mean()
else:
    loss = mse_per_sample.mean()
```

The weights are passed as a static comma-separated string, e.g. `--domain_loss_weights 1.0,1.5`, and only two values are supported (H36M = domain 0, MPI-INF-3DHP = domain 1).

### Limitations

1. **Global, time-invariant weights.** The same weight is used for every sample in a domain for the entire training run. It cannot react to per-sample difficulty, domain drift, or training stage.
2. **Only scales the 3-D MSE.** Reprojection, bone-length, physical, and auxiliary losses are left unweighted, so hard MPI samples still dominate those terms.
3. **No joint-level weighting.** H36M and MPI have different skeletons conventions, joint label noise, and occlusions. A scalar cannot up-weight the joints that are harder in one domain.
4. **No learning / adaptation.** The weights are hand-tuned hyperparameters; the optimizer does not adjust them based on validation performance.
5. **Fragile to domain count.** Adding a third domain (e.g., an expanded WebBridge subset) requires manually guessing a new scalar.

## 2. Proposed redesign: adaptive per-domain, per-joint, per-term loss

Replace the static scalar with a small family of domain-aware loss modifiers that are (1) learnable or schedule-driven, (2) applied to multiple loss terms, and (3) operate at the joint level when possible.

### 2.1 Core idea

For each training batch we want to compute a **domain difficulty signal** and use it to re-weight the loss:

* **Domain-level weight** `w_d` — how much the optimizer should focus on each domain right now.
* **Joint-level weight** `w_j` — which joints are currently hardest for each domain.
* **Loss-term weight** `w_term` — which loss components should be domain-scaled (MSE, reprojection, physical, bone).

The final loss for sample `i`, domain `d` becomes:

```
L_i = sum_term [ w_term * w_d(t, d) * w_j(d, j) * L_term(i, j) ]
```

with `w_d` normalized over the batch so that a single domain does not dominate gradient magnitudes.

### 2.2 Design options (pick one as the v41 default)

| Design | What it does | Pros | Cons |
|---|---|---|---|
| **A. Online domain loss averaging (DWA)** | Maintain running average of per-domain total loss; set `w_d ~ L_target / L_d` | Simple, no extra hyperparameters beyond temperature | Needs burn-in; can be noisy early |
| **B. GradNorm-style adaptive weights** | Keep per-domain loss gradients at a fixed relative scale; update weights via gradient descent | Theoretically cleaner, automatically balances domains | Adds an extra backward pass, more compute |
| **C. Domain-conditioned uncertainty (DUQ)** | Predict per-domain aleatoric uncertainty and down-weight high-uncertainty samples | Robust to label noise, principled | Needs a small uncertainty head, harder to tune |
| **D. Learned scalar via meta-gradient** | Treat domain weights as meta-parameters updated on a small validation batch | Strongest adaptation | Complex, easy to overfit to validation |

**Recommendation:** start with **A (DWA)** because it is a minimal change and already addresses the main failure mode (static weights). Keep B/C as follow-up ablations.

### 2.3 Recommended v41 redesign: Domain-Difficulty-Weighted Loss (DDWL)

#### 2.3.1 Compute per-domain running loss

Maintain an small exponential moving average (EMA) of the unweighted per-domain MSE:

```python
# updated once per epoch or once per N steps
for d in domains:
    loss_ema[d] = beta * loss_ema[d] + (1 - beta) * mean_loss_domain[d]
```

Use `beta = 0.9` and reset at epoch boundaries.

#### 2.3.2 Compute adaptive domain weights

```python
# relative inverse difficulty; temperature T controls smoothing
w_d_raw = (loss_ema / loss_ema.max()) ** (-1 / T)
w_d = w_d_raw / w_d_raw.sum() * num_domains
```

Default `T = 2.0`. Higher temperature makes weights more uniform; lower temperature focuses on the hardest domain.

#### 2.3.3 Apply to multiple loss terms

Instead of weighting only the MSE, pass the same per-sample weight into the other loss terms that already return per-sample values:

* 3-D MSE: weight full per-sample MSE.
* Reprojection loss: already per-sample via `_reprojection_loss`; scale by `w_d`.
* Bone / physical losses: these return a scalar per batch; if a per-joint version exists, weight by domain-joint importance.

For losses that are scalar only, use the domain weight at batch level rather than per-sample.

#### 2.3.4 Joint-level domain importance map

Add a small buffer `domain_joint_importance: Tensor[num_domains, n_joints]` initialized to ones. After each epoch, update:

```python
# per-domain, per-joint mean absolute error collected during training
joint_error = ema_per_domain_joint_error(d, j)
importance[d, j] = (joint_error[d, j] / joint_error[d].mean()).clamp(0.5, 2.0)
```

This map is used as `w_j(d, j)` in the MSE term. It lets the model focus on joints that are currently hardest for a given domain (e.g., MPI feet, H36M wrists).

#### 2.3.5 Schedule / warmup

Domain weights should be frozen at uniform for the first `domain_loss_warmup_epochs` (default 1) so the EMA has time to stabilize. Gradual ramp is not necessary for DDWL because the EMA itself provides smoothness.

### 2.4 CLI / configuration changes

In `experiments/train_omniview_fusion_v5_webbridge_multi.py`, add or modify flags:

```python
parser.add_argument("--domain_loss_mode", type=str, default="static", choices=["static", "ddwl"])
parser.add_argument("--domain_loss_weights", type=str, default=None)  # keep for backward compat
parser.add_argument("--domain_loss_warmup_epochs", type=int, default=1)
parser.add_argument("--domain_loss_temperature", type=float, default=2.0)
parser.add_argument("--domain_loss_ema_beta", type=float, default=0.9)
parser.add_argument("--domain_loss_apply_to_reproj", action="store_true")
parser.add_argument("--domain_loss_apply_to_physical", action="store_true")
parser.add_argument("--domain_loss_joint_importance", action="store_true")
```

When `domain_loss_mode == "ddwl"`, the static `--domain_loss_weights` is ignored.

## 3. Pseudocode for training step

```python
# In build_compute_loss closure, maintain:
#   loss_ema = {domain_id: 0.0}
#   joint_error_ema = torch.ones(num_domains, n_joints)

# Forward pass unchanged
mse_per_sample = F.mse_loss(pred_3d, y, reduction="none").mean(dim=(1, 2, 3))

if args.domain_loss_mode == "ddwl" and dataset_id is not None:
    # 1. Update EMA after burn-in
    with torch.no_grad():
        for d in range(num_domains):
            mask = (dataset_id.squeeze(-1) == d)
            if mask.any():
                loss_ema[d] = (beta * loss_ema[d]
                               + (1 - beta) * mse_per_sample[mask].mean().item())

    # 2. Compute adaptive weights
    if current_epoch >= args.domain_loss_warmup_epochs:
        ema_t = torch.tensor([loss_ema[d] for d in domain_ids], device=device)
        w_d_raw = (ema_t / ema_t.max() + 1e-6) ** (-1 / T)
        w_d = w_d_raw / w_d_raw.sum() * num_domains
    else:
        w_d = torch.ones(num_domains, device=device)

    # 3. Per-sample weights
    sample_weights = w_d[dataset_id.squeeze(-1).long()]
    loss = (mse_per_sample * sample_weights).mean()

    # 4. Joint-level reweighting
    if args.domain_loss_joint_importance:
        joint_w = joint_error_ema[dataset_id.squeeze(-1).long()]  # (B, J)
        per_joint_mse = F.mse_loss(pred_3d, y, reduction="none").mean(dim=(1, 3))  # (B, J)
        loss = (per_joint_mse * joint_w).mean()
else:
    loss = mse_per_sample.mean()
```

Reprojection and physical losses can be multiplied by `sample_weights.detach()` when their per-sample forms exist.

## 4. Expected impact

| Metric | Expected change |
|---|---|
| `val_MPJPE` | Small improvement on the harder domain (MPI-INF-3DHP) because its weight automatically rises when error is high. |
| Per-domain gap | Reduced H36MMPI gap because the optimizer spends more compute on the domain that is currently worse. |
| Joint-level errors | Lower error on domain-specific hard joints (e.g., MPI feet under self-occlusion). |
| Hyperparameter robustness | Less dependence on a hand-tuned `--domain_loss_weights`. |

## 5. Ablations

| Run | Configuration | Question |
|---|---|---|
| `v41_static_baseline` | Current `--domain_loss_weights 1.0,1.5` | Is the baseline stable? |
| `v41_ddwl_uniform` | DDWL with `T → inf` (uniform weights) | Does the DDWL machinery add overhead without benefit? |
| `v41_ddwl_mse_only` | DDWL applied only to MSE | Is reprojection/physical scaling needed? |
| `v41_ddwl_mse_reproj` | DDWL on MSE + reprojection | Does reprojection benefit? |
| `v41_ddwl_full` | DDWL on MSE + reprojection + physical + joint importance | Best case. |
| `v41_ddwl_gradnorm` | GradNorm variant | Does adaptive gradient balancing beat DDWL? |

## 6. Risks and mitigations

1. **Weight collapse.** If MPI is always harder, its weight may saturate and H36M can underfit. Mitigation: clamp `w_d` to `[0.5, 2.0]` and use a temperature `T >= 2.0`.
2. **EMA lag at the start of training.** Early loss values are noisy. Mitigation: use a 1-epoch warmup with uniform weights.
3. **Joint-importance overfit.** The per-joint map can focus on a single noisy joint. Mitigation: smooth the map with the same EMA and clamp values.
4. **Interaction with domain-balanced sampling.** If `DomainBalancedSampler` is also used, the two mechanisms can fight. Mitigation: when domain-balanced sampling is active, start DDWL with a higher temperature so the loss reweighting only fine-tunes.

## 7. Files to touch (if implemented)

* `experiments/train_omniview_fusion_v5_webbridge_multi.py` — add DDWL state, flags, and loss reweighting.
* `motionflow_mv/training/domain_loss_state.py` (new, optional) — small helper for EMA and weight computation.
* `docs/v41_domain_loss_redesign.md` — this proposal.

## 8. Go/no-go criteria

* Smoke test (2 epochs, 20 samples) completes and `val_MPJPE` is within 5% of the static v41 baseline.
* Full local RTX 4090 run shows a smaller H36M↔MPI `val_MPJPE` gap than the static baseline.
* A800 full run confirms the trend before this design is promoted to the default in v44/v45.
