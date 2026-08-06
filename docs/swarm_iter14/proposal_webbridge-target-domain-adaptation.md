# Cross-Dataset Domain Adaptation: Adapt WebBridge Features to Target Dataset

## 1. Problem

The 9.32 mm anchor model is trained and validated on MPI-INF-3DHP only, while WebBridge already provides canonical `.npz` versions of H36M, AIST++ and MPI-INF-3DHP with incompatible skeletons (17 vs 28 joints), leaving the project without a minimal, reproducible pipeline for transferring learned pose features across datasets.

## 2. Hypothesis

A shared 17-joint skeleton plus the existing GRL+FiLM `DomainAdaptationWrapper`, warm-started from the PP anchor and trained on mixed WebBridge source data, will produce lower target-dataset MPJPE than source-only fine-tuning because the wrapper can learn domain-invariant ray-attention features while FiLM preserves dataset-specific camera/statistical biases.

## 3. Method

### 3.1 Data: canonical 17-joint source/target mix

Create a new module `motionflow_mv/data/joint_mapping.py` that stores:
- `H36M_17_JOINTS`: names and parent indices of the H36M 17-joint skeleton.
- `MPIINF_TO_H36M_17`: a 28→17 index map from MPI-INF-3DHP joints to the H36M subset.
- `AIST_TO_H36M_17`: identity map (AIST++ already uses the same 17-joint layout).

Create `motionflow_mv/data/canonical_17_dataset.py` as a thin wrapper around the canonical `.npz` loader:
- On load, re-index `points_2d`, `confidences`, and `joints_3d` to 17 joints using the mapping above.
- Pad views to `MAX_VIEWS=14` as in `mixed_dataset.py`.
- Return `(x, y, K, R, t, domain_label)` where `domain_label=0` for WebBridge source and `domain_label=1` for the held-out target sequence.

Create `experiments/train_webbridge_target_domain_adapt.py` as the trainer. It will:
1. Load one or more WebBridge source `.npz` files (e.g., MPI train + H36M train) as domain 0.
2. Load the target dataset `.npz` (e.g., H36M subject 11 or MPI test subject) as domain 1.
3. Warm-start the pose backbone from the best anchor checkpoint (`outputs/ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp/best.pth` or equivalent).
4. Train `motionflow_mv.models.domain_adaptation_wrapper.DomainAdaptationWrapper` with the 17-joint configuration.

### 3.2 Model / loss changes

Reuse `motionflow_mv/models/domain_adaptation_wrapper.py` unchanged. Its current signature already accepts `domain_labels` and returns `domain_logits`, so it is sufficient for this experiment.

Loss:

```python
loss = pose_loss + lambda_domain * domain_loss + lambda_mmd * mmd_loss
```

- `pose_loss`: MSE on the 17-joint 3D pose, computed only on labeled source/target clips.
- `domain_loss`: cross-entropy from the GRL domain discriminator.
- `mmd_loss`: optional RBF-MMD computed on pooled features from the two domains in a batch (using `maximum_mean_discrepancy` in `motionflow_mv/fusion/domain_adaptation_wrapper.py`); initially disabled (`lambda_mmd=0`) to keep the smoke minimal.

### 3.3 Exact files to create or modify

| File | Action | Purpose |
|------|--------|---------|
| `motionflow_mv/data/joint_mapping.py` | Create | H36M 17-joint names + MPI/AIST-to-H36M index maps |
| `motionflow_mv/data/canonical_17_dataset.py` | Create | Load any canonical `.npz`, map to 17 joints, attach domain label |
| `experiments/train_webbridge_target_domain_adapt.py` | Create | Mixed source/target trainer with warm-start and GRL+FiLM |
| `motionflow_mv/models/domain_adaptation_wrapper.py` | Read-only | Wrap PP backbone; no changes needed |
| `motionflow_mv/fusion/domain_adaptation_wrapper.py` | Read-only | Optional MMD helper already exists |
| `configs/benchmark_webbridge_crossview_residual_smoke.yaml` | Modify (optional) | Add target-domain eval split if not already present |

### 3.4 Command-line example

```bash
KMP_DUPLICATE_LIB_OK=TRUE python experiments/train_webbridge_target_domain_adapt.py \
    --source data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
            data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz \
    --target data/webbridge/h36m_meters/s_11_acts_02_multiview_m.npz \
    --val data/webbridge/h36m_meters/s_11_acts_03_multiview_m.npz \
    --anchor_ckpt outputs/ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp/best.pth \
    --clip_len 13 --d 128 --n_st_layers 3 --residual_hidden 256 \
    --epochs 5 --batch_size 4 --train_samples 500 \
    --lambda_domain 0.1 --lambda_mmd 0.0
```

## 4. Smoke-Test Plan

Run a 3-epoch smoke on a tiny model with 100 random clips per domain.

```bash
python experiments/train_webbridge_target_domain_adapt.py \
    --source data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m_smoke.npz \
            data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz \
    --target data/webbridge/h36m_meters/s_11_acts_02_multiview_m.npz \
    --val data/webbridge/h36m_meters/s_11_acts_03_multiview_m.npz \
    --clip_len 9 --d 32 --n_st_layers 1 --residual_hidden 64 \
    --epochs 3 --batch_size 2 --train_samples 100 \
    --lambda_domain 0.1
```

Pass/fail criteria:
- Pass: script completes without NaNs or shape mismatches.
- Pass: `pose_loss` and `domain_loss` both decrease over the 3 epochs.
- Pass: validation MPJPE on the held-out target clip is finite and below a naive source-only baseline (e.g., < 50 mm for the smoke; exact threshold is flexible because the smoke model is tiny).
- Fail: any runtime error, shape mismatch, or validation MPJPE diverging to > 500 mm.

## 5. Evaluation Plan

After the smoke, run the full 20-epoch adaptation and evaluate with:

| Metric | Script / command |
|--------|----------------|
| Target-dataset MPJPE / PA-MPJPE | `experiments/eval_full_metrics.py --model domain_adapt --target data/webbridge/h36m_meters/s_11_acts_02_multiview_m.npz` (or create `experiments/eval_webbridge_target_domain_adapt.py` if the existing script does not support the wrapper) |
| Domain discriminator accuracy | Logged during training; should stay near 0.5 (chance) after convergence |
| Source-only baseline | Run the same trainer with `--lambda_domain 0 --no_film` for ablation |
| Ablation: GRL only / FiLM only / both | `--no_film`, `--no_domain_classifier`, and default flags |

Target success criterion for the full run:
- Target H36M MPJPE < source-only baseline by at least 5%.
- No clean accuracy degradation on MPI-INF-3DHP source validation (optional: keep a small source validation set).

## 6. Estimated GPU/CPU Cost on RTX 4090

| Phase | Hardware | Time | Notes |
|-------|----------|------|-------|
| Smoke (3 epochs, tiny model, 200 clips) | RTX 4090 / CPU | ~2–3 min | Runs comfortably on CPU if GPU is busy |
| Full adaptation (20 epochs, d=128, 3 ST layers) | RTX 4090 | ~4–6 hours | Comparable to the existing PP full run; bottleneck is the `(time × view)` attention |
| Evaluation | CPU | < 10 min | Single forward pass over the target validation sequence |

## 7. Risks & Fallback

| Risk | Mitigation / Fallback |
|------|----------------------|
| MPI→H36M joint mapping is ambiguous or lossy | Start with AIST++ → H36M (already 17 joints) as a same-skeleton sanity transfer, then add MPI once the mapping is validated. |
| `DomainAdaptationWrapper` has stale shape assumptions for the PP backbone | Smoke the wrapper first; if it breaks, fix the forward signature in `motionflow_mv/models/domain_adaptation_wrapper.py` before training. |
| GRL domain loss destabilizes the pose backbone | Reduce `lambda_domain` to 0.01 or disable the discriminator (`--no_domain_classifier`), keeping only FiLM. |
| Target dataset has no labels for real deployment | Add `--unlabeled_target` mode where pose loss is computed only on source clips and domain loss on both domains; use the existing semi-supervised logic in `experiments/train_domain_adapt_mpiinf3dhp.py`. |
| RTX 4090 is occupied | The smoke is CPU-only friendly; the full run can be queued behind the visibility-v2/factorized jobs. |

## 8. Definition of Done

- [ ] `motionflow_mv/data/joint_mapping.py` and `motionflow_mv/data/canonical_17_dataset.py` created and unit-tested.
- [ ] `experiments/train_webbridge_target_domain_adapt.py` runs the 3-epoch smoke without errors.
- [ ] Smoke validation MPJPE is finite and below the source-only baseline on the same tiny model.
- [ ] Full target-dataset adaptation is queued or run, with MPJPE/PA-MPJPE logged in `docs/results_iter14.md`.