# Direction 15: Self-Supervised / Masked Pre-Training Protocol

## Problem Statement

The project already has a masked-view reprojection pre-training stage
(`experiments/pretrain_ray_attention_ssl.py`, `motionflow_mv/data/ssl_dataset.py`)
and GPU launchers (`scripts/run_ssl_pretrain_h36m*.sh`). What is still missing is
a **quantified data-efficiency protocol**: we do not yet know how much labeled
MPI-INF-3DHP data is needed after SSL pre-training to match (or beat) the
fully-supervised baseline. Direction 15 therefore needs the smallest possible
next experiment that measures the SSL → fine-tune gain across label fractions.

## Simplest Concrete Next Experiment

Build deterministic 10 % / 25 % / 50 % / 100 % labeled-clip splits of the
MPI-INF-3DHP train set and generate ready-to-run fine-tuning launchers that
warm-start the best PP model from an SSL checkpoint. Do **not** launch GPU
jobs while the RTX 4090 is busy with the cross-view PP curriculum. The work
is CPU-only today: create the split manifest and scripts, run a smoke test on
synthetic data to validate the split logic, and queue the GPU launchers for
later execution.

## Files to Touch

| File | Change |
|------|--------|
| `experiments/prepare_ssl_data_efficiency_mpiinf3dhp.py` | New CPU-only script that builds label-fraction splits and writes launcher shell scripts. |
| `docs/swarm_iter7/self_supervised_masked_pretraining_protocol.md` | This report. |

### Rough diff / sketch

```python
# experiments/prepare_ssl_data_efficiency_mpiinf3dhp.py
# Builds non-overlapping clip indices, samples fractions deterministically,
# writes manifest.json and per-fraction launcher scripts.
all_indices = _build_clip_indices(args.train, args.clip_len, args.seed)
for frac in [0.10, 0.25, 0.50, 1.00]:
    n = max(1, int(round(len(all_indices) * frac)))
    split = all_indices[:n]
    write_launcher(f"run_ssl_finetune_{int(frac*100):02d}pct.sh",
                   warm_start=args.ssl_checkpoint)
```

```bash
# Generated launcher (GPU, queued only)
python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
    --train <10pct-clips> --val <val.npz> \
    --warm_start outputs/ray_attention_ssl_h36m.pth \
    --epochs 30 --batch_size 8 --pp_loss_weight 0.1 \
    --output outputs/ray_attention_ssl_finetune_10pct.pth
```

## Expected Success Metric

After GPU fine-tuning on each fraction, plot **MPI-INF-3DHP val MPJPE vs. labeled
fraction**. Success is either:

* SSL-pre-trained model matches the fully-supervised baseline with ≤ 50 % of the
  labels, or
* At 100 % labels, SSL warm-start improves over the same model trained from
  scratch by ≥ 0.3 mm MPJPE.

Secondary: lower masked-view reprojection loss during SSL pre-training should
correlate with better downstream MPJPE.

## Resource Requirement

* Today: **CPU-only** — split generation and smoke test.
* Later: **GPU** for four short fine-tuning runs (one per fraction). GPU jobs
  are queued, not started.

## Smoke Test Results

Run the CPU-only script with synthetic data:

```bash
python experiments/prepare_ssl_data_efficiency_mpiinf3dhp.py --smoke
```

Expected output: a manifest with total clips and per-fraction clip counts, plus
launcher scripts in `outputs/ssl_data_efficiency_smoke/`.

