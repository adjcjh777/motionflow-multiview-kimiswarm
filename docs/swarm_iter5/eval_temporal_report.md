# Temporal Model Evaluation Script (MPI-INF-3DHP)

## What was done

Created a minimal, reusable evaluation script that loads a trained
`RayAttentionFusionModelTemporal` checkpoint and reports **MPJPE**, **PA-MPJPE**, and
**PCK** on a canonical MPI-INF-3DHP `.npz` validation sequence.

- **New script:** `experiments/eval_ray_attention_temporal_v1.py`
- **Output JSON (example):** `outputs/eval_temporal_smoke.json`

The script reuses the existing `TemporalClipDataset` and `collate_fn` from
`experiments/train_ray_attention_temporal_mpiinf3dhp.py` and the metric helpers in
`motionflow_mv/eval/metrics.py`, so no existing code was duplicated.

### Supported metrics

- MPJPE (mm)
- PA-MPJPE (mm, rigid Procrustes alignment)
- PCK@50mm / @100mm / @150mm
- PCK AUC (0–150 mm)

Per-joint variants are also computed inside `compute_all_metrics`; the JSON summary
currently exposes only the scalar values for brevity.

## How to run

```bash
conda run -n mf python experiments/eval_ray_attention_temporal_v1.py \
    --checkpoint outputs/ray_attention_temporal_baseline.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --batch_size 8 \
    --out outputs/eval_temporal_baseline.json
```

Optional flags:

- `--d 64` (must match the checkpoint, default 64)
- `--n_temporal_layers 2` (must match the checkpoint, default 2)
- `--device cpu` to force CPU evaluation

## Blocker: requested baseline checkpoint is missing

The user asked to evaluate `outputs/ray_attention_temporal_baseline.pth`, but that
file does **not** exist in the repository:

```text
ls: cannot access '.../outputs/ray_attention_temporal_baseline.pth': No such file or directory
```

The only temporal checkpoint present is `outputs/ray_attention_temporal_smoke.pth`.
Therefore the smoke checkpoint was used to validate the script instead.

## Results (smoke checkpoint)

Command run:

```bash
conda run -n mf python experiments/eval_ray_attention_temporal_v1.py \
    --checkpoint outputs/ray_attention_temporal_smoke.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --batch_size 8 \
    --out outputs/eval_temporal_smoke.json
```

Cross-subject / cross-sequence validation on **S2 Seq1**:

| Metric        | Value   |
|---------------|--------|
| MPJPE         | 25.2532 mm  |
| PA-MPJPE      | 24.1625 mm  |
| PCK@50mm      | 0.9893      |
| PCK@100mm     | 1.0000      |
| PCK@150mm     | 1.0000      |
| PCK AUC (0-150mm) | 0.8317 |

- Clips evaluated: **6,490**
- Clip length: **13 frames**
- Joints: **28**
- Views: **14**

These numbers are consistent with the reported smoke-run MPJPE of ~25.25 mm.

## Next steps / follow-up

1. Generate or copy the requested `outputs/ray_attention_temporal_baseline.pth`.
2. Re-run the script above pointing `--checkpoint` to that file to obtain the
   official cross-subject/cross-sequence numbers for the baseline.

No new dependencies were introduced, so no `requirements` note is needed.
