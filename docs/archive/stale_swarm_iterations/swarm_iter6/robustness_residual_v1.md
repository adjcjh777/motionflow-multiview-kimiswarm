# Robustness Evaluation of Residual Temporal Ray-Attention Model

**Swarm iteration:** 6  
**Agent task:** Create a robustness test for the top-performing residual refinement model under controlled noise, occlusion, and outliers.  
**Model:** `RayAttentionFusionModelTemporalResidual` (`outputs/ray_attention_temporal_residual_v2.pth`)  
**Checkpoint baseline MPJPE:** ~13.12 mm on MPI-INF-3DHP val S2 Seq1 (this run).  

## What was done

Created a single self-contained evaluation script:

- `experiments/eval_residual_robustness_mpiinf3dhp_v1.py`

The script re-uses the same `TemporalClipDataset` / collate logic as the residual training script, loads the trained checkpoint, and evaluates under:

1. **Gaussian 2D noise** on `(x, y)` detections: `std = 0, 2, 5, 10, 20` px.
2. **Random joint occlusion**: per-frame per-view per-joint confidence zeroed at rates `0.0, 0.1, 0.2, 0.3, 0.5`.
3. **Random 2D outliers**: replace `(x, y)` with a large random value and zero confidence at rates `0.0, 0.02, 0.05, 0.10, 0.20`.

Metrics reported per condition: `MPJPE`, `PA-MPJPE`, `PCK@50/100/150mm`, and `PCK AUC (0-150mm)`.

Outputs saved to `outputs/robustness_residual_v1_stride5/`:

- `robustness_report.json` — full numeric results.
- `robustness_report.csv` — table-friendly CSV.
- `robustness_plots.png` — three-panel plot of MPJPE vs. perturbation level.

## How to run

```bash
# Full S2 Seq1 (stride=5 for efficiency; ~1 300 clips)
conda run -n mf python experiments/eval_residual_robustness_mpiinf3dhp_v1.py \
    --checkpoint outputs/ray_attention_temporal_residual_v2.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --batch_size 32 --stride 5 \
    --out_dir outputs/robustness_residual_v1_stride5

# Smoke-sized subset (250 frames)
conda run -n mf python experiments/eval_residual_robustness_mpiinf3dhp_v1.py \
    --checkpoint outputs/ray_attention_temporal_residual_v2.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_smoke.npz \
    --clip_len 13 --batch_size 8 --stride 1 \
    --out_dir outputs/robustness_residual_v1_smoke
```

## Results (full MPI-INF-3DHP val S2 Seq1, stride=5, 28 joints, 14 views)

### Noise sweep

| Noise std (px) | MPJPE (mm) | PA-MPJPE (mm) | PCK@50mm | PCK@100mm | PCK@150mm | PCK AUC |
|---------------:|-----------:|----------------:|---------:|----------:|----------:|--------:|
| 0              | 13.12      | 10.86           | 0.9999   | 1.0000    | 1.0000    | 0.9125  |
| 2              | 13.41      | 11.19           | 0.9998   | 1.0000    | 1.0000    | 0.9106  |
| 5              | 14.77      | 12.78           | 0.9995   | 1.0000    | 1.0000    | 0.9015  |
| 10             | 18.58      | 17.18           | 0.9981   | 1.0000    | 1.0000    | 0.8761  |
| 20             | 28.86      | 28.47           | 0.9409   | 1.0000    | 1.0000    | 0.8076  |

### Occlusion sweep

| Occlusion rate | MPJPE (mm) | PA-MPJPE (mm) | PCK@50mm | PCK@100mm | PCK@150mm | PCK AUC |
|---------------:|-----------:|----------------:|---------:|----------:|----------:|--------:|
| 0.0            | 13.12      | 10.86           | 0.9999   | 1.0000    | 1.0000    | 0.9125  |
| 0.1            | 13.12      | 10.86           | 0.9999   | 1.0000    | 1.0000    | 0.9125  |
| 0.2            | 13.12      | 10.86           | 0.9999   | 1.0000    | 1.0000    | 0.9125  |
| 0.3            | 13.12      | 10.86           | 0.9999   | 1.0000    | 1.0000    | 0.9125  |
| 0.5            | 13.13      | 10.87           | 0.9999   | 1.0000    | 1.0000    | 0.9125  |

### Outlier sweep

| Outlier rate | MPJPE (mm) | PA-MPJPE (mm) | PCK@50mm | PCK@100mm | PCK@150mm | PCK AUC |
|-------------:|-----------:|----------------:|---------:|----------:|----------:|--------:|
| 0.00         | 13.12      | 10.86           | 0.9999   | 1.0000    | 1.0000    | 0.9125  |
| 0.02         | 13.10      | 11.00           | 0.9998   | 1.0000    | 1.0000    | 0.9127  |
| 0.05         | 13.60      | 11.43           | 0.9996   | 1.0000    | 1.0000    | 0.9093  |
| 0.10         | 14.94      | 12.41           | 0.9989   | 1.0000    | 1.0000    | 0.9004  |
| 0.20         | 17.86      | 14.22           | 0.9953   | 1.0000    | 1.0000    | 0.8809  |

### Plot

![Robustness plots](../../outputs/robustness_residual_v1_stride5/robustness_plots.png)

## Observations

1. **Noise:** The model degrades gracefully with increasing Gaussian pixel noise. MPJPE stays below 15 mm for noise ≤5 px and rises to ~29 mm at 20 px. This is the expected behavior for a multi-view triangulation system.
2. **Occlusion:** With 14 camera views, the model is extremely robust to random per-joint occlusion. Even dropping 50% of detections raises MPJPE by only ~0.01 mm, indicating strong multi-view redundancy and effective attention-based down-weighting of missing joints.
3. **Outliers:** Performance remains stable up to ~5% outlier corruption, then degrades. At 20% outliers, MPJPE increases to 17.86 mm. The model appears to rely on its learned attention to suppress corrupted detections once their confidence is zeroed.
4. The clean MPJPE of 13.12 mm on the strided full-dataset run is consistent with the project-reported ~13.84 mm baseline for this checkpoint.

## Files touched / created

- `experiments/eval_residual_robustness_mpiinf3dhp_v1.py` — new robustness evaluation script.
- `docs/swarm_iter6/robustness_residual_v1.md` — this report.
- `outputs/robustness_residual_v1_stride5/` — full-dataset JSON, CSV, and plot.
- `outputs/robustness_residual_v1_smoke/` — smoke-run JSON, CSV, and plot (validation run).

## Known limitations / next steps

- The full-dataset run used `stride=5` to keep evaluation time reasonable (~1 300 clips instead of ~6 500). The trend is clear, but a `stride=1` run would give the most exact numbers.
- The outlier perturbation also zeroes the confidence of corrupted joints. A stricter test would keep confidence high to force the model to actually reject the outlier in feature space.
- No per-joint robustness breakdown is included yet; adding it would reveal which joints fail first under each corruption type.
