# Swarm Iter 6 — Residual Temporal Model Evaluation on MPI-INF-3DHP S2/Seq1

## Task
Comprehensively evaluate the current top-performing residual refinement model
(`RayAttentionFusionModelTemporalResidual`) on the cross-subject MPI-INF-3DHP
validation sequence (subject 2, sequence 1).  Extend the metric set beyond
MPJPE to include PA-MPJPE, PCK@50/100/150 mm, and PCK-AUC@150 mm, and produce
per-joint breakdowns.

## Files touched / created

| Path | Purpose |
|------|---------|
| `motionflow_mv/eval/metrics.py` | Read-only; already contains `compute_all_metrics` and PCK/AUC helpers used by the new eval. |
| `experiments/eval_ray_attention_temporal_v1.py` | Read-only; served as the base for the residual eval. |
| `experiments/eval_ray_attention_temporal_residual_v3.py` | **New** eval script for the residual model. Loads `RayAttentionFusionModelTemporalResidual`, runs the full S2/Seq1 sequence, and reports all metrics. |
| `outputs/eval_residual_v3_s2_seq1.json` | JSON summary produced by the eval run. |
| `docs/swarm_iter6/eval_residual_v3_s2_seq1_report.md` | This report. |

## How to reproduce

```bash
conda run -n mf python experiments/eval_ray_attention_temporal_residual_v3.py \
    --checkpoint outputs/ray_attention_temporal_residual_v2.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --batch_size 8 \
    --out outputs/eval_residual_v3_s2_seq1.json
```

Environment: `mf` conda env on local RTX 4090.

## Results on MPI-INF-3DHP S2/Seq1

- **Checkpoint:** `outputs/ray_attention_temporal_residual_v2.pth`
- **Validation data:** `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`
- **Model:** `RayAttentionFusionModelTemporalResidual`  
  243,428 parameters, `d=64`, `n_temporal_layers=2`, `residual_hidden=128`
- **Clips evaluated:** 6,490 non-overlapping clips of length 13 (28 joints)

### Metric summary

| Metric | Value |
|----------|-------|
| MPJPE | **13.8452 mm** |
| PA-MPJPE | **11.0678 mm** |
| PCK@50 mm | **0.999998** |
| PCK@100 mm | **1.000000** |
| PCK@150 mm | **1.000000** |
| PCK-AUC (0–150 mm) | **0.9077** |

### Per-joint MPJPE (mm)

Sorted by joint index (0–27):

```
10.91, 11.30, 10.69, 10.26, 11.03, 11.36, 11.62, 12.50,
11.31, 12.52, 14.15, 15.19, 15.96, 10.98, 11.85, 14.63,
15.83, 16.40, 11.28, 14.06, 15.95, 17.11, 17.53, 11.69,
16.42, 17.30, 18.70, 19.13
```

### Per-joint PA-MPJPE (mm)

Sorted by joint index (0–27):

```
 7.55,  8.48,  6.88,  7.75,  6.43,  8.66,  9.52, 11.75,
 8.19,  9.48, 10.61, 11.54, 12.70,  8.15,  9.28, 11.00,
12.49, 13.35,  7.22, 10.16, 15.12, 16.72, 17.05,  7.49,
12.22, 15.71, 17.05, 17.35
```

## Analysis

1. **MPJPE matches the reported ~13.84 mm** from the current checkpoint, confirming
   the eval pipeline is consistent with the training validation loop.
2. **PA-MPJPE of 11.07 mm** shows that even after rigid Procrustes alignment the
   model retains very low error, indicating the refinement is not merely rescaling
   the skeleton but actually improving joint-level structure.
3. **PCK is essentially saturated** at all three thresholds; on this sequence
   virtually every joint is within 50 mm of ground truth on average.  This is
   consistent with the low MPJPE but highlights that PCK alone is no longer a
   discriminative metric here.
4. **PCK-AUC (0–150 mm) = 0.9077** is the most informative “fine-grained” summary;
   it captures the full error distribution rather than a single threshold.
5. **Per-joint errors** show the largest residual errors are in the lower-body
   extremities (joints 21–23 and 25–27), which is typical for multiview pose
   datasets where feet/toes have the highest depth ambiguity.

## Blockers

None. The existing checkpoint and data files were present, the eval script ran
without errors, and all requested metrics were produced.

## Next-step suggestions

- Compare the residual model against the non-residual temporal baseline on the
  same sequence using the same metrics to quantify the residual head’s gain.
- Run the same evaluation on H36M or another test set to check generalization.
- Add failure-case visualisation for the highest-error joints (feet/toes) to
  guide further refinement.
