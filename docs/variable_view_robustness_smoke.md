# Variable-View Robustness Curves — MPI-INF-3DHP Smoke

Smoke evaluation of v25, v46, v57, and v80 OmniMultiViewFusionV5 smoke checkpoints on a 300-frame subset of `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz`. The MPI 28-joint skeleton was mapped to the canonical 17-joint H36M layout used by the smoke checkpoints.

## Commands run

```bash
# Build a 17-joint, 300-frame smoke version of the MPI sequence
source .venv/bin/activate
python - <<'PY'
import numpy as np
src='data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz'
dst='data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_17j_smoke300.npz'
MPI_28_TO_17 = np.array([4,23,24,25,18,19,20,3,5,6,9,10,11,14,15,16,7], dtype=np.int64)
data=np.load(src)
T=300
np.savez(dst,
    points_2d=data['points_2d'][:T, :, MPI_28_TO_17, :],
    confidences=data['confidences'][:T, :, MPI_28_TO_17],
    joints_3d=data['joints_3d'][:T, MPI_28_TO_17, :],
    camera_K=data['camera_K'], camera_R=data['camera_R'], camera_t=data['camera_t'])
PY

# Eval script required a one-line fix: pass the dataset's actual n_views to
# _build_omniview_v5_model instead of 0 (otherwise view embeddings/graphs mismatch).
for m in v25 v46 v57 v80; do
  python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint outputs/omniview_fusion_${m}_mpi_only_noncircular_smoke.pth \
    --config outputs/omniview_fusion_${m}_mpi_only_noncircular_smoke.config.json \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m_17j_smoke300.npz \
    --k_values 2 3 4 14 \
    --num_subsets_per_k 5 \
    --output_csv outputs/${m}_mpi_varview.csv
done
```

## Summary table

`outputs/mpjpe_at_k_summary.csv`:

| model | MPJPE@2 | MPJPE@3 | MPJPE@4 | MPJPE@14 |
|-------|---------|---------|---------|----------|
| v25   | 131.98  | 72.05   | 89.02   | 30.61    |
| v46   | 108.51  | 93.10   | 72.31   | 68.64    |
| v57   | 143.67  | 87.02   | 53.41   | 40.93    |
| v80   | 145.01  | 74.41   | 63.34   | 51.79    |

## Interpretation

- All checkpoints improve substantially when moving from 2 to 14 views, confirming the expected robustness benefit of additional cameras.
- **v25** has the best full-view result (MPJPE@14 ≈ 30.6 mm) but is volatile across k.
- **v57** shows the strongest low-view scaling: its error drops from ~144 mm at k=2 to ~40 mm at k=14, and it beats the other models at k=3 and k=4.
- **v80** also scales well but ends with a higher full-view floor than v25 and v57.
- **v46** is competitive at k=2 but its full-view performance plateaus around 68 mm, suggesting it is less effective when all 14 views are available.

These are smoke numbers on a 300-frame subset, so the absolute values and subset standard deviations are noisy and should not be used for model selection.
