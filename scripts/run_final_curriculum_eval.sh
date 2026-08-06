#!/usr/bin/env bash
# Final validation of the curriculum checkpoint once training finishes.
set -e
cd "$(dirname "$0")/.."

CKPT="outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth"
DATA="data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz"

echo "[final eval] waiting for curriculum checkpoint: $CKPT"
while [ ! -f "$CKPT" ]; do
    sleep 30
done

LAST_WRITE=$(stat -c %Y "$CKPT" 2>/dev/null || stat -f %m "$CKPT")
# Wait until no write for 60 seconds (training likely finished saving).
while true; do
    sleep 60
    CUR_WRITE=$(stat -c %Y "$CKPT" 2>/dev/null || stat -f %m "$CKPT")
    if [ "$CUR_WRITE" -eq "$LAST_WRITE" ]; then
        break
    fi
    LAST_WRITE=$CUR_WRITE
done

echo "[final eval] running clean + robustness matrix on full MPI S2"
KMP_DUPLICATE_LIB_OK=TRUE /tmp/mf_venv/bin/python experiments/eval_curriculum_robustness.py \
    --checkpoint "$CKPT" \
    --dataset "$DATA" \
    --out_json outputs/eval_curriculum_robustness_final.json \
    > outputs/eval_curriculum_robustness_final.log 2>&1

echo "[final eval] running variable-view MPJPE@k"
KMP_DUPLICATE_LIB_OK=TRUE /tmp/mf_venv/bin/python experiments/eval_variable_views.py \
    --dataset "$DATA" \
    --checkpoint "$CKPT" \
    --model_class crossview_residual_pp \
    --d 64 --residual_hidden 128 --n_temporal_layers 2 --clip_len 13 \
    --min_views 2 --max_views 14 --num_subsets_per_k 10 --seed 42 \
    --output_json outputs/variable_views_curriculum_final.json \
    --output_csv outputs/variable_views_curriculum_final.csv \
    > outputs/variable_views_curriculum_final.log 2>&1

echo "[final eval] done"
