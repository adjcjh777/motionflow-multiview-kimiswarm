#!/usr/bin/env bash
# Run full GPU variable-view eval, then visibility v2 training, after curriculum finishes.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE

echo "[queue] waiting for curriculum training to finish..."
while pgrep -f "train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py" > /dev/null; do
    sleep 60
done

CKPT="outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth"
DATA="data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz"

if [ -f "$CKPT" ]; then
    echo "[queue] running full GPU variable-view MPJPE@k on $CKPT"
    python experiments/eval_variable_views.py \
        --dataset "$DATA" \
        --checkpoint "$CKPT" \
        --model_class crossview_residual_pp \
        --d 64 --residual_hidden 128 --n_temporal_layers 2 --clip_len 13 \
        --min_views 2 --max_views 14 --num_subsets_per_k 5 --seed 42 \
        --output_json outputs/variable_views_curriculum_final.json \
        --output_csv outputs/variable_views_curriculum_final.csv \
        > outputs/variable_views_curriculum_final.log 2>&1
fi

echo "[queue] starting visibility v2 training"
bash scripts/run_crossview_residual_visibility_v2_wsl.sh
