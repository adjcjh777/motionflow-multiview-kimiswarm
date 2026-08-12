#!/usr/bin/env bash
# Launch v24 (v18 + fixed neural BA + KAP) on A800-D via tmux.
#
# Usage:
#   bash scripts/launch_v24_a800_tmux.sh [GPU ...]
#
# If no GPU is supplied, the script automatically picks the least-utilised
# free GPU.  If one or more GPUs are supplied, a tmux session is started on
# each.  The session is intended to be launched as soon as an A800 GPU frees
# up.
#
# Environment overrides:
#   MF_GPU          GPU index to use (default: auto-select a free GPU)
#   MF_ALLOWED_GPUS comma-separated GPUs to consider (default: 0,1,2,3,4,5,6,7)
#   MF_BUSY_GPUS    comma-separated GPUs to skip  (default: empty)
set -euo pipefail

cd /mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm-iter20
VENV="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u"

ALLOWED_GPUS="${MF_ALLOWED_GPUS:-0,1,2,3,4,5,6,7}"
BUSY_GPUS="${MF_BUSY_GPUS:-}"
MEMORY_THRESHOLD_MIB="${MF_MEMORY_THRESHOLD_MIB:-2000}"

find_free_gpu() {
    local gpu_list best_gpu best_util util mem_used
    gpu_list="${ALLOWED_GPUS//,/ }"
    best_gpu=""
    best_util=101

    for gpu in ${gpu_list}; do
        if [[ ",${BUSY_GPUS}," == *",${gpu},"* ]]; then
            continue
        fi

        if ! command -v nvidia-smi >/dev/null 2>&1; then
            best_gpu="$gpu"
            break
        fi

        util=$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "100")
        if [[ -z "$util" || "$util" == "[NotSupported]" || "$util" == "[InsufficientPermissions]" ]]; then
            util=100
        fi

        mem_used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "999999")
        if [[ -n "$mem_used" && "$mem_used" =~ ^[0-9]+$ && "$mem_used" -gt "$MEMORY_THRESHOLD_MIB" ]]; then
            continue
        fi

        if (( util < best_util )); then
            best_util="$util"
            best_gpu="$gpu"
        fi
    done

    echo "$best_gpu"
}

launch_one() {
    local gpu=$1
    local name="v24_kap_fixed_ba_gpu${gpu}"
    local output="outputs/omniview_fusion_v24_kap_fixed_ba_gpu${gpu}.pth"
    local log="outputs/omniview_fusion_v24_kap_fixed_ba_gpu${gpu}.log"

    tmux kill-session -t "$name" 2>/dev/null || true
    tmux new-session -d -s "$name" \
        "CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=/mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm-iter20 $VENV experiments/train_omniview_fusion_v5_webbridge_multi.py --use_mixed_loader --mixed_manifest configs/deprecated/circular/splits/webbridge_h36m_mpi_mixed_train_val.yaml --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding --use_deformable_cross_view_attention_v18 --use_neural_bundle_adjustment_v21 --use_kinematic_anthropometric_prior_v22 --kap_loss_weight 0.01 --num_workers 0 --d 128 --residual_hidden 256 --n_st_layers 3 --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 --epochs 20 --batch_size 16 --train_samples 2000 --val_stride 10 --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999 --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true --use_entropy_regularization true --attention_entropy_weight 0.01 --use_camera_view_embedding --use_set_view_aggregator --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 --reproj_loss_weight 0.1 --reproj_warmup_epochs 3 --aleatoric_reproj_loss_weight 0.1 --outlier_view_prob 0.3 --outlier_view_max_views 1 --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 --output $output > $log 2>&1"

    echo "Launched v24 on GPU $gpu as tmux session $name"
}

if [[ -n "${MF_GPU:-}" ]]; then
    gpus=($MF_GPU)
elif [[ $# -gt 0 ]]; then
    gpus=("$@")
else
    free_gpu=$(find_free_gpu)
    if [[ -z "$free_gpu" ]]; then
        echo "ERROR: No free GPU available. Allowed=${ALLOWED_GPUS}, busy=${BUSY_GPUS}." >&2
        exit 1
    fi
    gpus=("$free_gpu")
fi

for gpu in "${gpus[@]}"; do
    launch_one "$gpu"
done

tmux list-sessions
