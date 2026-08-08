#!/usr/bin/env bash
# Launch v23 small on A800 GPU4 and GPU6 via tmux.
set -euo pipefail

cd /mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm-iter20
VENV=/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python

run() {
    gpu=$1
    name=$2
    output=$3
    log="${output%.pth}.log"
    tmux kill-session -t "$name" 2>/dev/null || true
    tmux new-session -d -s "$name" \
        "CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=/mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm-iter20 $VENV experiments/train_omniview_fusion_v5_webbridge_multi.py --use_mixed_loader --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding --use_deformable_cross_view_attention_v18 --use_kinematic_anthropometric_prior_v22 --kap_loss_weight 0.01 --num_workers 4 --d 128 --residual_hidden 256 --n_st_layers 3 --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 --epochs 20 --batch_size 16 --train_samples 2000 --val_stride 10 --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999 --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true --use_entropy_regularization true --attention_entropy_weight 0.01 --use_camera_view_embedding --use_set_view_aggregator --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 --reproj_loss_weight 0.1 --reproj_warmup_epochs 3 --aleatoric_reproj_loss_weight 0.1 --outlier_view_prob 0.3 --outlier_view_max_views 1 --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 --output $output > $log 2>&1"
}

run 4 v23_kap_no_ba_gpu4 outputs/omniview_fusion_v23_kap_no_ba_gpu4.pth
run 6 v23_kap_no_ba_gpu6 outputs/omniview_fusion_v23_kap_no_ba_gpu6.pth

sleep 2
tmux list-sessions
