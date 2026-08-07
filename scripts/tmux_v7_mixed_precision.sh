#!/usr/bin/env bash
# v7 mixed-dataset + full precision DLT training launcher.
# Run this on A800-D when a GPU is free (default GPU 7).
set -euo pipefail
ROOT=/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20
cd "$ROOT"
export PYTHONPATH="$ROOT"

tmux new-session -d -s v7_mixed_precision -n main "CUDA_VISIBLE_DEVICES=7 .venv/bin/python -u experiments/train_omniview_fusion_v5_webbridge_multi.py --use_mixed_loader --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml --use_full_precision_dlt --d 128 --residual_hidden 256 --n_st_layers 3 --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 --epochs 60 --batch_size 16 --train_samples 10000 --val_stride 10 --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999 --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true --use_entropy_regularization true --attention_entropy_weight 0.01 --use_camera_view_embedding --use_set_view_aggregator --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute --monotonic_loss_weight 0.1 --monotonic_margin 5.0 --output outputs/omniview_fusion_v7_mixed_precision.pth > outputs/omniview_fusion_v7_mixed_precision.log 2>&1; read -p 'Press enter to close...'"
