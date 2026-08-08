#!/usr/bin/env python3
"""Poll A800-D GPU memory and launch v30a when enough memory frees.

Usage:
    python scripts/launch_v30_when_gpu_free.py
"""

from __future__ import annotations

import subprocess
import time

A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
SSH_HOST = "a800-D"
MIN_FREE_MIB = 65000  # Need a mostly-free GPU for v30a (d=128).
POLL_INTERVAL = 60  # seconds

# Build a launch script on A800 for the selected GPU.
LAUNCH_SCRIPT = (
    "cd {repo} && "
    "cat > /tmp/launch_v30a_gpu{gpu}.sh <<'EOF'\n"
    "#!/bin/bash\n"
    "cd {repo}\n"
    "CUDA_VISIBLE_DEVICES={gpu} python3 -u experiments/train_omniview_fusion_v5_webbridge_multi.py "
    "--use_mixed_loader --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml "
    "--use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding "
    "--use_deformable_cross_view_attention_v18 --use_multiview_geometry_fusion_v25 --v25_dropout 0.2 "
    "--v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
    "--num_workers 4 --d 128 --residual_hidden 256 --n_st_layers 3 --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 "
    "--epochs 20 --batch_size 8 --train_samples 4000 --val_stride 10 --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 "
    "--max_grad_norm 1.0 --ema_decay 0.999 --early_stopping_patience 5 --early_stopping_min_delta 0.001 "
    "--use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true --use_context_visibility true "
    "--use_skeleton_residual true --use_rotation_correction true --use_entropy_regularization true --attention_entropy_weight 0.01 "
    "--use_camera_view_embedding --use_set_view_aggregator --use_variable_view_training --variable_view_min_views 2 "
    "--variable_view_max_views 14 --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute "
    "--pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 --reproj_loss_weight 0.1 --reproj_warmup_epochs 3 "
    "--aleatoric_reproj_loss_weight 0.1 --outlier_view_prob 0.3 --outlier_view_max_views 1 "
    "--outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 "
    "--use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1 "
    "--use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001 "
    "--v29_physical_loss_warmup_epochs 3 --output outputs/omniview_fusion_v30a_hierarchical_physical_a800.pth "
    "> outputs/omniview_fusion_v30a_hierarchical_physical_a800.log 2>&1\n"
    "EOF\n"
    "chmod +x /tmp/launch_v30a_gpu{gpu}.sh\n"
    "tmux has-session -t v30a_gpu{gpu} 2>/dev/null || "
    "tmux new-session -d -s v30a_gpu{gpu} -n v30 'bash /tmp/launch_v30a_gpu{gpu}.sh'"
)


def a800_ssh(cmd: str) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", SSH_HOST, cmd],
        text=True,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    while True:
        out = a800_ssh(
            "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        )
        free_gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                free_gpus.append((int(parts[0]), int(parts[1])))

        candidates = [(g, f) for g, f in free_gpus if f >= MIN_FREE_MIB]
        if candidates:
            gpu, free = candidates[0]
            print(f"GPU {gpu} has {free} MiB free; launching v30a.")
            a800_ssh(LAUNCH_SCRIPT.format(repo=A800_REPO, gpu=gpu))
            print("v30a launched.")
            break
        else:
            print(
                f"No GPU with >= {MIN_FREE_MIB} MiB free; "
                f"sleeping {POLL_INTERVAL}s"
            )
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
