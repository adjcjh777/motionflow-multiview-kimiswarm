#!/usr/bin/env python3
"""Queue and launch a 20-run v29 hyper-parameter/variant sweep on A800-D.

The script polls GPU memory on A800-D via SSH and launches the next run on the
first GPU with enough free memory.  It keeps up to ``max_concurrent`` runs active
at once and sleeps between polls.

Usage (from local 4090 / any host with SSH to a800-D):
    python scripts/launch_v29_20run_sweep_a800.py
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
SSH_HOST = "a800-D"
MIN_FREE_MIB = 12000  # Require ~12 GiB free before launching another run.
MAX_CONCURRENT = 6
POLL_INTERVAL = 60  # seconds


COMMON_FLAGS = (
    "--use_mixed_loader "
    "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml "
    "--use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding "
    "--use_deformable_cross_view_attention_v18 "
    "--use_multiview_geometry_fusion_v25 --v25_dropout 0.2 "
    "--v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
    "--num_workers 4 --d 64 --residual_hidden 128 --n_st_layers 2 "
    "--graph_num_layers 1 --n_joint_layers 1 --n_heads 4 "
    "--epochs 20 --batch_size 8 --train_samples 1000 --val_stride 10 "
    "--lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 "
    "--max_grad_norm 1.0 --ema_decay 0.999 "
    "--early_stopping_patience 3 --early_stopping_min_delta 0.001 "
    "--use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true "
    "--use_context_visibility true --use_skeleton_residual true --use_rotation_correction true "
    "--use_entropy_regularization true --attention_entropy_weight 0.01 "
    "--use_camera_view_embedding --use_set_view_aggregator "
    "--use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 "
    "--variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute "
    "--pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 "
    "--reproj_loss_weight 0.1 --reproj_warmup_epochs 3 --aleatoric_reproj_loss_weight 0.1 "
    "--outlier_view_prob 0.3 --outlier_view_max_views 1 "
    "--outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0"
)


@dataclass
class SweepRun:
    name: str
    extra_flags: str
    output: str


SWEEP_RUNS: List[SweepRun] = [
    SweepRun("v29e_hierarchical_physical", "--use_hierarchical_multiview_v29 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29e_hierarchical_physical_a800"),
    SweepRun("v29f_n_part_layers_2", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --v29_n_part_layers 2 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29f_n_part_layers_2_a800"),
    SweepRun("v29g_tte_iters_5", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --v27_tte_n_iters 5 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29g_tte_iters_5_a800"),
    SweepRun("v29h_floor_weight_0_1", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.1 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29h_floor_weight_0_1_a800"),
    SweepRun("v29i_bone_weight_0_1", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.1 --v29_com_jitter_weight 0.001", "omniview_fusion_v29i_bone_weight_0_1_a800"),
    SweepRun("v29j_com_weight_0_01", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.01", "omniview_fusion_v29j_com_weight_0_01_a800"),
    SweepRun("v29k_full_d64", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --d 64 --residual_hidden 128 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29k_full_d64_a800"),
    SweepRun("v29l_full_batch_16", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --batch_size 16 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29l_full_batch_16_a800"),
    SweepRun("v29m_full_n_st_layers_1", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --n_st_layers 1 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29m_full_n_st_layers_1_a800"),
    SweepRun("v29n_hierarchical_d128", "--use_hierarchical_multiview_v29 --d 128 --residual_hidden 256 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29n_hierarchical_d128_a800"),
    SweepRun("v29o_hierarchical_n_st_3", "--use_hierarchical_multiview_v29 --n_st_layers 3 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29o_hierarchical_n_st_3_a800"),
    SweepRun("v29p_tte_thresh_0_1", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --v29_tte_residual_thresh_mm 0.1 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29p_tte_thresh_0_1_a800"),
    SweepRun("v29q_tte_sigma_3", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --v29_tte_sigma_reproj 3.0 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29q_tte_sigma_3_a800"),
    SweepRun("v29r_tte_physical_floor_low", "--use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.001 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29r_tte_physical_floor_low_a800"),
    SweepRun("v29s_tte_physical_bone_low", "--use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.001 --v29_com_jitter_weight 0.001", "omniview_fusion_v29s_tte_physical_bone_low_a800"),
    SweepRun("v29t_no_variable_views", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --variable_view_max_views 14 --variable_view_max_views_start 14 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29t_no_variable_views_a800"),
    SweepRun("v29u_hierarchical_n_heads_2", "--use_hierarchical_multiview_v29 --n_heads 2 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29u_hierarchical_n_heads_2_a800"),
    SweepRun("v29v_tte_no_physical_align", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --no_v29_tte_use_physical_space_alignment --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001", "omniview_fusion_v29v_tte_no_physical_align_a800"),
    SweepRun("v29w_floor_only", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.0 --v29_com_jitter_weight 0.0", "omniview_fusion_v29w_floor_only_a800"),
    SweepRun("v29x_bone_only", "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.0 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.0", "omniview_fusion_v29x_bone_only_a800"),
]


def a800_ssh(cmd: str) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", SSH_HOST, cmd],
        text=True,
        stderr=subprocess.STDOUT,
    )


def gpu_free_mibs() -> List[Tuple[int, int]]:
    """Return list of (gpu_index, free_mib) for all GPUs on A800-D."""
    out = a800_ssh("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits")
    pairs: List[Tuple[int, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def active_sessions(prefix: str) -> int:
    try:
        out = a800_ssh("tmux ls 2>/dev/null")
        return sum(1 for line in out.splitlines() if line.strip().startswith(prefix))
    except subprocess.CalledProcessError:
        return 0


def launch_run(run: SweepRun, gpu: int, idx: int) -> None:
    session = f"v29_sweep_{idx:02d}_gpu{gpu}"
    cmd = (
        f"cd {A800_REPO} && "
        f"CUDA_VISIBLE_DEVICES={gpu} python3 -u experiments/train_omniview_fusion_v5_webbridge_multi.py "
        f"{COMMON_FLAGS} {run.extra_flags} "
        f"--output outputs/{run.output}.pth "
        f"> outputs/{run.output}.log 2>&1"
    )
    a800_ssh(f"tmux new-session -d -s {session} -n v29 '{cmd}'")
    print(f"Launched {session} (GPU {gpu}) for {run.name}")


def main() -> None:
    queue = list(SWEEP_RUNS)
    launched: int = 0
    while queue:
        active = active_sessions("v29_sweep_")
        if active >= MAX_CONCURRENT:
            print(f"Active sweep sessions: {active}; sleeping {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            continue

        pairs = gpu_free_mibs()
        # Sort GPUs by free memory descending.
        pairs.sort(key=lambda x: x[1], reverse=True)
        selected_gpu: Optional[int] = None
        for gpu, free in pairs:
            if free >= MIN_FREE_MIB:
                selected_gpu = gpu
                break

        if selected_gpu is None:
            print(f"No GPU with >= {MIN_FREE_MIB} MiB free; sleeping {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            continue

        run = queue.pop(0)
        launch_run(run, selected_gpu, launched)
        launched += 1

    print(f"All {launched} sweep runs launched.")


if __name__ == "__main__":
    main()
