#!/usr/bin/env python3
"""Poll A800-D and launch the v32 A800 queue when a GPU frees.

This runs after the v31 top-5 variants.  It launches the pending v31
physical_floor_only run plus v32 ablations (domain-aware view curriculum,
trajectory-consistency refiner, and a combined run).
"""

from __future__ import annotations

import re
import subprocess
import time


A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
SSH_HOST = "a800-D"
MIN_FREE_MIB = 30000  # d=64 full run on A800
POLL_INTERVAL = 60  # seconds

COMMON_FLAGS = (
    "--use_mixed_loader "
    "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml "
    "--use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding "
    "--use_deformable_cross_view_attention_v18 "
    "--use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 "
    "--v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
    "--num_workers 4 --d 64 --residual_hidden 128 --n_st_layers 2 "
    "--graph_num_layers 1 --n_joint_layers 1 --n_heads 4 "
    "--clip_len 9 --epochs 20 --batch_size 8 --train_samples 1000 --val_stride 10 "
    "--lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999 "
    "--early_stopping_patience 5 --early_stopping_min_delta 0.001 "
    "--use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true "
    "--use_context_visibility true --use_skeleton_residual true --use_rotation_correction true "
    "--use_entropy_regularization true --attention_entropy_weight 0.01 "
    "--use_camera_view_embedding --use_set_view_aggregator "
    "--use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 "
    "--variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute "
    "--pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 "
    "--reproj_loss_weight 0.1 --reproj_warmup_epochs 3 --aleatoric_reproj_loss_weight 0.1 "
    "--outlier_view_prob 0.3 --outlier_view_max_views 1 "
    "--outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 "
    "--use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1 "
    "--use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001 "
    "--v29_physical_loss_warmup_epochs 3"
)

RUNS = [
    # Pending v31 ablation.
    (
        "v31_physical_floor_only",
        "--use_hierarchical_multiview_v30 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.0 --v29_com_jitter_weight 0.0 --v29_physical_loss_warmup_epochs 3",
        "omniview_fusion_v31_physical_floor_only_a800",
    ),
    # v32 ablations.
    (
        "v32_domain_aware_view_curriculum",
        "--domain_aware_view_curriculum",
        "omniview_fusion_v32_domain_aware_view_curriculum_a800",
    ),
    (
        "v32_trajectory_consistency_refiner",
        "--use_trajectory_consistency_v32 --v32_smooth_weight 1e-3 --v32_drift_weight 1e-2",
        "omniview_fusion_v32_trajectory_consistency_a800",
    ),
    (
        "v32_combined",
        "--domain_aware_view_curriculum --use_trajectory_consistency_v32 --v32_smooth_weight 1e-3 --v32_drift_weight 1e-2",
        "omniview_fusion_v32_combined_a800",
    ),
    (
        "v32_ray_attention",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias --v31_use_ray_attention",
        "omniview_fusion_v32_ray_attention_a800",
    ),
    (
        "v32_physical_alignment",
        "--use_physical_space_alignment_v32 --v28_floor_loss_weight 0.01 --v28_bone_temporal_weight 0.01",
        "omniview_fusion_v32_physical_alignment_a800",
    ),
]


def a800_ssh(cmd: str) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", SSH_HOST, cmd],
        text=True,
        stderr=subprocess.STDOUT,
    )


def gpu_free_mibs() -> list[tuple[int, int]]:
    out = a800_ssh("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits")
    pairs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def used_gpus_from_tmux() -> set[int]:
    """Return GPU indices already used by v31 or v32 tmux sessions."""
    gpus: set[int] = set()
    try:
        out = a800_ssh("tmux ls 2>/dev/null || true")
    except subprocess.CalledProcessError:
        return gpus
    for line in out.splitlines():
        # v31_top5_<name>_gpuN or v32_<name>_gpuN
        match = re.search(r"(?:v31_top5|v32)_(.+)_gpu(\d+):", line)
        if match:
            gpus.add(int(match.group(2)))
    return gpus


def running_run_names() -> set[str]:
    """Return the set of run keys already running on A800."""
    names: set[str] = set()
    try:
        out = a800_ssh("tmux ls 2>/dev/null || true")
    except subprocess.CalledProcessError:
        return names
    for line in out.splitlines():
        match = re.search(r"(?:v31_top5|v32)_(.+)_gpu\d+:", line)
        if match:
            names.add(match.group(1))
    return names


def launch_run(name: str, extra_flags: str, output: str, gpu: int) -> None:
    session = f"v32_{name}_gpu{gpu}"
    cmd = (
        f"cd {A800_REPO} && "
        f"CUDA_VISIBLE_DEVICES={gpu} python3 -u experiments/train_omniview_fusion_v5_webbridge_multi.py "
        f"{COMMON_FLAGS} {extra_flags} "
        f"--output outputs/{output}.pth "
        f"> outputs/{output}.log 2>&1"
    )
    a800_ssh(f"tmux has-session -t {session} 2>/dev/null || tmux new-session -d -s {session} -n v32 '{cmd}'")
    print(f"Launched {session} (GPU {gpu}) for {name}")


def main() -> None:
    queue = list(RUNS)
    print("Pulling latest main on A800-D...")
    try:
        a800_ssh(f"cd {A800_REPO} && timeout 15 git pull origin main || true")
    except subprocess.CalledProcessError as e:
        print(f"Warning: git pull on A800 failed: {e}")
    used_gpus = used_gpus_from_tmux()
    already_running = running_run_names()
    queue = [(n, f, o) for n, f, o in queue if n not in already_running]
    print(f"Already-used GPUs from tmux: {used_gpus}")
    print(f"Already-running runs names: {already_running}")
    print(f"Remaining queue: {[n for n, _, _ in queue]}")
    while queue:
        pairs = gpu_free_mibs()
        candidates = [
            (g, f) for g, f in pairs
            if f >= MIN_FREE_MIB and g not in used_gpus
        ]
        if not candidates:
            print(f"No GPU with >= {MIN_FREE_MIB} MiB free; sleeping {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            continue
        gpu, _ = max(candidates, key=lambda x: x[1])
        used_gpus.add(gpu)
        name, extra_flags, output = queue.pop(0)
        launch_run(name, extra_flags, output, gpu)
        time.sleep(60)
    print("All v32 runs launched.")


if __name__ == "__main__":
    main()
