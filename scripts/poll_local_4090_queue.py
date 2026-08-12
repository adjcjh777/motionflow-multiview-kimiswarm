#!/usr/bin/env python3
"""Poll local RTX 4090 and launch a queue of experiments when GPU memory frees.

Similar to scripts/launch_v33_a800_queue.py but for the local WSL RTX 4090.
Runs are launched as subprocesses (not tmux) because tmux is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil


LOG_DIR = Path("outputs")
MIN_FREE_MIB = 10000  # Require at least 10 GiB free before launching a run
POLL_INTERVAL = 300  # seconds


def _nvidia_smi_path() -> str:
    path = shutil.which("nvidia-smi")
    if path:
        return path
    # Common Windows location.
    candidates = [
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Windows\SysWOW64\nvidia-smi.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return "nvidia-smi"


def gpu_free_mib() -> int:
    try:
        out = subprocess.check_output(
            [_nvidia_smi_path(), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return int(out.strip().splitlines()[0].strip())
    except Exception:
        return 0


def is_running(name: str) -> bool:
    """Check if a run with the same output path already has a live python process."""
    marker = f"--output outputs/{name}.pth"
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any(marker in arg for arg in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def launch(name: str, flags: str) -> None:
    log_path = LOG_DIR / f"{name}.log"
    cmd = (
        f"python -u experiments/train_omniview_fusion_v5_webbridge_multi.py {flags} "
        f"--output outputs/{name}.pth >> {log_path} 2>&1"
    )
    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Launched {name}; log -> {log_path}")


COMMON_FLAGS = (
    "--use_mixed_loader "
    "--mixed_manifest configs/deprecated/circular/splits/webbridge_h36m_mpi_mixed_train_val.yaml "
    "--use_domain_embedding "
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
    "--v29_physical_loss_warmup_epochs 3 "
    "--use_hierarchical_multiview_v31 --v31_geometry_bias"
)

RUNS = [
    (
        "v34_hmsp_geometry_vjgn_stack_local_4090",
        COMMON_FLAGS
        + " --use_hierarchical_multiscale_spatial_pyramid_v33 --v33_hmsp_scales 1 2 4"
        + " --use_geometry_view_joint_graph_network_v34 --v34_gvjgn_n_layers 2 --v34_gvjgn_n_heads 4",
    ),
    (
        "v33_combined_fixed_plus_geometry_vjgn_local_4090",
        COMMON_FLAGS
        + " --use_uncertainty_aware_triangulation_v33 --v33_uat_loss_weight 0.01"
        + " --use_outlier_view_rejection_v33 --v33_outlier_z_thresh 3.0 --v33_outlier_soft_beta 1.0 --v33_outlier_supervised_weight 0.01"
        + " --use_ray_conditioned_attention_v33 --v33_n_heads 4 --v33_n_layers 2 --v33_use_ray_bias --v33_residual_gate_init -6.0"
        + " --use_hierarchical_multiscale_spatial_pyramid_v33 --v33_hmsp_scales 1 2 4"
        + " --use_geometry_view_joint_graph_network_v34 --v34_gvjgn_n_layers 2 --v34_gvjgn_n_heads 4"
        + " --weight_decay 1e-4",
    ),
]


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found; exiting.")
        sys.exit(1)

    for name, _ in RUNS:
        if is_running(name):
            print(f"{name} already running; skipping.")

    queue = list(RUNS)
    while queue:
        free = gpu_free_mib()
        if free < MIN_FREE_MIB:
            print(f"GPU free memory {free} MiB < {MIN_FREE_MIB} MiB; sleeping {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            continue

        name, flags = queue.pop(0)
        if is_running(name):
            print(f"{name} already running; skipping.")
            continue

        launch(name, flags)
        time.sleep(60)

    print("All local runs launched.")


if __name__ == "__main__":
    main()
