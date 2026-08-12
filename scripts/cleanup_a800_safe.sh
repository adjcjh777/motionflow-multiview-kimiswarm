#!/usr/bin/env bash
# Safe A800 disk cleanup script.
# Removes completed-run duplicate checkpoints and package-manager caches.
# Active checkpoints are detected by name and skipped.

set -euo pipefail

PROJECT_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
cd "$PROJECT_ROOT"

echo "=== A800 disk cleanup ==="
df -h /mnt/nvme0n1p1

# 1. Remove non-final duplicate checkpoints when a corresponding *_final.pth exists.
#    Skip any basename that is still running as a training/eval process.
echo "Scanning outputs/ablations for safe checkpoint duplicates..."
find outputs/ablations -maxdepth 1 -name '*_final.pth' -type f | while read -r final_path; do
    base="${final_path%_final.pth}"
    nonfinal="${base}.pth"
    name="$(basename "$base").pth"

    if [ ! -f "$nonfinal" ]; then
        continue
    fi

    if pgrep -f "$name" >/dev/null 2>&1; then
        echo "  SKIP (active process): $nonfinal"
    else
        echo "  REMOVE: $nonfinal"
        rm -f "$nonfinal"
    fi
done

# 2. Remove the abandoned/diverged mixed-dataset checkpoint.
MIXED_CKPT="outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.pth"
if [ -f "$MIXED_CKPT" ]; then
    if pgrep -f "v25_true_gt_mixed_dataset_stability_a800_gpu6" >/dev/null 2>&1; then
        echo "  SKIP (active process): $MIXED_CKPT"
    else
        echo "  REMOVE: $MIXED_CKPT"
        rm -f "$MIXED_CKPT"
    fi
fi

# 3. Purge pip package cache.
if command -v pip >/dev/null 2>&1; then
    echo "Purging pip cache..."
    yes | pip cache purge || true
fi

# 4. Purge uv package cache (uv not in PATH on A800, so fall back to directory removal).
UV_CACHE_DIR="/mnt/nvme0n1p1/zhangzy/.cache/uv"
if command -v uv >/dev/null 2>&1; then
    echo "Pruning uv cache..."
    uv cache prune || true
elif [ -d "$UV_CACHE_DIR" ]; then
    echo "Removing uv cache directory: $UV_CACHE_DIR"
    rm -rf "$UV_CACHE_DIR"
fi

echo ""
echo "=== Cleanup complete ==="
df -h /mnt/nvme0n1p1
