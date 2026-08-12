#!/usr/bin/env bash
# Dry-run A800 disk cleanup. Shows what would be archived/deleted.

set -euo pipefail

PROJECT_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
cd "$PROJECT_ROOT"

echo "=== Old root-level .pth in outputs/ ==="
find outputs -maxdepth 1 -name '*.pth' -type f -printf "  %p  --  %s bytes\n" | sort || true

echo ""
echo "=== Abandoned v83 checkpoint ==="
if [ -f "outputs/ablations/v83_true_gt_h36m_medium_a800.pth" ]; then
    stat --printf="  %n  --  %s bytes\n" "outputs/ablations/v83_true_gt_h36m_medium_a800.pth"
else
    echo "  (not found)"
fi

echo ""
echo "=== Staging tarball ==="
if [ -f "tmp_upload/aist_subset_for_a800.tar" ]; then
    stat --printf="  %n  --  %s bytes\n" "tmp_upload/aist_subset_for_a800.tar"
else
    echo "  (not found)"
fi
