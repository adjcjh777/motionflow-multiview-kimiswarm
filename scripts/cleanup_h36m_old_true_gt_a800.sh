#!/usr/bin/env bash
# Remove the old circular/misaligned H36M .npz files on A800 after v85 finishes.
#
# The old files are:
#   - data/h36m_hf/*.npz                         : circular labels (direct MJE ~0 mm)
#   - data/h36m_true_gt/*_multiview.npz          : misaligned with stored cameras/2D
#   - data/h36m_true_gt/*_multiview_m.npz        : metre-convention, misaligned
#
# They will be replaced by the corrected v2 files in data/h36m_true_gt_v2/.
#
# Usage (on A800, after v85 training has finished):
#   bash scripts/cleanup_h36m_old_true_gt_a800.sh

set -euo pipefail

REPO=${REPO:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20}
cd "$REPO"

ARCHIVE_DIR="archive_cleanup_h36m_old_true_gt_$(date +%Y%m%d)"
mkdir -p "$ARCHIVE_DIR"

# Move (not delete) the misaligned true-GT files so they can be recovered if needed.
for f in data/h36m_true_gt/*.npz; do
    [ -e "$f" ] || continue
    mv -v "$f" "$ARCHIVE_DIR/"
done

# The circular h36m_hf files are not used by the new protocol; archive them too.
for f in data/h36m_hf/*.npz; do
    [ -e "$f" ] || continue
    mv -v "$f" "$ARCHIVE_DIR/"
done

echo "Archived old H36M .npz to $ARCHIVE_DIR"
echo "Next: copy data/h36m_true_gt_v2/*.npz to A800 and update the training manifest."
