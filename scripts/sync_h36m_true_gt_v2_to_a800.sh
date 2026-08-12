#!/usr/bin/env bash
# Sync the corrected H36M true-GT v2 .npz files from local WSL to A800.
#
# Assumes the local files are in data/h36m_true_gt_v2/.
# Copies to the same path on A800.
#
# Usage:
#   bash scripts/sync_h36m_true_gt_v2_to_a800.sh

set -euo pipefail

A800_HOST=${A800_HOST:-a800-D}
A800_REPO=${A800_REPO:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20}
LOCAL_DIR=${LOCAL_DIR:-data/h36m_true_gt_v2}

# rsync over SSH (long-lived connection)
rsync -avz --progress \
    -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3" \
    "$LOCAL_DIR/" \
    "$A800_HOST:$A800_REPO/$LOCAL_DIR/"

echo "Synced $LOCAL_DIR to $A800_HOST:$A800_REPO/$LOCAL_DIR/"
echo "Next: update A800 training configs to use configs/splits/h36m_true_gt_v2_standard.yaml"
