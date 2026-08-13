#!/usr/bin/env bash
set -euo pipefail
pkill -f launch_v81_after_v85_dlt_fallback.sh || true
sleep 2
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20
chmod +x scripts/launch_v81_after_v85_dlt_fallback.sh
nohup bash scripts/launch_v81_after_v85_dlt_fallback.sh > outputs/launch_v81_after_v85_dlt_fallback.log 2>&1 &
sleep 3
ps -ef | grep launch_v81_after_v85 | grep -v grep || true
tail -n 5 outputs/launch_v81_after_v85_dlt_fallback.log || true
