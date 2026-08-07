#!/usr/bin/env bash
/usr/bin/tmux kill-session -t omniview_v2_full 2>/dev/null || true
cd /mnt/d/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm
/usr/bin/tmux new-session -d -s omniview_v2_full "bash scripts/run_omniview_fusion_v2_full_wsl.sh 2>&1 | tee outputs/omniview_fusion_v2_d128_no_graph.log"
