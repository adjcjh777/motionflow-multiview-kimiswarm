#!/usr/bin/env bash
# Run failure analysis for the cross-view PP model on WSL.
# Adjust DATASET and CHECKPOINT to match the target .npz and checkpoint.
set -e
cd /mnt/d/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm

DATASET=${1:-data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz}
CHECKPOINT=${2:-outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth}

conda run -n mf python experiments/analyze_failures_crossview_pp.py \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128 \
  --batch_size 32 \
  --out_dir outputs/failure_analysis_crossview_pp \
  --report_dir docs/swarm_iter_next
