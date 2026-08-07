#!/usr/bin/env bash
# Poll for completed OmniMultiViewFusion v2/v3 checkpoints and run full eval +
# robustness + variable-view curve once, then post a summary to GitHub issue #74.
#
# Designed to be safe to run repeatedly; it creates a .eval_done marker next to
# each checkpoint so evaluation runs only once.
set -euo pipefail

LOCK="/tmp/motionflow_auto_eval.lock"
if [ -f "$LOCK" ]; then
    echo "Another auto-eval is already running."
    exit 0
fi
touch "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd "$(dirname "$0")/.."

GH_BIN="/tmp/gh/bin/gh.exe"
REPO="adjcjh777/motionflow-multiview-kimiswarm"
ISSUE="74"

# Resolve GitHub token from git credential helper.
GH_TOKEN="${GH_TOKEN:-}"
if [ -z "$GH_TOKEN" ] && command -v git >/dev/null 2>&1; then
    GH_TOKEN=$(printf 'url=https://github.com/adjcjh777/motionflow-multiview-kimiswarm.git\n\n' | git credential fill 2>/dev/null | awk -F= '/^password=/{print $2}')
fi
export GH_TOKEN

post_comment() {
    local body="$1"
    if [ -n "$GH_TOKEN" ] && [ -f "$GH_BIN" ]; then
        "$GH_BIN" api "repos/$REPO/issues/$ISSUE/comments" -X POST -f body="$body" >/dev/null 2>&1 || true
    fi
}

# Common model hyperparameters used during training.
V2_ARGS="--d 128 --residual_hidden 256 --n_st_layers 3 --graph_num_layers 1 --n_joint_layers 1"

run_eval_local() {
    local ckpt="$1"
    local dataset="$2"
    local out_json="$3"
    local out_csv="$4"
    local log_file="$5"

    if [ -f "$out_json" ]; then
        return 0
    fi

    python experiments/eval_omniview_fusion_v2_mpiinf3dhp.py \
        $V2_ARGS \
        --checkpoint "$ckpt" \
        --dataset "$dataset" \
        --run_robustness \
        --run_variable_views \
        --out_json "$out_json" \
        --out_csv "$out_csv" \
        >"$log_file" 2>&1
}

extract_mpjpe() {
    python - <<PY
import json, sys
try:
    with open('$1') as f:
        d = json.load(f)
    print(f"{d['clean']['mpjpe']:.2f}")
except Exception:
    print("N/A")
PY
}

# ---------------------------------------------------------------------------
# 4090 WSL dense+graph v2
# ---------------------------------------------------------------------------
CKPT_4090="outputs/omniview_fusion_v2_d128_dense_graph_v2.pth"
if [ -f "$CKPT_4090" ] && [ ! -f "$CKPT_4090.eval_done" ]; then
    DATA_4090="data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz"
    JSON_4090="outputs/eval_omniview_fusion_v2_d128_dense_graph_v2.json"
    CSV_4090="outputs/eval_omniview_fusion_v2_d128_dense_graph_v2.csv"
    LOG_4090="outputs/eval_omniview_fusion_v2_d128_dense_graph_v2.log"
    if run_eval_local "$CKPT_4090" "$DATA_4090" "$JSON_4090" "$CSV_4090" "$LOG_4090"; then
        touch "$CKPT_4090.eval_done"
        MPJPE=$(extract_mpjpe "$JSON_4090")
        post_comment "Auto-eval: 4090 dense+graph v2 checkpoint ready. Clean MPJPE=${MPJPE} mm. See $JSON_4090."
    fi
fi

# ---------------------------------------------------------------------------
# A800-D v2 MPI (accessed via ssh)
# ---------------------------------------------------------------------------
A800_REPO="/mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm"
CKPT_A800="$A800_REPO/outputs/omniview_fusion_v2_d128_dense_graph_v2_a800.pth"
if ssh a800-D "test -f $CKPT_A800 && ! test -f $CKPT_A800.eval_done" >/dev/null 2>&1; then
    JSON_A800="$A800_REPO/outputs/eval_omniview_fusion_v2_d128_dense_graph_v2_a800.json"
    CSV_A800="$A800_REPO/outputs/eval_omniview_fusion_v2_d128_dense_graph_v2_a800.csv"
    LOG_A800="$A800_REPO/outputs/eval_omniview_fusion_v2_d128_dense_graph_v2_a800.log"
    if ssh a800-D "cd $A800_REPO && source .venv/bin/activate && python experiments/eval_omniview_fusion_v2_mpiinf3dhp.py $V2_ARGS --checkpoint outputs/omniview_fusion_v2_d128_dense_graph_v2_a800.pth --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz --run_robustness --run_variable_views --out_json $JSON_A800 --out_csv $CSV_A800 >$LOG_A800 2>&1"; then
        ssh a800-D "touch $CKPT_A800.eval_done"
        MPJPE=$(extract_mpjpe <(ssh a800-D "cat $JSON_A800"))
        post_comment "Auto-eval: A800-D v2 MPI checkpoint ready. Clean MPJPE=${MPJPE} mm. See $JSON_A800."
    fi
fi

# ---------------------------------------------------------------------------
# A800-D H36M v2
# ---------------------------------------------------------------------------
CKPT_H36M="$A800_REPO/outputs/omniview_fusion_v2_h36m_d128_dense_graph_a800.pth"
if ssh a800-D "test -f $CKPT_H36M && ! test -f $CKPT_H36M.eval_done" >/dev/null 2>&1; then
    JSON_H36M="$A800_REPO/outputs/eval_omniview_fusion_v2_h36m_d128_dense_graph_a800.json"
    CSV_H36M="$A800_REPO/outputs/eval_omniview_fusion_v2_h36m_d128_dense_graph_a800.csv"
    LOG_H36M="$A800_REPO/outputs/eval_omniview_fusion_v2_h36m_d128_dense_graph_a800.log"
    VAL_H36M="data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz"
    if ssh a800-D "cd $A800_REPO && source .venv/bin/activate && python experiments/eval_omniview_fusion_v2_mpiinf3dhp.py $V2_ARGS --checkpoint outputs/omniview_fusion_v2_h36m_d128_dense_graph_a800.pth --dataset $VAL_H36M --run_robustness --run_variable_views --out_json $JSON_H36M --out_csv $CSV_H36M >$LOG_H36M 2>&1"; then
        ssh a800-D "touch $CKPT_H36M.eval_done"
        MPJPE=$(extract_mpjpe <(ssh a800-D "cat $JSON_H36M"))
        post_comment "Auto-eval: A800-D H36M v2 checkpoint ready. Clean MPJPE=${MPJPE} mm. See $JSON_H36M."
    fi
fi

# ---------------------------------------------------------------------------
# A800-D v3 MPI
# ---------------------------------------------------------------------------
CKPT_V3="$A800_REPO/outputs/omniview_fusion_v3_d128_dense_graph_v2_a800_no_warm.pth"
if ssh a800-D "test -f $CKPT_V3 && ! test -f $CKPT_V3.eval_done" >/dev/null 2>&1; then
    JSON_V3="$A800_REPO/outputs/eval_omniview_fusion_v3_d128_dense_graph_v2_a800.json"
    CSV_V3="$A800_REPO/outputs/eval_omniview_fusion_v3_d128_dense_graph_v2_a800.csv"
    LOG_V3="$A800_REPO/outputs/eval_omniview_fusion_v3_d128_dense_graph_v2_a800.log"
    V3_FLAGS="--use_multiscale_fusion --use_camera_conditioning --use_epipolar_bias"
    if ssh a800-D "cd $A800_REPO && source .venv/bin/activate && python experiments/eval_omniview_fusion_v3_mpiinf3dhp.py $V2_ARGS $V3_FLAGS --checkpoint outputs/omniview_fusion_v3_d128_dense_graph_v2_a800_no_warm.pth --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz --run_robustness --run_variable_views --out_json $JSON_V3 --out_csv $CSV_V3 >$LOG_V3 2>&1"; then
        ssh a800-D "touch $CKPT_V3.eval_done"
        MPJPE=$(extract_mpjpe <(ssh a800-D "cat $JSON_V3"))
        post_comment "Auto-eval: A800-D v3 MPI checkpoint ready. Clean MPJPE=${MPJPE} mm. See $JSON_V3."
    fi
fi
