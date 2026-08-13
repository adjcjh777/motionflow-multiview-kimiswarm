#!/usr/bin/env bash
# Run scripts/prepare_mpi_submission.py on the A800 host.
#
# Default GPU: 6 (project policy for MotionFlow-MultiView).
# Override with:
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_mpi_submission_a800.sh ...
# or
#   bash scripts/run_mpi_submission_a800.sh --gpu 7 ...
#
# The script rsyncs this repo to the A800, then runs the Python submission
# script via ssh. It is safe to run from the local WSL checkout.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
A800_HOST="${A800_HOST:-a800-D}"
A800_REPO="${A800_REPO:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"

# Parse optional --gpu / --host / --a800-repo from the front of the argument list.
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --gpu=*)
            GPU="${1#*=}"
            shift
            ;;
        --host)
            A800_HOST="$2"
            shift 2
            ;;
        --host=*)
            A800_HOST="${1#*=}"
            shift
            ;;
        --a800-repo)
            A800_REPO="$2"
            shift 2
            ;;
        --a800-repo=*)
            A800_REPO="${1#*=}"
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Remaining arguments are passed to prepare_mpi_submission.py.
PYTHON_ARGS=("$@")

# If no arguments were given, print usage and exit.
if [[ "${#PYTHON_ARGS[@]}" -eq 0 ]]; then
    cat <<'USAGE'
Usage: bash scripts/run_mpi_submission_a800.sh [--gpu N] [--host HOST] [--a800-repo PATH] --config CONFIG --checkpoint PTH [extra args]

Required Python arguments (passed through to scripts/prepare_mpi_submission.py):
  --config CONFIG      YAML training config, e.g.
                       configs/ablations/v25_true_gt_v2_medium_a800.yaml
  --checkpoint PTH     Checkpoint path, e.g.
                       outputs/ablations/v25_true_gt_v2_medium_a800.pth

Optional:
  --gpu N              CUDA device to use on A800 (default: 6).
  --host HOST          SSH host alias/name for A800 (default: a800-D).
  --a800-repo PATH     Remote repo path on A800 (default: /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20).
  --method_name NAME   Method name for the submission manifest.
  --test_root PATH     Path to official MPI test set root (optional; enables local-eval mat).
  --clip_len N         Temporal clip length (default: 13).
  --batch_size N       Inference batch size (default: 8).
  --stride N           Clip stride (default: 1).
  --device auto|cpu|cuda:0  PyTorch device (default: auto).
  --no_zip             Do not zip the submission.

Examples:
  # v25 true-GT v2 medium
  bash scripts/run_mpi_submission_a800.sh \
      --config configs/ablations/v25_true_gt_v2_medium_a800.yaml \
      --checkpoint outputs/ablations/v25_true_gt_v2_medium_a800.pth \
      --method_name v25_true_gt_v2_medium

  # v85 random view dropout
  bash scripts/run_mpi_submission_a800.sh \
      --config configs/ablations/v85_random_view_dropout_medium_a800.yaml \
      --checkpoint outputs/ablations/v85_random_view_dropout_medium_a800.pth \
      --method_name v85_random_view_dropout

USAGE
    exit 0
fi

# Validate required arguments are present.
has_config=false
has_checkpoint=false
for arg in "${PYTHON_ARGS[@]}"; do
    [[ "$arg" == "--config" ]] && has_config=true
    [[ "$arg" == "--checkpoint" ]] && has_checkpoint=true
done

if [[ "$has_config" == false ]] || [[ "$has_checkpoint" == false ]]; then
    echo "Error: --config and --checkpoint are required." >&2
    echo "Run without arguments to see usage." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Sync code to A800 and run.
# ---------------------------------------------------------------------------
echo "A800 host:  $A800_HOST"
echo "Remote:     $A800_REPO"
echo "GPU:        $GPU"
echo "Command:    python scripts/prepare_mpi_submission.py ${PYTHON_ARGS[*]}"

# Push local changes. (Adjust rsync excludes as needed.)
rsync -avz --exclude='.git/' --exclude='outputs/' --exclude='tmp/' --exclude='data/' \
    "$REPO_ROOT/" "${A800_HOST}:${A800_REPO}/"

# Run on A800 via ssh. Use nohup so the session can close without killing the job.
# Output is redirected to a timestamped log in outputs/mpi_submissions/.
LOG_DIR="${A800_REPO}/outputs/mpi_submissions"
LOG_FILE="${LOG_DIR}/run_mpi_submission_$(date +%Y%m%d_%H%M%S).log"

ssh "$A800_HOST" "mkdir -p '${LOG_DIR}' && cd '${A800_REPO}' && CUDA_VISIBLE_DEVICES=${GPU} nohup python scripts/prepare_mpi_submission.py ${PYTHON_ARGS[*]} > '${LOG_FILE}' 2>&1 &"

echo "Submitted to A800."
echo "  Host:     $A800_HOST"
echo "  GPU:      $GPU"
echo "  Log:      $LOG_FILE"
echo "Monitor with:"
printf '  ssh %s "tail -f %s"\n' "$A800_HOST" "$LOG_FILE"
