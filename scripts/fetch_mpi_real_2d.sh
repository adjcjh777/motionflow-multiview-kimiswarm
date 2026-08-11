#!/usr/bin/env bash
# Fetch raw MPI-INF-3DHP training videos and generate real (detected) 2D .npz.
#
# Two-stage wrapper:
#   1. Ensure raw S1..S8 camera zips are present under
#      data/webbridge/mpi_inf_3dhp/raw/ (uses scripts/download_mpi_train_images.sh).
#   2. Run MediaPipe Pose detection directly from the in-zip AVIs with
#      scripts/generate_mpi_detected_2d_from_avi.py to produce
#      data/webbridge/mpi_inf_3dhp_detected_2d/.
#
# The script is idempotent: missing zips are resumed, and detection will reuse
# the existing canonical .npz files under data/webbridge/mpi_inf_3dhp/.
#
# Examples
#   # Full pipeline on CPU (default, ~hours depending on GPU).
#   bash scripts/fetch_mpi_real_2d.sh
#
#   # Smoke-test: first 300 frames only.
#   bash scripts/fetch_mpi_real_2d.sh --smoke
#
#   # CUDA detection with custom Python / model.
#   CUDA_VISIBLE_DEVICES=0 PYTHON=python3 bash scripts/fetch_mpi_real_2d.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
RAW_DIR="${RAW_DIR:-data/webbridge/mpi_inf_3dhp/raw}"
INPUT_DIR="${INPUT_DIR:-data/webbridge/mpi_inf_3dhp}"
OUTPUT_DIR="${OUTPUT_DIR:-data/webbridge/mpi_inf_3dhp_detected_2d}"
MODEL="${MODEL:-models/mediapipe/pose_landmarker_full.task}"
DETECT_SIZE="${DETECT_SIZE:-384}"
WORKERS="${WORKERS:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
SMOKE=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE=true
      MAX_FRAMES=300
      shift
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --detect-size)
      DETECT_SIZE="$2"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --raw-dir)
      RAW_DIR="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--smoke] [--workers N] [--model PATH] [--detect-size N]"
      echo "         [--max-frames N] [--output-dir PATH] [--raw-dir PATH]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$RAW_DIR" "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 1. Ensure a Python interpreter with mediapipe, numpy, and opencv is available.
# ---------------------------------------------------------------------------
if ! command -v "$PYTHON" &>/dev/null; then
  echo "ERROR: Python interpreter not found: $PYTHON" >&2
  exit 1
fi

missing_pkgs=()
for pkg in mediapipe numpy cv2; do
  if ! "$PYTHON" -c "import $pkg" 2>/dev/null; then
    missing_pkgs+=("$pkg")
  fi
done

if [[ ${#missing_pkgs[@]} -gt 0 ]]; then
  echo "ERROR: Missing Python packages: ${missing_pkgs[*]}" >&2
  echo "Install with: $PYTHON -m pip install mediapipe opencv-python numpy" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Download raw camera zips if any are missing.
# ---------------------------------------------------------------------------
need_download=false
for subject in 1 2 3 4 5 6 7 8; do
  for seq in 1 2; do
    for group in vnect_cameras.zip other_angled_cameras.zip ceiling_cameras.zip; do
      zip_path="$RAW_DIR/S$subject/Seq$seq/imageSequence/$group"
      if [[ ! -s "$zip_path" ]]; then
        need_download=true
        break 3
      fi
    done
  done
done

if ! $need_download; then
  echo "[fetch_mpi_real_2d] Raw MPI-INF-3DHP camera zips already present."
else
  echo "[fetch_mpi_real_2d] Downloading missing raw MPI-INF-3DHP camera zips..."
  bash scripts/download_mpi_train_images.sh
fi

# ---------------------------------------------------------------------------
# 3. Verify MediaPipe model asset.
# ---------------------------------------------------------------------------
if [[ ! -s "$MODEL" ]]; then
  echo "ERROR: MediaPipe model not found: $MODEL" >&2
  echo "Download a pose landmarker model from Google MediaPipe and place it at the path above." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 4. Generate detected-2D .npz files from the raw AVIs.
# ---------------------------------------------------------------------------
extra_args=()
if [[ "$MAX_FRAMES" -gt 0 ]]; then
  extra_args+=(--max_frames "$MAX_FRAMES")
fi

echo "[fetch_mpi_real_2d] Generating detected-2D MPI-INF-3DHP .npz files..."
echo "  python      : $PYTHON"
echo "  raw_dir     : $RAW_DIR"
echo "  input_dir   : $INPUT_DIR"
echo "  output_dir  : $OUTPUT_DIR"
echo "  model       : $MODEL"
echo "  detect_size : $DETECT_SIZE"
echo "  workers     : $WORKERS"
if [[ "$MAX_FRAMES" -gt 0 ]]; then
  echo "  max_frames  : $MAX_FRAMES (smoke mode)"
fi

"$PYTHON" scripts/generate_mpi_detected_2d_from_avi.py \
  --raw_dir "$RAW_DIR" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --detect_size "$DETECT_SIZE" \
  --workers "$WORKERS" \
  --subjects 1,2,3,4,5,6,7,8 \
  --seqs 1,2 \
  "${extra_args[@]}"

echo "[fetch_mpi_real_2d] Done. Output: $OUTPUT_DIR"
