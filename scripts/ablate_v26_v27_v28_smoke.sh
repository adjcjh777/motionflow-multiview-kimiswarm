#!/usr/bin/env bash
# Smoke ablation matrix for v26/v27/v28 failure analysis.
#
# Runs one-epoch synthetic-smoke trainings for a set of v25/v26/v27/v28
# combinations and writes a CSV summary of val_MPJPE.
#
# Usage:
#   bash scripts/ablate_v26_v27_v28_smoke.sh
#
# The harness is intentionally CPU-friendly so it does not contend with any
# long-running GPU training on the local RTX 4090.
#
set -euo pipefail

# Defensive: stale .pyc files can hide recent source changes when multiple
# agents edit the same modules. Clear them before each sweep.
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-999}
PYTHON=${PYTHON:-python}
OUT_DIR=${OUT_DIR:-outputs/ablate_v26_v27_v28_smoke}
SEED=${SEED:-42}

mkdir -p "$OUT_DIR"

# Each entry: "name|extra_flags"
ABLATIONS=(
  "v25_baseline|--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment"
  "v26|--use_temporal_geometry_fusion_v26 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment"
  "v25_udp|--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment --use_uncertainty_depth_proposals_v27"
  "v26_udp|--use_temporal_geometry_fusion_v26 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment --use_uncertainty_depth_proposals_v27"
  "v26_udp_v28|--use_temporal_geometry_fusion_v26 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment --use_uncertainty_depth_proposals_v27 --use_physical_space_alignment_v28 --v28_floor_loss_weight 0.01 --v28_bone_temporal_weight 0.01"
)

SUMMARY="$OUT_DIR/summary.csv"
echo "name,val_mpjpe_mm,val_loss" > "$SUMMARY"

for entry in "${ABLATIONS[@]}"; do
  name="${entry%%|*}"
  flags="${entry#*|}"
  ckpt="$OUT_DIR/${name}.pth"
  log="$OUT_DIR/${name}.log"

  echo ""
  echo "=== Running $name ==="
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    $PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
      --smoke \
      --seed "$SEED" \
      --epochs 1 \
      $flags \
      --output "$ckpt" \
      > "$log" 2>&1

  # Parse val_MPJPE and val_loss from the log.
  val_mpjpe=$(grep -oP 'Epoch 1: .*val_MPJPE=\K[0-9.]+' "$log" || echo "")
  val_loss=$(grep -oP 'Epoch 1: .*val_loss=\K[0-9.]+' "$log" || echo "")
  if [[ -z "$val_mpjpe" ]]; then
    val_mpjpe=$(grep -oP 'Best val MPJPE: \K[0-9.]+' "$log" || echo "NA")
  fi
  if [[ -z "$val_loss" ]]; then
    val_loss="NA"
  fi

  echo "$name,$val_mpjpe,$val_loss" >> "$SUMMARY"
  echo "  $name -> val_MPJPE=${val_mpjpe}mm val_loss=${val_loss}"
done

echo ""
echo "Summary written to $SUMMARY"
cat "$SUMMARY"
