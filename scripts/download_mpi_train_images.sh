#!/usr/bin/env bash
# Download MPI-INF-3DHP training-set imageSequence zips (vnect cameras only)
# from the official server, per the official get_dataset.sh pattern
# (subjects S1..S8, Seq1+Seq2, no extra wall/ceiling cameras, no masks).
#
# ~420 MB per sequence, 16 sequences -> ~6.5 GB. The academic server is slow;
# curl -C - resumes interrupted downloads. Logs to
# data/webbridge/mpi_inf_3dhp/download_train_images.log.
#
# Unzip step is intentionally separate (scripts/unzip_mpi_train_images.sh)
# because extracting ~7 GB of JPEGs is best done after the full download
# completes (disk: 7 GB zip + 7 GB extracted transiently).
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_URL="http://gvv.mpi-inf.mpg.de/3dhp-dataset"   # redirects to vcai.mpi-inf.mpg.de
DEST=data/webbridge/mpi_inf_3dhp/raw
LOG=data/webbridge/mpi_inf_3dhp/download_train_images.log
mkdir -p "$DEST"
: >> "$LOG"

for subject in 1 2 3 4 5 6 7 8; do
  for seq in 1 2; do
    DIR="$DEST/S$subject/Seq$seq/imageSequence"
    mkdir -p "$DIR"
    # annot.mat / camera.calibration may already exist from the audited raw
    # layout; fetch only if missing.
    for f in annot.mat camera.calibration; do
      if [ ! -s "$DEST/S$subject/Seq$seq/$f" ]; then
        curl -sSL "$BASE_URL/S$subject/Seq$seq/$f" -o "$DEST/S$subject/Seq$seq/$f.tmp" \
          && mv "$DEST/S$subject/Seq$seq/$f.tmp" "$DEST/S$subject/Seq$seq/$f"
      fi
    done
    ZIP="$DIR/vnect_cameras.zip"
    if [ ! -s "$ZIP" ]; then
      echo "$(date -Is) downloading S$subject/Seq$seq ..." >> "$LOG"
      # curl exit 33 = resume offset already at EOF (file complete); accept it.
      rc=0
      curl -SL -C - "$BASE_URL/S$subject/Seq$seq/imageSequence/vnect_cameras.zip" -o "$ZIP.part" \
        2>> "$LOG" || rc=$?
      if [ "${rc:-0}" -eq 0 ] || [ "${rc:-0}" -eq 33 ]; then
        mv "$ZIP.part" "$ZIP"
      else
        echo "$(date -Is) S$subject/Seq$seq curl rc=$rc; keeping .part for resume" >> "$LOG"
      fi
    fi
    echo "$(date -Is) S$subject/Seq$seq done ($(du -h "$ZIP" 2>/dev/null | cut -f1))" >> "$LOG"
  done
done
echo "$(date -Is) ALL DOWNLOADS COMPLETE" >> "$LOG"
