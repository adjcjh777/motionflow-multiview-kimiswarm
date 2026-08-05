"""Batch convert MPI-INF-3DHP subjects/sequences to the canonical WebBridge .npz.

This script discovers raw ``annot.mat`` / ``camera.calibration`` pairs under a
given root, optionally downloads missing sequences from the official MPI-INF
host, and writes canonical multi-view .npz files for both 14-view and 4-view
subsets, in millimeters and meters.

Usage
-----
    # Convert every locally available S* / Seq* pair.
    conda run -n mf python experiments/batch_convert_mpiinf3dhp_v1.py

    # Download + convert S3/Seq1 and S4/Seq1 (only annot.mat + camera.calibration).
    conda run -n mf python experiments/batch_convert_mpiinf3dhp_v1.py \
        --subjects 3 4 --sequences 1 --download --yes-download

    # Convert only, with custom paths.
    conda run -n mf python experiments/batch_convert_mpiinf3dhp_v1.py \
        --raw_root data/webbridge/mpi_inf_3dhp/raw \
        --out_dir data/webbridge/mpi_inf_3dhp

The downloader fetches only the annotation/calibration files (~200 MB per
sequence) because image frames are not required by the canonical .npz format.
Videos are therefore skipped.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.webbridge_loader import convert_mpiinf3dhp


BASE_URL = "https://vcai.mpi-inf.mpg.de/3dhp-dataset"


def _subject_seq_pairs(args):
    """Return the (subject, seq) pairs to process."""
    subjects = args.subjects or list(range(1, 9))
    sequences = args.sequences or [1, 2]
    return [(s, q) for s in subjects for q in sequences]


def _find_missing(pairs, raw_root):
    """Return pairs whose annot.mat/camera.calibration are not present."""
    missing = []
    for s, q in pairs:
        seq_dir = raw_root / f"S{s}" / f"Seq{q}"
        if not (seq_dir / "annot.mat").exists() or not (seq_dir / "camera.calibration").exists():
            missing.append((s, q))
    return missing


def _download_file(url, dest):
    """Download *url* to *dest* with progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response:
        total = int(response.headers.get("Content-Length", 0))
        block = 8192
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(block)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {dest.name}: {pct:.1f}%", end="", flush=True)
    print()  # newline after progress


def _download_sequence(subject, seq, raw_root, yes_download=False):
    """Download annot.mat and camera.calibration for one subject/sequence."""
    seq_dir = raw_root / f"S{subject}" / f"Seq{seq}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    annot_url = f"{BASE_URL}/S{subject}/Seq{seq}/annot.mat"
    calib_url = f"{BASE_URL}/S{subject}/Seq{seq}/camera.calibration"

    annot_path = seq_dir / "annot.mat"
    calib_path = seq_dir / "camera.calibration"

    if not annot_path.exists():
        if not yes_download:
            print(f"  Would download {annot_url} -> {annot_path}")
            return False
        print(f"Downloading S{subject}/Seq{seq} annot.mat ...")
        _download_file(annot_url, annot_path)

    if not calib_path.exists():
        if not yes_download:
            print(f"  Would download {calib_url} -> {calib_path}")
            return False
        print(f"Downloading S{subject}/Seq{seq} camera.calibration ...")
        _download_file(calib_url, calib_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert MPI-INF-3DHP to canonical .npz."
    )
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=Path("data/webbridge/mpi_inf_3dhp/raw"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/webbridge/mpi_inf_3dhp"),
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[],
        help="Subjects to process (default: 1-8).",
    )
    parser.add_argument(
        "--sequences",
        type=int,
        nargs="+",
        default=[],
        help="Sequences to process (default: 1,2).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing annot.mat/camera.calibration files.",
    )
    parser.add_argument(
        "--yes-download",
        action="store_true",
        dest="yes_download",
        help="Confirm downloads (without this, only dry-run listing).",
    )
    parser.add_argument(
        "--skip-v4",
        action="store_true",
        dest="skip_v4",
        help="Skip the 4-view subset.",
    )
    parser.add_argument(
        "--skip-v14",
        action="store_true",
        dest="skip_v14",
        help="Skip the 14-view subset.",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _subject_seq_pairs(args)
    print(f"Requested pairs: {pairs}")

    if args.download:
        missing = _find_missing(pairs, raw_root)
        if missing:
            print(f"Missing {len(missing)} sequence(s): {missing}")
            for s, q in missing:
                ok = _download_sequence(s, q, raw_root, yes_download=args.yes_download)
                if not ok and not args.yes_download:
                    print("  Re-run with --yes-download to fetch files.")
                    return
        else:
            print("All requested sequences already present locally.")

    # Convert every pair that is now available.
    for s, q in pairs:
        seq_dir = raw_root / f"S{s}" / f"Seq{q}"
        if not (seq_dir / "annot.mat").exists() or not (seq_dir / "camera.calibration").exists():
            print(f"Skipping S{s}/Seq{q}: raw files not found at {seq_dir}")
            continue

        # 14-view
        if not args.skip_v14:
            out_v14 = out_dir / f"s_{s:02d}_seq_{q:02d}_v14_multiview.npz"
            out_v14_m = out_dir / f"s_{s:02d}_seq_{q:02d}_v14_multiview_m.npz"
            if not out_v14.exists():
                print(f"Converting S{s}/Seq{q} -> 14-view ...")
                convert_mpiinf3dhp(seq_dir, out_v14, n_views=None)
            else:
                print(f"Exists: {out_v14}")

            if not out_v14_m.exists():
                print(f"  -> meters variant ...")
                data = dict(np.load(out_v14))
                data["camera_t"] = data["camera_t"] / 1000.0
                data["joints_3d"] = data["joints_3d"] / 1000.0
                np.savez(out_v14_m, **data)

        # 4-view
        if not args.skip_v4:
            out_v4 = out_dir / f"s_{s:02d}_seq_{q:02d}_v4_multiview.npz"
            out_v4_m = out_dir / f"s_{s:02d}_seq_{q:02d}_v4_multiview_m.npz"
            if not out_v4.exists():
                print(f"Converting S{s}/Seq{q} -> 4-view ...")
                convert_mpiinf3dhp(seq_dir, out_v4, n_views=4)
            else:
                print(f"Exists: {out_v4}")

            if not out_v4_m.exists():
                print(f"  -> meters variant ...")
                data = dict(np.load(out_v4))
                data["camera_t"] = data["camera_t"] / 1000.0
                data["joints_3d"] = data["joints_3d"] / 1000.0
                np.savez(out_v4_m, **data)

    print("Batch conversion complete.")


if __name__ == "__main__":
    main()
