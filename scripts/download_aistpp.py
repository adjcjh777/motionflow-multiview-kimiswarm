"""Download AIST++ annotations from the official GitHub release.

This script fetches the annotation-only archives (no videos) required for
converting AIST++ to the canonical WebBridge ``.npz`` format:

* keypoints2d.zip  (~1.3 GB)
* keypoints3d.zip  (~0.8 GB)
* motions.zip      (~0.1 GB)
* cameras.zip      (~0.03 MB)

The archives are extracted into ``data/webbridge/aistpp``. If any archive is
already present, it is skipped unless ``--force`` is passed.

Example
-------
    conda run -n mf python scripts/download_aistpp.py --out data/webbridge/aistpp

Reference
---------
https://google.github.io/aistplusplus_dataset/download.html
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


RELEASE_URL = "https://github.com/google/aistplusplus_dataset/releases/download/v1.0"

ARCHIVES = {
    "keypoints2d.zip": "keypoints2d.zip",
    "keypoints3d.zip": "keypoints3d.zip",
    "motions.zip": "motions.zip",
    "cameras.zip": "cameras.zip",
}


def download_file(url: str, out_path: Path) -> None:
    """Download ``url`` to ``out_path`` using curl, showing progress."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-L",
        "-C",
        "-",
        "--retry",
        "5",
        "--fail",
        "-o",
        str(out_path),
        url,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download AIST++ annotations.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/webbridge/aistpp"),
        help="Destination directory for the archives and extracted files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download archives even if they already exist.")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Only download archives; do not extract them.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for archive_name in ARCHIVES.values():
        archive_path = out_dir / archive_name
        if archive_path.exists() and not args.force:
            print(f"Skipping {archive_name}, already exists.")
        else:
            url = f"{RELEASE_URL}/{archive_name}"
            print(f"Downloading {archive_name} from {url} ...")
            download_file(url, archive_path)
            print(f"Saved {archive_path}")

        if not args.skip_extract:
            print(f"Extracting {archive_name} ...")
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(out_dir)
            print(f"Extracted {archive_name} to {out_dir}")

    print("AIST++ annotations ready at", out_dir)


if __name__ == "__main__":
    main()
