"""WebBridge dataset sourcing for motionflow-multiview.

This script collects publicly accessible direct-download and HuggingFace URLs for
six benchmark datasets used by the MotionFlow-MultiView ray-attention fusion
pipeline, and provides a minimal command-line downloader for each.

Datasets covered:
  * Human3.6M        -- HuggingFace mirror (ryushinn/Human3.6M)
  * MPI-INF-3DHP     -- MPI-INF starter zip + per-subject archives
  * 3DPW             -- MPI-INF direct zips (accept license first)
  * AIST++           -- GitHub release annotation zips + official video downloader
  * CMU Panoptic     -- CMU/SNU per-sequence scripts + direct sample files
  * SURREAL          -- INRIA tarball / official download script (credentials)

Important findings / caveats:
  * Human3.6M has no official anonymous direct-download; registration at
    vision.imar.ro is required. The HuggingFace mirror is used as the only
    scriptable source here.
  * 3DPW and SURREAL are gated by a license page / credentials; the direct URLs
    below are reachable only after accepting the license (3DPW) or receiving
    login details (SURREAL).
  * MPI-INF-3DHP's top-level zip is only a starter; the real videos/images are
    downloaded by the bundled shell script from gvv.mpi-inf.mpg.de.
  * AIST++ raw videos must be fetched with the official downloader.py from AIST;
    the annotation archives are direct GitHub release downloads.
  * CMU Panoptic is best downloaded per-sequence via getData.sh from the
    panoptic-toolbox repository; a small calibration sample is available via HTTP.

Verified on 2026-08-04: all representative URLs are reachable; small test
fetches for MPI-INF-3DHP, CMU Panoptic (sampleData), and 3DPW succeeded.

Usage examples:
    python experiments/download_webbridge_datasets.py --dataset all --dry-run
    python experiments/download_webbridge_datasets.py --dataset 3dpw --yes
    python experiments/download_webbridge_datasets.py --dataset panoptic --sequence 171204_pose1_sample --yes
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Public dataset sources
# ---------------------------------------------------------------------------

H36M_HOME = "http://vision.imar.ro/human3.6m"
H36M_HF_REPO = "ryushinn/Human3.6M"

DATASET_URLS = {
    "h36m": {
        "homepage": f"{H36M_HOME}/eula.php",
        "hf_repo": H36M_HF_REPO,
        "notes": (
            "Official H36M requires registration.  HF mirror contains the "
            "original image/annotation splits as parquet files."
        ),
    },
    "mpiinf3dhp": {
        "homepage": "https://vcai.mpi-inf.mpg.de/3dhp-dataset/",
        "starter_zip": "https://vcai.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp.zip",
        "base_path": "http://gvv.mpi-inf.mpg.de/3dhp-dataset",
        "notes": (
            "Top-level zip is only the starter kit; real videos are pulled "
            "by the bundled get_dataset.sh."
        ),
    },
    "3dpw": {
        "homepage": "https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html",
        "readme": "https://virtualhumans.mpi-inf.mpg.de/3DPW/readme_and_demo.zip",
        "sequence_files": "https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip",
        "image_files": "https://virtualhumans.mpi-inf.mpg.de/3DPW/imageFiles.zip",
        "notes": "License must be accepted on the homepage before downloading.",
    },
    "aistpp": {
        "homepage": "https://google.github.io/aistplusplus_dataset/download.html",
        "annotations": {
            "motions.zip": "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/motions.zip",
            "keypoints2d.zip": "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/keypoints2d.zip",
            "keypoints3d.zip": "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/keypoints3d.zip",
            "cameras.zip": "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/cameras.zip",
        },
        "video_downloader": "https://raw.githubusercontent.com/google/aistplusplus_api/main/downloader.py",
        "notes": "Annotations are direct; videos require downloader.py + AIST agreement.",
    },
    "panoptic": {
        "homepage": "http://domedb.perception.cs.cmu.edu/dataset.html",
        "toolbox_script": "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/master/scripts/getData.sh",
        "sample_calib": "http://domedb.perception.cs.cmu.edu/webdata/dataset/sampleData/calibration_sampleData.json",
        "base_url": "http://domedb.perception.cs.cmu.edu/webdata/dataset",
        "notes": "Per-sequence downloads via panoptic-toolbox getData.sh; direct files require a sequence name.",
    },
    "surreal": {
        "homepage": "https://www.di.ens.fr/willow/research/surreal/data/",
        "download_script": "https://raw.githubusercontent.com/gulvarol/surreal/master/download/download_surreal.sh",
        "tarball": "https://lsh.paris.inria.fr/SURREAL/SURREAL_v1.tar.gz",
        "notes": "Tarball is password-protected; fill in credentials from the license request.",
    },
}


def _head_ok(url: str) -> bool:
    """Return True if the URL is reachable (HEAD or GET)."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception:
        # Some servers do not support HEAD; fall back to a tiny GET.
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                _ = r.read(1)
                return r.status == 200
        except Exception:
            return False


def _download(url: str, dest: Path, dry_run: bool = False) -> None:
    """Download *url* to *dest*.  Respects *dry_run*."""
    if dry_run:
        print(f"  [dry-run] would download -> {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}\n       -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"  saved {dest} ({dest.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# Per-dataset downloaders
# ---------------------------------------------------------------------------

def download_human36m(out_dir: Path, dry_run: bool = False) -> None:
    """Download Human3.6M from the HuggingFace mirror.

    Uses huggingface_hub if available; otherwise explains how to install it.
    """
    ds_dir = out_dir / "human36m"
    print("\n[Human3.6M]")
    print(f"  homepage : {DATASET_URLS['h36m']['homepage']}")
    print(f"  HF repo  : {DATASET_URLS['h36m']['hf_repo']}")
    print(f"  notes    : {DATASET_URLS['h36m']['notes']}")

    if dry_run:
        print(f"  [dry-run] would create {ds_dir}")
        return

    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore
    except ImportError:
        print(
            "  SKIP: install huggingface_hub to download Human3.6M:\n"
            "        pip install huggingface_hub"
        )
        return

    ds_dir.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(DATASET_URLS["h36m"]["hf_repo"])
    for fname in files:
        if not fname.startswith("data/"):
            continue
        local = ds_dir / fname
        if local.exists():
            print(f"  exists {local}")
            continue
        print(f"  fetching {fname} ...")
        hf_hub_download(
            repo_id=DATASET_URLS["h36m"]["hf_repo"],
            filename=fname,
            repo_type="dataset",
            local_dir=ds_dir,
        )


def download_mpiinf3dhp(out_dir: Path, dry_run: bool = False) -> None:
    """Download the MPI-INF-3DHP starter kit."""
    ds_dir = out_dir / "mpi_inf_3dhp"
    info = DATASET_URLS["mpiinf3dhp"]
    print("\n[MPI-INF-3DHP]")
    print(f"  homepage    : {info['homepage']}")
    print(f"  starter zip : {info['starter_zip']}")
    print(f"  notes       : {info['notes']}")

    if dry_run:
        print(f"  [dry-run] would download starter zip to {ds_dir}")
        return

    ds_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ds_dir / "mpi_inf_3dhp.zip"
    _download(info["starter_zip"], zip_path, dry_run=False)
    print(f"  extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ds_dir)


def download_3dpw(out_dir: Path, dry_run: bool = False) -> None:
    """Download 3DPW archives."""
    ds_dir = out_dir / "3dpw"
    info = DATASET_URLS["3dpw"]
    print("\n[3DPW]")
    print(f"  homepage      : {info['homepage']}")
    print(f"  readme        : {info['readme']}")
    print(f"  sequenceFiles : {info['sequence_files']}")
    print(f"  imageFiles    : {info['image_files']}")
    print(f"  notes         : {info['notes']}")

    if dry_run:
        print(f"  [dry-run] would download 3DPW zips to {ds_dir}")
        return

    ds_dir.mkdir(parents=True, exist_ok=True)
    _download(info["readme"], ds_dir / "readme_and_demo.zip")
    _download(info["sequence_files"], ds_dir / "sequenceFiles.zip")
    _download(info["image_files"], ds_dir / "imageFiles.zip")


def download_aistpp(out_dir: Path, dry_run: bool = False) -> None:
    """Download AIST++ annotation archives and the video downloader."""
    ds_dir = out_dir / "aistpp"
    info = DATASET_URLS["aistpp"]
    print("\n[AIST++]")
    print(f"  homepage    : {info['homepage']}")
    print(f"  annotations : {list(info['annotations'].keys())}")
    print(f"  notes       : {info['notes']}")

    if dry_run:
        print(f"  [dry-run] would download annotation zips to {ds_dir}")
        return

    ds_dir.mkdir(parents=True, exist_ok=True)
    for name, url in info["annotations"].items():
        _download(url, ds_dir / name)
    _download(info["video_downloader"], ds_dir / "downloader.py")
    print(
        "  AIST++ videos: python aistpp/downloader.py "
        "--download_folder=<dir> --num_processes=5"
    )


def download_panoptic(out_dir: Path, sequence: str = "sampleData", dry_run: bool = False) -> None:
    """Download a CMU Panoptic sequence (or just the sample calibration file)."""
    ds_dir = out_dir / "panoptic"
    info = DATASET_URLS["panoptic"]
    print("\n[CMU Panoptic]")
    print(f"  homepage       : {info['homepage']}")
    print(f"  toolbox script : {info['toolbox_script']}")
    print(f"  sequence       : {sequence}")
    print(f"  notes          : {info['notes']}")

    if dry_run:
        print(f"  [dry-run] would create {ds_dir}")
        return

    ds_dir.mkdir(parents=True, exist_ok=True)
    script_path = ds_dir / "getData.sh"
    _download(info["toolbox_script"], script_path)

    if sequence == "sampleData":
        _download(info["sample_calib"], ds_dir / "calibration_sampleData.json")
    else:
        # Let the official toolbox script do the heavy lifting.
        print(f"  use: cd {ds_dir} && bash getData.sh {sequence}")


def download_surreal(out_dir: Path, dry_run: bool = False) -> None:
    """Stage SURREAL downloader; actual fetch needs credentials."""
    ds_dir = out_dir / "surreal"
    info = DATASET_URLS["surreal"]
    print("\n[SURREAL]")
    print(f"  homepage      : {info['homepage']}")
    print(f"  download script : {info['download_script']}")
    print(f"  tarball       : {info['tarball']}")
    print(f"  notes         : {info['notes']}")

    if dry_run:
        print(f"  [dry-run] would stage SURREAL downloader in {ds_dir}")
        return

    ds_dir.mkdir(parents=True, exist_ok=True)
    _download(info["download_script"], ds_dir / "download_surreal.sh")
    print("  set SURREAL_USER and SURREAL_PASS, then run:")
    print("    bash download_surreal.sh /path/to/data $SURREAL_USER $SURREAL_PASS")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WebBridge dataset sourcing for motionflow-multiview."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "h36m", "mpiinf3dhp", "3dpw", "aistpp", "panoptic", "surreal"],
        help="Which dataset to stage/download (default: all).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/webbridge",
        help="Root directory for downloaded datasets (default: data/webbridge).",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="sampleData",
        help="CMU Panoptic sequence name (default: sampleData).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs and destinations without downloading.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually download files (default is dry-run behavior for safety).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Probe each dataset URL and report reachability.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    dry_run = args.dry_run or not args.yes

    if args.verify:
        print("URL reachability probe:")
        # Flatten a few representative URLs per dataset.
        urls = {
            "h36m_hf": f"https://huggingface.co/datasets/{H36M_HF_REPO}",
            "mpiinf3dhp_starter": DATASET_URLS["mpiinf3dhp"]["starter_zip"],
            "3dpw_readme": DATASET_URLS["3dpw"]["readme"],
            "3dpw_sequences": DATASET_URLS["3dpw"]["sequence_files"],
            "aistpp_keypoints2d": list(DATASET_URLS["aistpp"]["annotations"].values())[1],
            "panoptic_sample_calib": DATASET_URLS["panoptic"]["sample_calib"],
            "surreal_script": DATASET_URLS["surreal"]["download_script"],
        }
        for name, url in urls.items():
            ok = _head_ok(url)
            print(f"  {name:25s} {'OK' if ok else 'FAIL'}  {url}")
        return

    selected = list(DATASET_URLS.keys()) if args.dataset == "all" else [args.dataset]
    print(f"WebBridge out_dir: {out_dir}")
    print(f"mode: {'dry-run' if dry_run else 'download'}")

    dispatch = {
        "h36m": lambda: download_human36m(out_dir, dry_run=dry_run),
        "mpiinf3dhp": lambda: download_mpiinf3dhp(out_dir, dry_run=dry_run),
        "3dpw": lambda: download_3dpw(out_dir, dry_run=dry_run),
        "aistpp": lambda: download_aistpp(out_dir, dry_run=dry_run),
        "panoptic": lambda: download_panoptic(out_dir, sequence=args.sequence, dry_run=dry_run),
        "surreal": lambda: download_surreal(out_dir, dry_run=dry_run),
    }

    for ds in selected:
        try:
            dispatch[ds]()
        except Exception as exc:
            print(f"  ERROR downloading {ds}: {exc}")

    print("\nDone.  Use --yes to fetch files; --dry-run to preview.")


if __name__ == "__main__":
    main()
