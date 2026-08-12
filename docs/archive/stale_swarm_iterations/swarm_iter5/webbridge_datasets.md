# WebBridge Dataset Sourcing

**Deliverable:** `experiments/download_webbridge_datasets.py`

## Goal
Collect publicly reachable direct-download and HuggingFace URLs for the six
most relevant multi-view / 3D human pose datasets, and provide a single
minimal script that can stage them for the MotionFlow-MultiView pipeline.

## Datasets and Links

| Dataset | Source Type | URL / Command | Notes |
|---|---|---|---|
| **Human3.6M** | HuggingFace | `https://huggingface.co/datasets/ryushinn/Human3.6M` | Official site requires registration; this is the only scriptable mirror found. |
| **MPI-INF-3DHP** | Direct zip + shell script | `https://vcai.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp.zip` | 11 KB starter kit; real videos are fetched by bundled `get_dataset.sh` from `gvv.mpi-inf.mpg.de`. |
| **3DPW** | Direct zips | `readme_and_demo.zip`, `sequenceFiles.zip`, `imageFiles.zip` from `https://virtualhumans.mpi-inf.mpg.de/3DPW/` | License must be accepted on the homepage before download. |
| **AIST++** | GitHub release zips + downloader | `motions.zip`, `keypoints2d.zip`, `keypoints3d.zip`, `cameras.zip`; videos via `downloader.py` | Annotations are direct; raw videos require AIST agreement. |
| **CMU Panoptic** | Toolbox script + per-sequence files | `https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/master/scripts/getData.sh` | Best downloaded per sequence; `sampleData` calibration is a small direct file. |
| **SURREAL** | Shell script (gated) | `https://raw.githubusercontent.com/gulvarol/surreal/master/download/download_surreal.sh` | 86 GB tarball is password-protected; credentials come from the license request. |

## Key Findings

1. **Human3.6M has no anonymous direct download.** The official portal requires an
   EULA/registration. The HuggingFace dataset `ryushinn/Human3.6M` is currently the
   only programmable source for the raw images and annotations.
2. **MPI-INF-3DHP starter zip is tiny.** It ships shell scripts that fetch per-subject
   archives from `gvv.mpi-inf.mpg.de`.
3. **3DPW links work after the license page is accepted.** `readme_and_demo.zip`
   is only ~4.5 KB and is a good connectivity probe; `sequenceFiles.zip` is
   ~278 MB and `imageFiles.zip` is larger.
4. **AIST++ annotations are direct GitHub release downloads.** Raw videos need the
   official `downloader.py` plus the AIST Dance DB agreement.
5. **CMU Panoptic is sequence-oriented.** The `getData.sh` script from the
   panoptic-toolbox repo is the canonical way to fetch calibration, videos, and
   3D pose `.tar` files.
6. **SURREAL is credential-gated.** The script stages the downloader; the actual
   86 GB `SURREAL_v1.tar.gz` requires HTTP Basic Auth credentials.

## Verification

Run the script in dry-run / probe mode to confirm all links are reachable:

```bash
python experiments/download_webbridge_datasets.py --verify
python experiments/download_webbridge_datasets.py --dataset all --dry-run
```

Verification run on 2026-08-04:

* `--verify`: all representative URLs returned HTTP 200.
* `--dataset mpiinf3dhp --yes`: starter zip downloaded and extracted successfully.
* `--dataset panoptic --sequence sampleData --yes`: `getData.sh` and `calibration_sampleData.json`
  downloaded successfully.
* `--dataset 3dpw --yes`: `readme_and_demo.zip`, `sequenceFiles.zip`, and `imageFiles.zip`
  were fetched successfully from the MPI-INF server.

## Risks / Next Steps

- License acceptance is required for **3DPW** and **SURREAL** before any automated
  download; the script documents this but cannot bypass it.
- **Human3.6M** via HuggingFace is convenient but not the official release;
  verify skeleton conventions and camera parameters separately.
- All datasets need adapters to the MotionFlow `HumanMotionIR` format before
  they can train `ray_attention_v3_model.py`.
