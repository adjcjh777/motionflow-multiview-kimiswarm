## Summary

I investigated the project’s dataset provenance and licensing status. The repo already ingests five public sources under `data/webbridge/`, has a downloader script, and a canonical loader, but there is **no consolidated license/attribution manifest** and the paper draft lacks a datasets & licenses section. Because this subagent is read-only and has no file-write tool, I could not persist the report to disk; the full markdown is below and should be saved to:

`docs/swarm_iter7/webbridge_dataset_source_license.md`

---

```markdown
# WebBridge Dataset Source & License

## 1. Current state

The WebBridge ingestion layer is already functional:

| Dataset | Local path | Source script / file | Status in repo |
|---|---|---|---|
| MPI-INF-3DHP | `data/webbridge/mpi_inf_3dhp/*.npz` | `experiments/batch_convert_mpiinf3dhp_v1.py` | Converted; main benchmark |
| AIST++ | `data/webbridge/aistpp_canonical/*.npz` | `experiments/convert_aistpp_v1.py`, `scripts/download_aistpp.py` | Converted |
| 3DPW | `data/webbridge/3dpw/` | `experiments/download_webbridge_datasets.py` | Zips present, pseudo-multi-view converted |
| CMU Panoptic | `data/webbridge/panoptic/171204_pose1_sample/` | `experiments/convert_panoptic_v1.py` | Sample sequence only |
| Shelf / Campus | `data/webbridge/shelf_campus/*.npz` | `motionflow_mv/data/webbridge_loader.py` | Converted; used for zero-shot eval |

Relevant code:
- `experiments/download_webbridge_datasets.py:38–104` — URLs and license notes for six datasets.
- `motionflow_mv/data/webbridge_loader.py:1–30` — canonical `.npz` format and supported sources.
- `docs/paper_draft_icra_cvpr_2027.md:62–66` — datasets and metrics section, **no license/attribution subsection**.
- `docs/swarm_iter5/webbridge_datasets.md` and `docs/swarm_iter3/webbridge_dataset_search.md` — prior surveys, but no machine-readable manifest.

What is missing:
- A single `dataset_manifest.yaml` (or `.md`) with official URLs, license terms, registration requirements, citations, and checksums.
- A license section in the paper draft.
- Automated verification that every downloaded archive matches a known checksum.

## 2. Gap / opportunity

A CVPR/ICRA 2027 submission must clearly state dataset provenance and comply with research-only licenses. Human3.6M, MPI-INF-3DHP, 3DPW, CMU Panoptic, and Shelf/Campus all have usage restrictions; the current repo documents them only in scattered markdown notes. Consolidating this into a manifest:

- Reduces reviewer risk.
- Makes it easy to re-download or audit any dataset.
- Provides a place to record SHA-256 checksums and accepted license URLs.

## 3. Concrete next step

Add a `docs/dataset_manifest.yaml` and a small audit script.

`docs/dataset_manifest.yaml` example:

```yaml
datasets:
  mpi_inf_3dhp:
    official_url: https://vcai.mpi-inf.mpg.de/3dhp-dataset/
    starter_zip: https://vcai.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp.zip
    license: research-only, cite Mehta et al. 3DV 2017
    citation: |
      Mehta et al., "Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision," 3DV 2017.
    notes: Starter zip pulls per-subject archives via bundled get_dataset.sh.

  human3.6m:
    official_url: http://vision.imar.ro/human3.6m/
    hf_mirror: ryushinn/Human3.6M
    license: academic EULA, registration required
    citation: Ionescu et al., TPAMI 2014.

  3dpw:
    official_url: https://virtualhumans.mpi-inf.mpg.de/3DPW/
    license: research agreement, must accept on homepage
    citation: von Marcard et al., ECCV 2018.

  aistpp:
    official_url: https://google.github.io/aistplusplus_dataset/download.html
    license: AIST Dance DB agreement + CC-BY-like research terms
    citation: Li et al., CVPR 2021.

  cmu_panoptic:
    official_url: http://domedb.perception.cs.cmu.edu/
    license: research-only
    citation: Joo et al., TPAMI 2017.

  shelf_campus:
    official_url: http://campar.in.tum.de/Chair/CampusShelfDataset
    license: research benchmark
    citation: Belagiannis et al., CVPR 2014.
```

Then create `experiments/audit_dataset_licenses.py` that:
1. Loads `docs/dataset_manifest.yaml`.
2. For each local `data/webbridge/<dataset>/` directory, prints the dataset name, expected official URL, and license summary.
3. Warns if any required license is not explicitly marked as accepted.

Optionally, add a “Datasets and licenses” paragraph to `docs/paper_draft_icra_cvpr_2027.md` right after §4.1.

## 4. Expected success metric

- `docs/dataset_manifest.yaml` exists and is parseable.
- `python experiments/audit_dataset_licenses.py --check` exits with no warnings for all datasets present in `data/webbridge/`.
- The paper draft contains a license/acknowledgment paragraph listing every dataset, its official URL, and its citation.

## 5. Risks / blockers

- **License gating:** Human3.6M, 3DPW, and CMU Panoptic require explicit registration or license acceptance; the manifest cannot bypass this.
- **Large files:** `data/webbridge/` contains multi-GB archives and `.npz` files. The repo already ignores `data/` in `.gitignore`; do not commit archives.
- **A800-D / Docker read-only:** Do not modify anything outside the working directory or inside read-only mounts. The manifest and audit script should be stored only in `docs/` and `experiments/`.
- **No official license text:** The repo does not host original dataset EULAs; the manifest should link to official pages, not reproduce legal text.
```