# WebBridge loader &amp; conversion script

I investigated the WebBridge loader and conversion script topic. As a read-only exploration subagent I do not have file-write tools, so I cannot persist the report to disk, but the full report content is below and can be saved to `docs/swarm_iter7/webbridge_loader_and_conversion_script.md`.

## Short summary

The project already has a fairly complete WebBridge layer: `motionflow_mv/data/webbridge_loader.py` implements canonical `.npz` converters for Human3.6M, Shelf/Campus, Synthetic/AMASS, MPI-INF-3DHP and AIST++, plus stubs for Panoptic/3DPW. However, there is no end-to-end batch converter for the Hugging Face Human3.6M preprocessed archive, which is the main dataset driving the current best numbers. The most actionable next step is to add `experiments/batch_convert_h36m_webbridge.py` that iterates over all subjects/actions in the archive and produces the canonical `.npz` files used by `TemporalClipDataset`.

---

# WebBridge Loader & Conversion Script

## 1. Current state

| Component | Location | What it does |
|---|---|---|
| Canonical loader / converter | `motionflow_mv/data/webbridge_loader.py:1` | Implements `convert_human36m`, `convert_shelf_campus`, `convert_synthetic_amass`, `convert_mpiinf3dhp`, `convert_aistpp`; stubs for `convert_panoptic` and `convert_3dpw`. |
| Public exports | `motionflow_mv/data/__init__.py:1` | Exports `convert_human36m`, `convert_shelf_campus`, `convert_synthetic_amass`, `convert_panoptic`, `convert_3dpw`. |
| Dataset downloader | `experiments/download_webbridge_datasets.py:1` | CLI to stage Human3.6M (HF), MPI-INF-3DHP, 3DPW, AIST++, CMU Panoptic, SURREAL. |
| MPI batch converter | `experiments/batch_convert_mpiinf3dhp_v1.py:1` | Downloads `annot.mat`/`camera.calibration` and produces 14-view + 4-view canonical `.npz` in mm and meters. |
| AIST++ converter | `experiments/convert_aistpp_v1.py:1` | Wraps `convert_aistpp` to write per-sequence `.npz`. |
| Panoptic converter | `experiments/convert_panoptic_v1.py:1` | Self-contained single-sequence downloader + converter (COCO19 skeleton). |
| 3DPW converter | `experiments/convert_3dpw_multiview.py:1` | Produces pseudo multi-view or actual single-view `.npz` from 3DPW `.pkl`. |
| Consumer format | `motionflow_mv/data/temporal_clip_dataset.py:1` | Reads canonical keys `points_2d`, `confidences`, `joints_3d`, `camera_K/R/t` for training. |

The canonical `.npz` schema is stable (`points_2d (T,V,J,2)`, `confidences (T,V,J)`, `joints_3d (T,J,3)`, `camera_K/R (V,3,3)`, `camera_t (V,3)`) and is already consumed by the temporal training scripts.

## 2. Gap / opportunity

The best published numbers (Human3.6M MPJPE 5.74 mm / PA-MPJPE 3.99 mm) come from the Human3.6M multi-view pseudo-GT, yet the H36M conversion path still requires manually calling `convert_human36m` per subject/action. There is no equivalent of `batch_convert_mpiinf3dhp_v1.py` for H36M. Closing this gap would:

* Remove manual iteration over H36M subjects/actions.
* Standardize mm → m conversion for the H36M pipeline.
* Make it trivial to regenerate the full training corpus if the preprocessed archive is updated.

## 3. Concrete next step

Add a new script `experiments/batch_convert_h36m_webbridge.py` that:

1. Reads `data/h36m_hf/h36m_sh_conf_cam_source_final.pkl.zip` and `data/h36m_hf/camera_params.json`.
2. Discovers all `(subject, action, split)` groups in the pickle archive.
3. Calls `motionflow_mv.data.webbridge_loader.convert_human36m` for each group.
4. Writes canonical `.npz` files under `data/webbridge/h36m/<split>/`.
5. Optionally also writes a meters variant by dividing `joints_3d` and `camera_t` by 1000.

A minimal CLI:

```bash
python experiments/batch_convert_h36m_webbridge.py \
    --data_root data/h36m_hf \
    --out_dir data/webbridge/h36m \
    --splits train test
```

## 4. Expected success metric

* The script produces `.npz` files for all requested H36M subject/action groups with the expected shapes.
* A smoke test loads one output with `TemporalClipDataset(..., clip_len=13)` and runs one forward pass of `RayAttentionFusionModelTemporalResidual` without shape errors.
* Enables a full H36M multi-view training run; cross-subject MPJPE target ≤ 6 mm.

## 5. Risks / blockers

* **A800-D and Docker are read-only** — do not modify anything on those systems.
* **Large files** — H36M data must be downloaded on demand; never commit the raw archive or generated `.npz` outputs to Git.
* **Windows NumPy BLAS crash** — if `np.linalg` operations fail in this environment, fall back to the torch-based DLT helper already in `motionflow_mv/fusion/triangulation.py`.
* **Skeleton convention** — the preprocessed archive uses 17 joints; verify that the generated `.npz` matches the joint ordering expected by the temporal training scripts.