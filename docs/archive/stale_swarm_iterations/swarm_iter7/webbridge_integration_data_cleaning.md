# WebBridge Integration & Data Cleaning — H36M S9/S11 Test Fix

## Problem Statement

The unified WebBridge benchmark is the prerequisite for any cross-dataset result
table, but the Human3.6M test-subject files (S9, S11) were only present in the
raw `data/webbridge/h36m` directory in **millimeters**. The meter-scale directory
`data/webbridge/h36m_meters` contained only the training-subject files (S1/S5–S8)
and therefore any WebBridge evaluation that requested H36M test data silently
fell back to the wrong split or the wrong units. This is the most likely cause of
the reported ~101 mm H36M cross-dataset failure, because the mixed-dataset and
benchmark pipelines expect `gt_scale=1.0` with meter units. The immediate,
smallest fix is to generate the canonical meter-scale `.npz` files for S9 and S11
and add a smoke benchmark manifest so the GPU benchmark can run once the RTX 4090
is free.

## Simplest Concrete Next Step

Run the CPU-only meter converter for S9/S11, verify with the WebBridge audit
script, and queue the H36M test smoke benchmark.

## Files Touched / Sketch

- `experiments/convert_h36m_test_subjects_to_meters.py` *(new)* – reads every
  `s_09_acts_XX_multiview.npz` and `s_11_acts_XX_multiview.npz` from
  `data/webbridge/h36m`, divides `joints_3d` and `camera_t` by 1000, and writes
  the corresponding `..._m.npz` files to `data/webbridge/h36m_meters`.

- `configs/benchmark_webbridge_h36m_test_smoke.yaml` *(new)* – 30-entry manifest
  pointing at the newly converted S9/S11 meter files for the best PP model.

- `scripts/run_webbridge_h36m_test_smoke.sh` *(new)* – GPU skeleton that runs
  `experiments/run_webbridge_benchmark.py` against the manifest and then calls
  `experiments/summarize_webbridge_benchmark.py`.

- `docs/swarm_iter7/webbridge_integration_data_cleaning.md` – this report.

### Core converter logic

```python
# experiments/convert_h36m_test_subjects_to_meters.py
import argparse
from pathlib import Path
import numpy as np

def convert_file(src: Path, dst: Path, scale: float = 1000.0):
    data = dict(np.load(src))
    data["camera_t"] = data["camera_t"] / scale
    data["joints_3d"] = data["joints_3d"] / scale
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **data)

subjects = ["s_09", "s_11"]
for src in sorted(Path("data/webbridge/h36m").glob("s_09_acts_*.npz")):
    convert_file(src, Path("data/webbridge/h36m_meters") / f"{src.stem}_m.npz")
```

No existing experiment runners were modified; the converter only writes new
files under `data/webbridge/h36m_meters`.

## CPU-Only Script Run

Command:

```bash
python experiments/convert_h36m_test_subjects_to_meters.py
python experiments/audit_webbridge_npz.py \
    --root data/webbridge/h36m_meters \
    --report docs/swarm_iter_next/webbridge_h36m_meters_quality_report.md
```

Result:

- Converted **30** new meter-scale files for H36M test subjects S9 and S11.
- Audit of `data/webbridge/h36m_meters` reports **45 / 45 files OK**,
  including the 30 new S9/S11 files, all canonical `(T, 4, 17, 2)`
  points, `(T, 4, 17)` confidences, and `(T, 17, 3)` joints with no NaN/Inf.
- Camera translation norms are now in meters (e.g. ~5.5 m instead of ~5500 mm).

## GPU Skeleton / How to Run Later

```bash
# Only after the cross-view PP curriculum training has finished.
bash scripts/run_webbridge_h36m_test_smoke.sh
```

This will evaluate the best principal-point checkpoint on all 30 S9/S11 action
files and produce a summary Markdown table.

## Expected Success Metric

- **CPU (done):** all 30 S9/S11 meter files pass the WebBridge audit.
- **GPU (pending):** H36M S9/S11 test MPJPE from the smoke benchmark should drop
  well below the previous ~101 mm cross-dataset failure, with a reasonable
  target < 50 mm on this 4-view, 17-joint test set (PA-MPJPE and per-action
  breakdown will be included in the summary table).

## Resource Requirement

- Meter conversion + audit: **CPU-only**, safe to run while the RTX 4090 trains.
- Benchmark smoke: **GPU**, queued as a shell skeleton only; not launched.

## Repo State

The new converter, benchmark manifest, launcher skeleton, and this report are
committed to branch `multiview-residual-exploration`.
