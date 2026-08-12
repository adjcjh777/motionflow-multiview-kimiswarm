#!/usr/bin/env python3
"""Watch A800 v25 true-GT stability run, then update docs with S9/S11 test results."""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

A800_HOST = "a800-D"
A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
A800_JSON = f"{A800_REPO}/outputs/eval_v25_true_gt_stability_h36m_test.json"
A800_LOG = f"{A800_REPO}/outputs/ablations/v25_true_gt_stability_a800.log"

LOCAL_REPO = Path(__file__).resolve().parents[1]
LOCAL_DOCS = LOCAL_REPO / "docs" / "results_true_gt_h36m.md"

SLEEP_SECONDS = 300  # 5 minutes
MAX_WAIT_SECONDS = 72 * 3600  # 72 hours


def a800_run(cmd: str, check: bool = True) -> str:
    full = f"ssh {A800_HOST} {cmd!r}"
    result = subprocess.run(full, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"A800 command failed: {cmd}\n{result.stderr}")
    return result.stdout


def json_exists() -> bool:
    out = a800_run(f"test -f {A800_JSON} && echo yes || echo no", check=False)
    return out.strip() == "yes"


def fetch_json() -> dict:
    tmp = LOCAL_REPO / "tmp" / "eval_v25_true_gt_stability_h36m_test.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["scp", f"{A800_HOST}:{A800_JSON}", str(tmp)],
        check=True,
    )
    with tmp.open("r") as f:
        return json.load(f)


def parse_results(data: dict):
    """Extract S9/S11/combined MPJPE and PA-MPJPE from the eval JSON."""
    # The v46 test script writes keys like s9_mpjpe_direct, s11_mpjpe_direct, combined_mpjpe_direct,
    # and pa_mpjpe.
    s9 = data.get("s9_mpjpe_direct")
    s11 = data.get("s11_mpjpe_direct")
    combined = data.get("combined_mpjpe_direct")
    pa = data.get("pa_mpjpe")

    # Fallback: try common alternative keys
    if combined is None:
        combined = data.get("combined")
    if s9 is None:
        s9 = data.get("s9")
    if s11 is None:
        s11 = data.get("s11")
    if pa is None:
        pa = data.get("pa_mpjpe_direct")

    return s9, s11, combined, pa


def update_docs(s9, s11, combined, pa):
    text = LOCAL_DOCS.read_text(encoding="utf-8")

    # Update the in-flight table entry
    old_entry = (
        "| `v25_true_gt_stability_a800` | 6 | `configs/splits/h36m_true_gt_standard.yaml` | **running** |"
    )
    new_entry = (
        f"| `v25_true_gt_stability_a800` | 6 | `configs/splits/h36m_true_gt_standard.yaml` | **completed** |"
        f" **{combined:.2f} mm** (test) |"
    )
    if old_entry in text:
        text = text.replace(old_entry, new_entry, 1)
    else:
        # If the table entry was already updated, don't fail; still append summary.
        pass

    # Replace the bullet line about v25 stability
    old_bullet = (
        "- **v25 mixed-dataset** is in its first epoch, so no validation numbers are available yet. "
        "**v25 stability** is now at Epoch 12 and has reached **31.22 mm** validation MPJPE @ Epoch 11 (continuing to improve)."
    )
    new_bullet = (
        "- **v25 stability** finished with a **test MPJPE of "
        f"{combined:.2f} mm** (S9 {s9:.2f} mm / S11 {s11:.2f} mm; PA-MPJPE {pa:.2f} mm). "
        "Source: `outputs/eval_v25_true_gt_stability_h36m_test.json`."
    )
    if old_bullet in text:
        text = text.replace(old_bullet, new_bullet, 1)

    # Add a dedicated v25 stability summary section if not already present
    section_marker = "### v25 true-GT stability (A800 GPU 6)\n"
    if section_marker not in text:
        summary = f"""
{section_marker}
- **Status:** completed.
- **Best val MPJPE:** 31.22 mm @ Epoch 11 (early-stopping patience 3, still improving when last observed).
- **Test MPJPE:** {combined:.2f} mm (S9 {s9:.2f} mm / S11 {s11:.2f} mm, stride 13, PA-MPJPE {pa:.2f} mm).
- **Checkpoint:** `outputs/ablations/v25_true_gt_stability_a800.pth`.
- **Log:** `outputs/ablations/v25_true_gt_stability_a800.log`.
- **Test JSON:** `outputs/eval_v25_true_gt_stability_h36m_test.json`.

"""
        # Insert before "### In-flight A800 ablations"
        insert_at = text.find("### In-flight A800 ablations")
        if insert_at >= 0:
            text = text[:insert_at] + summary + text[insert_at:]

    LOCAL_DOCS.write_text(text, encoding="utf-8")

    # Sync to A800 via a temporary file
    tmp_md = LOCAL_REPO / "tmp" / "results_true_gt_h36m_v25_stability.md"
    tmp_md.write_text(text, encoding="utf-8")
    subprocess.run(
        ["scp", str(tmp_md), f"{A800_HOST}:{A800_REPO}/docs/results_true_gt_h36m.md"],
        check=True,
    )


def main():
    start = time.time()
    print(f"Watching for {A800_JSON} on {A800_HOST}...")

    while time.time() - start < MAX_WAIT_SECONDS:
        if json_exists():
            print("Test eval JSON found. Fetching and updating docs...")
            try:
                data = fetch_json()
                s9, s11, combined, pa = parse_results(data)
                if combined is None:
                    raise ValueError(f"Could not parse combined MPJPE from {data.keys()}")
                update_docs(s9, s11, combined, pa)
                print(f"Docs updated. Test MPJPE: {combined:.2f} mm (S9 {s9:.2f}, S11 {s11:.2f}, PA {pa:.2f}).")
                return 0
            except Exception as e:
                print(f"Error processing results: {e}", file=sys.stderr)
                return 1
        else:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - JSON not yet present, sleeping {SLEEP_SECONDS}s...")
            time.sleep(SLEEP_SECONDS)

    print("Timeout waiting for test eval JSON.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
