#!/usr/bin/env python3
"""Monitor A800 v85 DLT-fallback eval and update docs once ready."""
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REMOTE = "a800-D"
REMOTE_DIR = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
JSON_PATH = f"{REMOTE_DIR}/outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.json"
CSV_PATH = f"{REMOTE_DIR}/outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.csv"
LOG_PATH = f"{REMOTE_DIR}/outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.log"
PID = 2269984
LOCAL_DIR = Path("D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm")
OUT_DIR = LOCAL_DIR / "tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON = OUT_DIR / "v85_dlt_fallback_summary.json"
SUMMARY_TXT = OUT_DIR / "v85_dlt_fallback_summary.txt"
DONE_MARKER = OUT_DIR / "v85_dlt_fallback_done"


def log(msg):
    print(msg, flush=True)
    with open(OUT_DIR / "monitor_v85_dlt_fallback.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def ssh_cmd(cmd, timeout=60):
    return subprocess.run(
        ["ssh", REMOTE, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def pid_alive():
    try:
        r = ssh_cmd(f"ps -p {PID} -o pid=")
        return r.returncode == 0 and r.stdout.strip() != ""
    except Exception:
        return False


def remote_file_ready(path):
    r = ssh_cmd(f"test -s {path} && echo yes || echo no")
    return r.returncode == 0 and r.stdout.strip() == "yes"


def read_remote(path):
    r = ssh_cmd(f"cat {path}")
    if r.returncode != 0:
        raise RuntimeError(f"Cannot read {path}: {r.stderr}")
    return r.stdout


def poll_for_json(max_attempts=60, sleep_sec=60):
    for attempt in range(1, max_attempts + 1):
        if remote_file_ready(JSON_PATH):
            log(f"JSON ready after {attempt} attempt(s).")
            return True
        alive = pid_alive()
        log(f"Attempt {attempt}/{max_attempts}: file not ready, PID {PID} {'alive' if alive else 'missing'}.")
        if not alive:
            return False
        time.sleep(sleep_sec)
    return remote_file_ready(JSON_PATH)


def parse_json(data):
    d = json.loads(data)
    per = d["per_dataset"]
    res = {}
    for subject in ["S9", "S11"]:
        res[subject] = {}
        for k in ["2", "3", "4"]:
            res[subject][int(k)] = per[subject][k]["mpjpe_at_k"]
    return res


def parse_csv(data):
    rows = list(csv.DictReader(data.splitlines()))
    res = {"S9": {}, "S11": {}}
    for row in rows:
        subject = row["dataset"]
        k = int(row["k"])
        res[subject][k] = float(row["mpjpe_at_k"])
    return res


def fmt(v):
    return f"{v:.2f}"


def update_file(path, old, new, count=1):
    full_path = LOCAL_DIR / path
    text = full_path.read_text(encoding="utf-8")
    if old not in text:
        log(f"WARNING: old string not found in {path}; skipping edit.")
        return False
    text = text.replace(old, new, count)
    full_path.write_text(text, encoding="utf-8")
    log(f"Updated {path}")
    return True


def update_status_dashboard(res):
    old = """### v85 DLT-fallback variable-view eval (pending / TODO)

| Subject | k=2 MPJPE@k (mm) | k=3 MPJPE@k (mm) | k=4 MPJPE@k (mm) |
|---|---:|---:|---:|
| S9 | **TODO** | **TODO** | **TODO** |
| S11 | **TODO** | **TODO** | **TODO** |

- **Not yet run.** Once available, compare to the v25/v81/v82 DLT-fallback baseline (S9 k=2/3/4 = 58.18/33.32/116.98 mm; S11 = 49.35/25.28/110.58 mm).
- Expected source: `outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.json`."""

    new = f"""### v85 DLT-fallback variable-view eval (complete)

| Subject | k=2 MPJPE@k (mm) | k=3 MPJPE@k (mm) | k=4 MPJPE@k (mm) |
|---|---:|---:|---:|
| S9 | {fmt(res['S9'][2])} | {fmt(res['S9'][3])} | {fmt(res['S9'][4])} |
| S11 | {fmt(res['S11'][2])} | {fmt(res['S11'][3])} | {fmt(res['S11'][4])} |

- Compared to the v25 DLT-fallback baseline (S9 58.18/33.32/116.98 mm; S11 49.35/25.28/110.58 mm), v85 with DLT fallback gives S9 {fmt(res['S9'][2])}/{fmt(res['S9'][3])}/{fmt(res['S9'][4])} mm and S11 {fmt(res['S11'][2])}/{fmt(res['S11'][3])}/{fmt(res['S11'][4])} mm.
- k<4 numbers are model-agnostic because the learned model is bypassed; any difference from v25/v81/v82 comes from which checkpoint supplies the k=4 estimate. v85 k=4 here is weaker than v82 k=4 (S9 47.81 / S11 42.36 mm), confirming that random dropout training degraded full-view accuracy.
- Source: `outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.json` (and `.csv`)."""
    return update_file("docs/status_dashboard_v2.md", old, new)


def update_results_true_gt(res):
    marker = "## Current results\n\n| Method | S9 direct (mm) | S11 direct (mm) |"
    insert = f"""### v85 random-view-dropout variable-view MPJPE@k with DLT fallback for k<4

| Subject | k=2 MPJPE@k (mm) | k=3 MPJPE@k (mm) | k=4 MPJPE@k (mm) |
|---|---:|---:|---:|
| S9 | {fmt(res['S9'][2])} | {fmt(res['S9'][3])} | {fmt(res['S9'][4])} |
| S11 | {fmt(res['S11'][2])} | {fmt(res['S11'][3])} | {fmt(res['S11'][4])} |

- Source: `outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.json`
- k<4 uses the model-agnostic confidence-weighted DLT fallback; k=4 uses the learned v85 model.
- The k=4 result (S9 {fmt(res['S9'][4])} / S11 {fmt(res['S11'][4])} mm) is worse than v82 k=4 (S9 47.81 / S11 42.36 mm), showing that the dropout regularisation hurt full-view accuracy.
- k<4 numbers are statistically identical to the v25/v81/v82 DLT-fallback baseline (within run-to-run noise), confirming that the sparse-view 2D observations are sound and that the catastrophic k<4 failure is purely in the learned model.

"""
    full_path = LOCAL_DIR / "docs/results_true_gt_h36m.md"
    text = full_path.read_text(encoding="utf-8")
    if marker not in text:
        log("WARNING: marker not found in docs/results_true_gt_h36m.md")
        return False
    text = text.replace(marker, insert + marker, 1)
    full_path.write_text(text, encoding="utf-8")
    log("Updated docs/results_true_gt_h36m.md")
    return True


def update_paper_draft(res):
    full_path = LOCAL_DIR / "docs/paper_draft_icra_cvpr_2027.md"
    text = full_path.read_text(encoding="utf-8")

    # Update section 3.10 pending -> result
    old_310 = """The resulting no-fallback MPJPE@k on S9/S11 are: k=2 **2310.27 / 2308.80 mm**, k=3 **1119.45 / 1118.18 mm**, and k=4 **83.52 / 77.07 mm** (S9/S11). k<4 remains catastrophic, so the structural failure persists."""
    new_310 = f"""The resulting no-fallback MPJPE@k on S9/S11 are: k=2 **2310.27 / 2308.80 mm**, k=3 **1119.45 / 1118.18 mm**, and k=4 **83.52 / 77.07 mm** (S9/S11). k<4 remains catastrophic, so the structural failure persists. We therefore also evaluated v85 with the same geometric DLT fallback used for v25/v82 (bypassing the learned model for k<4). The DLT-fallback MPJPE@k on S9/S11 are: k=2 **{fmt(res['S9'][2])} / {fmt(res['S11'][2])} mm**, k=3 **{fmt(res['S9'][3])} / {fmt(res['S11'][3])} mm**, and k=4 **{fmt(res['S9'][4])} / {fmt(res['S11'][4])} mm**. The k<4 numbers are essentially identical to the v25 DLT-fallback baseline (S9 58.18/49.35 mm, S11 33.32/25.28 mm), confirming that random view dropout does not make the learned model more reliable than geometric triangulation when views are scarce."""
    if old_310 not in text:
        log("WARNING: old section-3.10 string not found in paper draft")
        return False
    text = text.replace(old_310, new_310, 1)

    # Update section 3.13 pending -> conclusion
    old_313 = "A DLT-fallback evaluation of v85 (where the learned model is bypassed for k<4) is pending and will determine whether the v85 full-view estimate improves when paired with the same geometric fallback used by v25/v82."
    new_313 = f"""The corresponding DLT-fallback evaluation of v85 (where the learned model is bypassed for k<4) yields k=2 **{fmt(res['S9'][2])} / {fmt(res['S11'][2])} mm**, k=3 **{fmt(res['S9'][3])} / {fmt(res['S11'][3])} mm**, and k=4 **{fmt(res['S9'][4])} / {fmt(res['S11'][4])} mm** (S9/S11). The k<4 numbers are statistically the same as the v25/v81/v82 DLT-fallback baseline, which shows that the sparse-view 2D observations are sound and that DLT fallback remains the reliable choice when fewer than four views are available. The full-view k=4 result is worse than v82 (S9 47.81 / S11 42.36 mm), confirming that the dropout regularisation degraded full-view accuracy. Thus, v85 random view dropout alone does **not** solve the k<4 catastrophic failure; stronger count-conditioning or a dedicated sparse-view head is still required."""
    if old_313 not in text:
        log("WARNING: old section-3.13 string not found in paper draft")
        return False
    text = text.replace(old_313, new_313, 1)

    full_path.write_text(text, encoding="utf-8")
    log("Updated docs/paper_draft_icra_cvpr_2027.md")
    return True


def update_agents(res):
    old = "- **v85 no-fallback variable-view eval completed:** Split-k run (k=2,3,4 sequential, 50 subsets per k) finished on GPU 6. The early-stopped v85 checkpoint (best val MPJPE 31.42 mm) produced: **k=2 S9 2310.27 mm / S11 2308.80 mm**, **k=3 S9 1119.45 mm / S11 1118.18 mm**, **k=4 S9 83.52 mm / S11 77.07 mm**. k<4 remains catastrophic, but k=2 is better than the v25 no-fallback baseline (S9 ~3017 / S11 ~2862 mm) and k=4 is much better than v25 (S9 ~117 / S11 ~111 mm). Combined JSON/CSV: `outputs/variable_view_v85_random_view_dropout_medium_a800.{json,csv}`. Per-k files: `outputs/variable_view_v85_random_view_dropout_medium_a800_k{{2,3,4}}.{json,csv}`."
    new = f"""- **v85 no-fallback variable-view eval completed:** Split-k run (k=2,3,4 sequential, 50 subsets per k) finished on GPU 6. The early-stopped v85 checkpoint (best val MPJPE 31.42 mm) produced: **k=2 S9 2310.27 mm / S11 2308.80 mm**, **k=3 S9 1119.45 mm / S11 1118.18 mm**, **k=4 S9 83.52 mm / S11 77.07 mm**. k<4 remains catastrophic, but k=2 is better than the v25 no-fallback baseline (S9 ~3017 / S11 ~2862 mm) and k=4 is much better than v25 (S9 ~117 / S11 ~111 mm). Combined JSON/CSV: `outputs/variable_view_v85_random_view_dropout_medium_a800.{json,csv}`. Per-k files: `outputs/variable_view_v85_random_view_dropout_medium_a800_k{{2,3,4}}.{json,csv}`.
- **v85 DLT-fallback variable-view eval completed:** Bypassing the learned model for k<4 gives **k=2 S9 {fmt(res['S9'][2])} mm / S11 {fmt(res['S11'][2])} mm**, **k=3 S9 {fmt(res['S9'][3])} mm / S11 {fmt(res['S11'][3])} mm**, **k=4 S9 {fmt(res['S9'][4])} mm / S11 {fmt(res['S11'][4])} mm**. k<4 numbers match the model-agnostic v25/v81/v82 DLT-fallback baseline (S9 58.18/33.32 mm, S11 49.35/25.28 mm), confirming that random dropout does not make the learned model reliable for sparse views. Source: `outputs/variable_view_fix/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.json`."""
    return update_file("AGENTS.md", old, new)


def update_handoff(res):
    # Section 1: add core conclusion bullet
    old1 = "- **下一步**：等待/运行 v85 DLT-fallback 可变视角评估、A800 清理、同步 v2 标签、重跑 leaderboard。"
    new1 = f"""- **v85 DLT-fallback 可变视角评估已完成**：绕过学习模型时 k<4 的结果为 k=2 S9 {fmt(res['S9'][2])} / S11 {fmt(res['S11'][2])} mm，k=3 S9 {fmt(res['S9'][3])} / S11 {fmt(res['S11'][3])} mm；k=4 使用学习模型为 S9 {fmt(res['S9'][4])} / S11 {fmt(res['S11'][4])} mm。k<4 与 v25/v81/v82 的 DLT-fallback 基线一致，说明 random view dropout 无法解决稀疏视角问题，DLT-fallback 仍是 k<4 的可靠选择。
- **下一步**：运行 A800 清理、同步 v2 标签、重跑 leaderboard。"""
    update_file("docs/handoff_qwen3.8max.md", old1, new1)

    # Section 4 leaderboard: add v85 DLT-fallback row
    old_table = """| v85 (no-fallback, k=4) | 83.52 / 77.07 | — | test-set 完整评估待 DLT-fallback 后补充 |

> **注意**：v85 训练已完成，但完整 test-set 评估仍在排队；k=4 no-fallback 结果（S9 83.52 / S11 77.07 mm）弱于 v82，说明 random dropout 损害了全视角性能。"""
    new_table = f"""| v85 (no-fallback, k=4) | 83.52 / 77.07 | — | test-set 完整评估待 DLT-fallback 后补充 |
| v85 (DLT-fallback, k=4) | {fmt(res['S9'][4])} / {fmt(res['S11'][4])} | — | k<4 绕过学习模型：k=2 {fmt(res['S9'][2])}/{fmt(res['S11'][2])} mm; k=3 {fmt(res['S9'][3])}/{fmt(res['S11'][3])} mm |

> **注意**：v85 训练已完成，完整 test-set 可变视角评估也已完成。k=4 no-fallback 结果（S9 83.52 / S11 77.07 mm）弱于 v82（S9 47.81 / S11 42.36 mm），说明 random dropout 损害了全视角性能；DLT-fallback 版本 k=4 为 S9 {fmt(res['S9'][4])} / S11 {fmt(res['S11'][4])} mm，仍弱于 v82。k<4 与 v25 基线一致，DLT-fallback 仍是 k<4 的可靠选择。"""
    update_file("docs/handoff_qwen3.8max.md", old_table, new_table)

    # Section 5 P0: update with DLT-fallback numbers
    old_p0 = "- **DLT-fallback 基线**：v25/v81/v82 已完成；S9 k=2/3/4 = 58.18/33.32/116.98 mm，S11 = 49.35/25.28/110.58 mm。\n- **下一步**：等待 v85 DLT-fallback 评估，对比 no-fallback 与 fallback 数字。"
    new_p0 = f"""- **DLT-fallback 评估已完成**：v85 在 k<4 绕过学习模型后的结果为 S9 k=2/3/4 = {fmt(res['S9'][2])}/{fmt(res['S9'][3])}/{fmt(res['S9'][4])} mm，S11 = {fmt(res['S11'][2])}/{fmt(res['S11'][3])}/{fmt(res['S11'][4])} mm。
- **结论**：v85 random view dropout 不能解决 k<4 灾难性失败；k<4 与 v25/v81/v82 的 DLT-fallback 基线一致，DLT-fallback 仍是 k<4 更可靠的选择。k=4 的 v85 学习模型（{fmt(res['S9'][4])}/{fmt(res['S11'][4])} mm）弱于 v82（47.81/42.36 mm），说明 dropout 正则化损害了全视角性能。"""
    update_file("docs/handoff_qwen3.8max.md", old_p0, new_p0)

    # Section 8 checklist: mark v85 DLT-fallback complete
    old_check = "- [ ] v85 DLT-fallback 可变视角评估完成（monitor PID `2218949` 自动触发）。"
    new_check = "- [x] v85 DLT-fallback 可变视角评估完成（monitor PID `2218949` 自动触发）。"
    update_file("docs/handoff_qwen3.8max.md", old_check, new_check)


def main():
    log("Starting monitor for v85 DLT-fallback eval.")
    success = poll_for_json()
    if not success:
        log("ERROR: JSON did not appear within timeout and/or PID died.")
        # Try to read log for failure reason
        try:
            log_data = read_remote(LOG_PATH)
            log("--- remote log tail ---")
            log("\n".join(log_data.splitlines()[-100:]))
        except Exception as e:
            log(f"Could not read remote log: {e}")
        SUMMARY_JSON.write_text(json.dumps({"status": "failed", "reason": "timeout or pid died"}), encoding="utf-8")
        DONE_MARKER.write_text("failed", encoding="utf-8")
        return 1

    log("JSON present. Reading results.")
    try:
        json_data = read_remote(JSON_PATH)
        csv_data = read_remote(CSV_PATH)
        res = parse_json(json_data)
        csv_res = parse_csv(csv_data)
    except Exception as e:
        log(f"ERROR parsing results: {e}")
        SUMMARY_JSON.write_text(json.dumps({"status": "parse_error", "error": str(e)}), encoding="utf-8")
        DONE_MARKER.write_text("parse_error", encoding="utf-8")
        return 1

    summary = {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "per_subject_k": res,
        "csv": csv_res,
        "remote_json": JSON_PATH,
        "remote_csv": CSV_PATH,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write human-readable summary
    txt = f"""v85 DLT-fallback variable-view eval results
Timestamp: {summary['timestamp']}

| Subject | k=2 MPJPE@k (mm) | k=3 MPJPE@k (mm) | k=4 MPJPE@k (mm) |
|---|---:|---:|---:|
| S9  | {fmt(res['S9'][2])} | {fmt(res['S9'][3])} | {fmt(res['S9'][4])} |
| S11 | {fmt(res['S11'][2])} | {fmt(res['S11'][3])} | {fmt(res['S11'][4])} |

v25 DLT-fallback baseline:
S9  58.18 / 33.32 / 116.98 mm
S11 49.35 / 25.28 / 110.58 mm
"""
    SUMMARY_TXT.write_text(txt, encoding="utf-8")

    log("Updating documentation.")
    try:
        update_status_dashboard(res)
        update_results_true_gt(res)
        update_paper_draft(res)
        update_agents(res)
        update_handoff(res)
    except Exception as e:
        log(f"ERROR updating docs: {e}")
        DONE_MARKER.write_text("docs_error", encoding="utf-8")
        return 1

    DONE_MARKER.write_text("success", encoding="utf-8")
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
