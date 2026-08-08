#!/usr/bin/env python
"""Print a concise status summary of local and A800 runs."""
import json
import re
from pathlib import Path


def local_log_status(log_path: Path) -> str:
    if not log_path.exists():
        return "not started"
    text = log_path.read_text(errors="ignore")
    steps = re.findall(r"train step (\d+): loss=([\d.]+)", text)
    vals = re.findall(r"Epoch\s+(\d+):\s+train_loss=[\d.]+,\s+val_loss=[\d.]+,\s+val_MPJPE=([\d.]+)mm", text)
    best = min(vals, key=lambda x: float(x[1])) if vals else None
    last_step = steps[-1] if steps else None
    last_loss = steps[-1][1] if steps else None
    if best:
        return f"epoch={best[0]} val={best[1]}mm | last step {last_step} loss={last_loss}"
    return f"training step {last_step} loss={last_loss}" if last_step else "initializing"


def v25_local_status() -> str:
    status_path = Path("outputs/v25_local_4090_status.json")
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
            best = status.get("best")
            last = status.get("last")
            if best and last:
                return f"status={status['status']} best epoch={best['epoch']} val={best['val_mpjpe_mm']}mm | last epoch={last['epoch']} val={last['val_mpjpe_mm']}mm"
            if best:
                return f"best epoch={best['epoch']} val={best['val_mpjpe_mm']}mm"
        except Exception:
            pass
    log = Path("outputs/omniview_fusion_v25_geometry_fusion_small_local_4090.log")
    return local_log_status(log)


def main() -> None:
    print("Local 4090:")
    for variant in ["v26_udp", "v26_udp_gmm", "v26_udp_v28", "v26_udp_gmm_v28", "v29_udp_gmm_v28_graph_outlier"]:
        log = Path(f"outputs/omniview_fusion_{variant}_full_local_4090.log")
        print(f"  {variant}: {local_log_status(log)}")
    print(f"  v25_baseline: {v25_local_status()}")

    print("\nA800 v25 state:")
    state_path = Path(".monitor_a800_v25_state.json")
    if state_path.exists():
        state = json.loads(state_path.read_text())
        for k, v in state.items():
            print(f"  {k}: best {v.get('best_val')} mm @ step {v.get('last_val_step')}")
    else:
        print("  (no state)")

    print("\nA800 v26 state:")
    state_path = Path(".monitor_a800_v26_state.json")
    if state_path.exists():
        state = json.loads(state_path.read_text())
        for k, v in state.items():
            print(f"  {k}: best {v.get('best_val')} mm @ step {v.get('last_val_step')}")
    else:
        print("  (no state)")


if __name__ == "__main__":
    main()
