#!/usr/bin/env python3
"""Lightweight ablation runner: execute N training/eval variants from one YAML config.

Summary
-------
``run_ablation_variants.py`` lets you describe a *base* command plus an arbitrary
number of *variants* in a single YAML file.  For each variant the runner builds a
CLI command, runs it as a subprocess, captures the log, and writes a JSON summary
with status / return code / elapsed time.

Config schema
-------------
.. code-block:: yaml

    output_dir: outputs/ablations/my_run      # default: outputs/ablations/<timestamp>

    base:
      script: experiments/train_omniview_fusion_v5_webbridge_multi.py
      env:                                    # optional extra env vars
        PYTHONPATH: "."
      args:                                   # mapping flag -> value
        --use_mixed_loader: true
        --mixed_manifest: "configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml"
        --epochs: 2
        --batch_size: 4

    variants:
      - name: v23_kap_no_ba
        args:
          --use_kinematic_anthropometric_prior_v22: true
          --use_neural_bundle_adjustment_v21: false
          --output: outputs/ablations/my_run/v23_kap_no_ba.pth

      - name: v24_kap_fixed_ba
        args:
          --use_kinematic_anthropometric_prior_v22: true
          --use_neural_bundle_adjustment_v21: true
          --output: outputs/ablations/my_run/v24_kap_fixed_ba.pth

Boolean flags
-------------
* ``true``  -> the flag is emitted (``--flag``).
* ``false`` -> the flag is omitted.
* Numeric/string/list values are emitted as ``--flag value``.  Lists are joined
  with a single space, e.g. ``--manifest a.yaml b.yaml``.

Usage
-----
    # Run all variants sequentially
    python scripts/run_ablation_variants.py --config configs/ablations/kap_ba_sweep.yaml

    # Dry-run: print commands without running
    python scripts/run_ablation_variants.py --config configs/ablations/kap_ba_sweep.yaml --dry-run

    # Run up to 3 variants in parallel
    python scripts/run_ablation_variants.py --config configs/ablations/kap_ba_sweep.yaml --max-workers 3

Output
------
For each variant a log file is written to ``<output_dir>/logs/<variant_name>.log``.
A JSON summary is written to ``<output_dir>/ablation_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from exc


REQUIRED_BASE_KEYS = {"script"}
REQUIRED_VARIANT_KEYS = {"name"}


def _load_yaml(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    return yaml.safe_load(raw)


def _format_arg_value(value: Any) -> str:
    """Return a string suitable for the command line."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _build_command(script: str, args: Mapping[str, Any]) -> List[str]:
    """Build a command list from a script path and an flag->value mapping."""
    cmd = [sys.executable, str(script)]
    for flag, value in args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(str(flag))
        else:
            formatted = _format_arg_value(value)
            if formatted:
                cmd.extend([str(flag), formatted])
    return cmd


def _merge_args(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge two argument mappings (override wins at the leaf level)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_args(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_config(cfg: Any) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must contain a top-level dictionary")

    base = cfg.get("base")
    if base is None:
        raise ValueError("Config missing required 'base' section")
    missing = REQUIRED_BASE_KEYS - set(base.keys())
    if missing:
        raise ValueError(f"base missing required keys: {missing}")

    variants = cfg.get("variants")
    if not variants:
        raise ValueError("Config must contain at least one variant under 'variants'")

    names = [v.get("name") for v in variants]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate variant names: {duplicates}")


def _run_variant(
    variant: Dict[str, Any],
    base: Dict[str, Any],
    output_dir: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    """Run a single variant and return a report dictionary."""
    name = variant["name"]
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    base_args = base.get("args", {})
    variant_args = variant.get("args", {})
    merged_args = _merge_args(base_args, variant_args)

    script = base["script"]
    cmd = _build_command(script, merged_args)

    # Allow per-variant env to override base env.
    env = os.environ.copy()
    env.update(base.get("env", {}))
    env.update(variant.get("env", {}))

    report: Dict[str, Any] = {
        "name": name,
        "command": cmd,
        "log_path": str(log_path),
        "status": "pending",
        "returncode": None,
        "elapsed_seconds": 0.0,
    }

    if dry_run:
        report["status"] = "dry_run"
        report["returncode"] = 0
        return report

    start = time.perf_counter()
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"# Variant: {name}\n")
            log_file.write(f"# Command: {' '.join(cmd)}\n")
            log_file.write(f"# Started: {datetime.now().isoformat()}\n\n")
            log_file.flush()

            proc = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            report["returncode"] = proc.returncode
            report["status"] = "completed" if proc.returncode == 0 else "failed"
    except Exception as exc:  # pragma: no cover - should not happen in normal use
        report["status"] = "failed"
        report["error"] = str(exc)
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - start, 2)

    return report


def _run_all(
    variants: Sequence[Dict[str, Any]],
    base: Dict[str, Any],
    output_dir: Path,
    dry_run: bool,
    max_workers: int,
) -> List[Dict[str, Any]]:
    """Run all variants, optionally in parallel."""
    if max_workers <= 1:
        return [_run_variant(v, base, output_dir, dry_run) for v in variants]

    results: List[Optional[Dict[str, Any]]] = [None] * len(variants)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_variant, v, base, output_dir, dry_run): i
            for i, v in enumerate(variants)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # pragma: no cover
                results[idx] = {
                    "name": variants[idx]["name"],
                    "status": "failed",
                    "error": str(exc),
                }
    return [r for r in results if r is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run N ablation variants from a single YAML config",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML ablation config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Run up to N variants in parallel (default 1, sequential)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = _load_yaml(config_path)
    _validate_config(cfg)

    base = cfg["base"]
    variants = cfg["variants"]

    default_output = Path("outputs") / "ablations" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.get("output_dir", str(default_output)))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ablation runner: {len(variants)} variant(s)")
    print(f"Output directory: {output_dir.resolve()}")
    if args.dry_run:
        print("Mode: dry-run (commands printed, not executed)")

    # In dry-run mode, just print commands and write a summary.
    if args.dry_run:
        for v in variants:
            name = v["name"]
            merged = _merge_args(base.get("args", {}), v.get("args", {}))
            cmd = _build_command(base["script"], merged)
            print(f"\n[{name}]")
            print("  ", " ".join(cmd))

    results = _run_all(variants, base, output_dir, args.dry_run, args.max_workers)

    summary = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "is_dry_run": args.dry_run,
        "max_workers": args.max_workers,
        "total_variants": len(variants),
        "completed": sum(1 for r in results if r.get("status") == "completed"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "dry_run": sum(1 for r in results if r.get("status") == "dry_run"),
        "variants": results,
    }

    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {summary_path}")
    print(
        f"Status: {summary['completed']} completed, "
        f"{summary['failed']} failed, "
        f"{summary['dry_run']} dry-run"
    )

    # Exit with non-zero if any real run failed.
    if not args.dry_run and summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
