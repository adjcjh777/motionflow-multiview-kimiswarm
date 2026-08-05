# Evaluation protocol, metrics & reproducibility — next step

## Problem statement

`motionflow_mv/eval/benchmark_protocol.py` and `motionflow_mv/eval/metrics.py` already implement the core metrics (MPJPE, PA-MPJPE, root-relative MPJPE, velocity MPJPE, bone-length error, PCK, AUC) and a `BenchmarkProtocol` class. The remaining gap for a publishable result table is the *reproducibility harness*: when we run 3–5 seeds, the current launcher only writes one aggregate `manifest.json`. Downstream paper scripts need a **per-seed manifest** so that every checkpoint can be traced back to its exact command, seed, and status. The next minimal step is therefore to make the protocol write one manifest per seed and add a thin CLI wrapper that uses it.

## Simplest concrete next change

1. Extend `BenchmarkProtocol.run_multi_seed` to emit `manifest_seed{seed}.json` alongside the aggregate `manifest.json`.
2. Add `experiments/run_repeated_seeds_benchmark.py`, a CLI around `BenchmarkProtocol.run_multi_seed` with a `--dry_run` mode.
3. Smoke-test the launcher with a tiny CPU-only dummy training script to verify per-seed manifests and the aggregate manifest are produced.

This is **CPU-only and safe to run now**; no GPU training is started.

## Files to touch / sketch

### `motionflow_mv/eval/benchmark_protocol.py`

Inside `run_multi_seed`, after each seed entry is built, write a per-seed manifest:

```python
# In BenchmarkProtocol.run_multi_seed(...)
for seed in seeds:
    checkpoint = out_dir / f"{base_name}_seed{seed}.pth"
    entry = self.train(
        script=script,
        base_args=base_args,
        seed=seed,
        output=checkpoint,
        dry_run=dry_run,
    )
    manifest["checkpoints"][str(seed)] = entry

    # Per-seed manifest for reproducibility and external tooling.
    seed_manifest = {
        "seed": seed,
        "base_name": base_name,
        "checkpoint": str(checkpoint),
        "status": entry["status"],
        "command": entry["command"],
    }
    seed_manifest_path = out_dir / f"manifest_seed{seed}.json"
    with open(seed_manifest_path, "w") as f:
        json.dump(seed_manifest, f, indent=2)

# Aggregate manifest (existing behavior)
manifest_path = out_dir / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
```

### `experiments/run_repeated_seeds_benchmark.py` (new)

```python
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from motionflow_mv.eval import BenchmarkConfig, BenchmarkProtocol

parser = argparse.ArgumentParser()
parser.add_argument("--script", required=True)
parser.add_argument("--base_args", required=True)
parser.add_argument("--seeds", type=int, nargs="+", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--base_name", required=True)
parser.add_argument("--dataset", default="mpiinf3dhp")
parser.add_argument("--split", default="test")
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

cfg = BenchmarkConfig(dataset=args.dataset, split=args.split, seed=args.seeds[0])
protocol = BenchmarkProtocol(cfg)
manifest = protocol.run_multi_seed(
    script=args.script,
    base_args=args.base_args,
    seeds=args.seeds,
    out_dir=args.out_dir,
    base_name=args.base_name,
    dry_run=args.dry_run,
)
print(json.dumps(manifest, indent=2))
```

## Command run (CPU-only)

A tiny dummy training script `tmp/dummy_train_for_manifest.py` was created for smoke testing. It accepts `--output` and `--seed` and writes a checkpoint file.

Run 3 seeds for real (CPU only):

```bash
python experiments/run_repeated_seeds_benchmark.py \
    --script tmp/dummy_train_for_manifest.py \
    --base_args "" \
    --seeds 42 43 44 \
    --out_dir tmp/repeated_seeds_smoke \
    --base_name pp_smoke \
    --dataset mpiinf3dhp \
    --split test
```

Result:

```
[dummy train] seed=42 -> tmp\repeated_seeds_smoke\pp_smoke_seed42.pth
[dummy train] seed=43 -> tmp\repeated_seeds_smoke\pp_smoke_seed43.pth
[dummy train] seed=44 -> tmp\repeated_seeds_smoke\pp_smoke_seed44.pth

Aggregate manifest: tmp\repeated_seeds_smoke\manifest.json
Per-seed manifests written to: tmp\repeated_seeds_smoke
```

Directory contents:

```
manifest.json
manifest_seed42.json
manifest_seed43.json
manifest_seed44.json
pp_smoke_seed42.pth
pp_smoke_seed43.pth
pp_smoke_seed44.pth
```

Example per-seed manifest (`manifest_seed42.json`):

```json
{
  "seed": 42,
  "base_name": "pp_smoke",
  "checkpoint": "tmp\\repeated_seeds_smoke\\pp_smoke_seed42.pth",
  "status": "completed",
  "command": [
    "D:\\anaconda3\\python.exe",
    "tmp\\dummy_train_for_manifest.py",
    "--seed",
    "42",
    "--output",
    "tmp\\repeated_seeds_smoke\\pp_smoke_seed42.pth"
  ]
}
```

Dry-run mode (no training launched) also works:

```bash
python experiments/run_repeated_seeds_benchmark.py \
    --script experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
    --base_args "--epochs 1 --batch_size 2" \
    --seeds 42 43 44 \
    --out_dir tmp/repeated_seeds_dry \
    --base_name pp_dry \
    --dataset mpiinf3dhp \
    --split test \
    --dry_run
```

This produced per-seed manifests with `"status": "dry_run"` and the exact command strings, confirming the launcher can queue GPU jobs without executing them.

## Test verification

```bash
python -m pytest tests/test_benchmark_protocol.py -v
```

Result: 4 passed.

## Expected success metric

- Every multi-seed run now produces `manifest.json` plus `manifest_seed{seed}.json`.
- Existing tests still pass.
- When the GPU queue frees up, the same launcher can be run with `--dry_run` removed on the real training script (e.g. `train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py`) to produce 3–5 reproducible checkpoints and a publishable mean/std metric row.

## Compute requirement

**CPU-only for this step.** The smoke test uses a dummy CPU script; the dry-run mode only writes manifests. No GPU training was started. The actual multi-seed training will require GPU once the current cross-view PP curriculum run finishes and the GPU queue reaches this job.

## Notes / follow-up

- H36M S9/S11 preprocessing and adding MPI S6–S8 to the standard test set are still open. The current data only contains MPI S1–S3 and H36M S1/S5/S6/S9/S11; a separate data-conversion effort is needed before those splits can be finalized.
- The existing `experiments/run_repeated_seeds.py` was left untouched; `experiments/run_repeated_seeds_benchmark.py` is the new `BenchmarkProtocol`-based launcher.
