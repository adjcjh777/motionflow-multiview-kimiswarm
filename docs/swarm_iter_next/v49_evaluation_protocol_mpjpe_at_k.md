# v49: Evaluation Protocol `MPJPE@k`

**Status:** Proposal  
**Labels:** `experiment`, `P1-next`  
**Tracking issue:** #166 (proposed)  
**Depends on:** #160 (v46), #162 (v47), #164 (v48)  

---

## 1. Problem statement

`MPJPE@k` is already the primary metric for v46-v48 (sparse-view generalization, temporal aggregation, and domain generalization), but the evaluation protocol is ad hoc: view subsets are sampled differently across scripts, the handling of `k > V` or `k < min_views` is inconsistent, per-domain reporting is optional, and there is no canonical manifest format. We need a single, reproducible `MPJPE@k` protocol that:

1. Works for any checkpoint from v25 through v48.
2. Reports `MPJPE@k` for `k = 1, 2, 3, 4, full` in a consistent way.
3. Handles per-domain / per-dataset summaries and cross-domain gap.
4. Integrates cleanly with the existing `scripts/run_full_benchmark.py` protocol.
5. Provides feedback to the self-evolution loop (reliability / uncertainty gates).

---

## 2. Proposed approach

### 2.1 Standardize the view-subset sampling

- Enumerate all `C(V, k)` subsets when `V <= 8` and `k <= 4`.
- Otherwise sample `num_subsets_per_k` random subsets with a fixed seed.
- Skip `k < min_views` (default `2` for multi-view, `1` for monocular/3DPW actual).
- For `k = full`, evaluate all available views once.
- For `k = 1`, evaluate every single view independently (monocular baseline / 3DPW actual).

### 2.2 Metric definitions

For each `(dataset, k, subset)` compute:

- `mpjpe_at_k`: per-joint Euclidean error in mm, mean over joints and frames.
- `pa_mpjpe_at_k`: Procrustes-aligned MPJPE.
- `n_mpjpe_at_k`: MPJPE after root-normalization (pelvis at origin).
- `temporal_jerk_at_k`: mean magnitude of 3rd temporal derivative.
- `per_joint_mpjpe_at_k`: `(J,)` array for diagnostic plots.

Aggregate per `(dataset, k)` as the mean over subsets.

### 2.3 Output schema

A single JSON / CSV with:

```json
{
  "model": "outputs/v48_domain/best.pth",
  "dataset": "h36m_val",
  "V": 4,
  "min_views": 2,
  "results": {
    "2": { "mpjpe_at_k": 31.4, "pa_mpjpe_at_k": 28.1, "n_mpjpe_at_k": 22.5,
           "temporal_jerk_at_k": 4.2, "n_subsets": 6 },
    "3": { "mpjpe_at_k": 21.7, ... },
    "4": { "mpjpe_at_k": 18.9, ... },
    "full": { "mpjpe_at_k": 18.9, ... }
  }
}
```

### 2.4 Fit with v46-v48 and the overall pipeline

- **v46:** The protocol is the *official* way to measure sparse-view robustness; it consumes `v46` checkpoints and reports `MPJPE@2/3/4/full`.
- **v47:** Adds `temporal_jerk@k` to verify temporal smoothing does not regress per-frame error.
- **v48:** Adds per-domain `MPJPE@k` and 3DPW actual `MPJPE@1`; the protocol computes the `domain_gap` metric.
- **Paper story:** Produces the variable-view table for the ICRA/CVPR 2027 submission (e.g., Table 3: MPJPE@k across H36M/MPI/3DPW).

### 2.5 Self-evolution feedback loop

`MPJPE@k` per subset exposes which view combinations hurt accuracy. Feed this back into the self-evolution loop:

- Store per-subset `MPJPE@k` in the training log.
- Update the v37/v39 reliability gates with a *view-combination regret* term:
  ```python
  regret_k = max(0, MPJPE(subset) - MPJPE@full)
  reliability_loss += lambda_reliability * regret_k * (1 - r_view)
  ```
- Use the per-domain `MPJPE@k` to update the v41/v48 domain-weighted loss weights (DDWL).

---

## 3. Concrete code-level changes

### 3.1 New / modified files

| File | Change |
|------|--------|
| `motionflow_mv/eval/mpjpe_at_k_protocol.py` | New module: `evaluate_mpjpe_at_k(model, dataset, k_values, ...) -> dict`. Encapsulates subsetting, metric computation, and JSON output. |
| `experiments/eval_variable_views.py` | Refactor to call `evaluate_mpjpe_at_k`; keep CLI backward-compatible. |
| `scripts/run_mpjpe_at_k_benchmark.py` | New driver: takes a YAML manifest like `run_full_benchmark.py` and produces per-dataset `MPJPE@k` tables. |
| `configs/benchmark_v49_mpjpe_at_k_smoke.yaml` | Smoke config: 2-4 views, 20 subsets, synthetic / H36M tiny. |
| `scripts/run_v49_mpjpe_at_k_smoke_local_4090.sh` | Smoke script wrapper. |
| `motionflow_mv/eval/metrics.py` | Add `mpjpe_at_k`, `pa_mpjpe_at_k`, `n_mpjpe_at_k` helpers if not already present. |
| `docs/swarm_iter_next/v49_eval_manifest_example.txt` | Example manifest for H36M/MPI/3DPW actual. |

### 3.2 New flags / config keys

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `eval_mpjpe_at_k_values` | list[int] | `[1, 2, 3, 4, "full"]` | View counts to evaluate. |
| `eval_mpjpe_at_k_num_subsets` | int | `20` | Random subsets per `k` when enumeration is infeasible. |
| `eval_mpjpe_at_k_seed` | int | `42` | RNG seed for subset sampling. |
| `eval_mpjpe_at_k_min_views` | int | `2` | Minimum `k` allowed (set to `1` for 3DPW actual). |
| `eval_mpjpe_at_k_per_domain` | bool | `True` | Report per-domain `MPJPE@k`. |
| `eval_mpjpe_at_k_write_per_joint` | bool | `False` | Include per-joint errors in JSON. |

### 3.3 Interface sketch

```python
# motionflow_mv/eval/mpjpe_at_k_protocol.py

from typing import List, Union

def evaluate_mpjpe_at_k(
    model,
    points_2d: np.ndarray,      # (T, V, J, 2)
    confidences: np.ndarray,    # (T, V, J)
    joints_3d: np.ndarray,      # (T, J, 3)
    cameras: List[Camera],
    k_values: List[Union[int, str]] = [2, 3, 4, "full"],
    num_subsets_per_k: int = 20,
    seed: int = 42,
    min_views: int = 2,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Return canonical MPJPE@k results."""
```

---

## 4. Risks / failure modes

| Risk | Mitigation |
|------|------------|
| `k=1` with multi-view models is undefined / ill-posed | Document it as a monocular fallback; skip if model expects `V >= 2`. |
| Enumerating all subsets for large `V` is slow | Cap enumeration at `V <= 8, k <= 4`; otherwise use `num_subsets_per_k`. |
| Inconsistent alignment between `MPJPE@k` and `MPJPE@full` | Always use the same root-normalization / Procrustes options as `eval_full_metrics.py`. |
| 3DPW actual `MPJPE@1` needs single-view camera trajectories | Reuse `webbridge_3dpw_actual_loader.py` from v48; validate `V=1`. |
| Subset sampling randomness makes comparisons noisy | Fix default seed; report mean and std over subsets. |
| Self-evolution feedback loop destabilizes training | Gate feedback with an exponential moving average; only update every `N` steps. |

---

## 5. Success metrics and recommended experiments

### 5.1 Success criteria

1. `MPJPE@k` table is reproducible across runs (same checkpoint, same seed -> identical results).
2. `MPJPE@full` from the new protocol matches `scripts/run_full_benchmark.py` within 0.1 mm.
3. Per-domain `MPJPE@k` JSON is valid and reports all requested `k` values.
4. 3DPW actual `MPJPE@1` runs without crash.
5. Self-evolution feedback reduces worst-subset `MPJPE@k` by ≥3% relative after one epoch of fine-tuning.

### 5.2 Smoke experiment

| Field | Value |
|-------|-------|
| Hardware | RTX 4090 / CPU |
| Config | `configs/benchmark_v49_mpjpe_at_k_smoke.yaml` |
| Script | `bash scripts/run_v49_mpjpe_at_k_smoke_local_4090.sh` |
| Input | Synthetic 6-view, 17-joint skeleton or a single tiny H36M/MPI `.npz` |
| k values | `2, 3, 4, full` (synthetic); `1, 2, 3, 4, full` if 3DPW actual test file available |
| Expected outcome | JSON produced; `MPJPE@full` finite; no NaN/OOM; runtime < 5 min on CPU |

### 5.3 Full experiment

| Field | Value |
|-------|-------|
| Hardware | A800-D or local RTX 4090 |
| Checkpoints | Best v46/v47/v48 checkpoints |
| Datasets | H36M val, MPI-INF-3DHP val, 3DPW pseudo val, 3DPW actual val |
| k values | `1, 2, 3, 4, full` |
| Output | `outputs/v49_mpjpe_at_k/full_benchmark.json` and `.csv` |
| Expected outcome | Reproduces v46/v47/v48 claims; cross-domain gap ≤ 15 mm at `k=full`; 3DPW actual `MPJPE@1` reported |

---

## 6. Relation to self-evolution feedback loop

This protocol closes the evaluation-to-training loop for v37/v39 reliability gates and v41/v48 DDWL:

1. **Per-subset regret:** For each evaluated subset `S` of size `k`, compute `regret(S) = MPJPE(S) - MPJPE(full)`. Subsets with high regret identify unreliable views.
2. **Reliability update:** Add a term to the v37 view-reliability loss that pushes down reliability for views that appear in high-regret subsets:
   ```python
   reliability_target = sigmoid(-regret(S) / tau)
   ```
3. **Domain reweighting:** Use per-domain `MPJPE@k` to update the v48 DDWL temperature online: increase the weight of domains with higher `MPJPE@k`.
4. **Iteration:** Re-evaluate after each training epoch; use the delta in `MPJPE@k` as the convergence signal for the self-evolution loop.

---

## 7. Next steps

1. Implement `motionflow_mv/eval/mpjpe_at_k_protocol.py` and unit tests.
2. Refactor `experiments/eval_variable_views.py` to use the new protocol while preserving the existing CLI.
3. Add `scripts/run_mpjpe_at_k_benchmark.py` manifest driver.
4. Create `configs/benchmark_v49_mpjpe_at_k_smoke.yaml` and the smoke script.
5. Run smoke on RTX 4090 and verify `MPJPE@full` matches `run_full_benchmark.py`.
6. Run full benchmark on v46/v47/v48 checkpoints and update the paper tables.
