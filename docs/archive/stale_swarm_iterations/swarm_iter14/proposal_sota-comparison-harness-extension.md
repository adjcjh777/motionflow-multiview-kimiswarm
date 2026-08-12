# SOTA Comparison Harness Extension: More Baselines and Metrics

## 1. Problem

The current `experiments/compare_sota_baselines.py` only evaluates plain confidence-weighted DLT, robust IRLS, and the learned PP model on a single MPI-INF-3DHP clean split, so the paper’s SOTA claim is thin and lacks per-joint/per-view error structure, runtime, and cross-dataset validation.

## 2. Hypothesis

Adding two additional strong geometric baselines (RANSAC-DLT and top-2-confidence-view DLT), a richer metrics layer (per-joint, per-view, bone-length, velocity, latency, parameter count), and a tiny reporting module will preserve the anchor’s ~15 mm margin while producing CVPR/ICRA-ready tables in a single CPU/GPU eval run.

## 3. Method

### 3.1 Extend `experiments/compare_sota_baselines.py`

Add the following helpers and wire them into the main results dictionary.

**New baselines**

```python
def ransac_dlt_baseline(points_2d, confidences, K, R, t,
                        subset_size=3, max_iters=100, inlier_thr_px=5.0):
    """Robust DLT: sample camera subsets, pick the subset with lowest reprojection error."""
    B, T, V, J, _ = points_2d.shape
    preds = []
    for b in range(B):
        preds_b = []
        for tt in range(T):
            pred_j = []
            for j in range(J):
                p2d = points_2d[b, tt, :, j, :].cpu().numpy()
                conf = confidences[b, tt, :, j].cpu().numpy()
                K_np = K[b].cpu().numpy()
                R_np = R[b].cpu().numpy()
                t_np = t[b].cpu().numpy()
                P = _projection_matrices(K_np, R_np, t_np)

                best_pred, best_score = None, float("inf")
                rng = np.random.default_rng(42 + j)
                for _ in range(max_iters):
                    idx = rng.choice(V, size=min(subset_size, V), replace=False)
                    pred = triangulate_dlt(p2d[idx], P[idx], weights=conf[idx])
                    # score = mean reprojection error over all views under this 3D point
                    x_pred = (P @ np.concatenate([pred, [1]]))  # (V, 3)
                    x_pred = x_pred[:, :2] / (x_pred[:, 2:3] + 1e-8)
                    err = np.linalg.norm(x_pred - p2d, axis=-1).mean()
                    if err < best_score:
                        best_score = err
                        best_pred = pred
                pred_j.append(best_pred)
            preds_b.append(np.stack(pred_j, axis=0))
        preds.append(np.stack(preds_b, axis=0))
    return np.stack(preds, axis=0)


def top2_dlt_baseline(points_2d, confidences, K, R, t):
    """Triangulate each joint from the two views with highest confidence."""
    B, T, V, J, _ = points_2d.shape
    preds = []
    for b in range(B):
        preds_b = []
        for tt in range(T):
            pred_j = []
            for j in range(J):
                p2d = points_2d[b, tt, :, j, :].cpu().numpy()
                conf = confidences[b, tt, :, j].cpu().numpy()
                K_np = K[b].cpu().numpy()
                R_np = R[b].cpu().numpy()
                t_np = t[b].cpu().numpy()
                top2 = np.argsort(conf)[-2:]
                P = _projection_matrices(K_np, R_np, t_np)
                pred = triangulate_dlt(p2d[top2], P[top2], weights=conf[top2])
                pred_j.append(pred)
            preds_b.append(np.stack(pred_j, axis=0))
        preds.append(np.stack(preds_b, axis=0))
    return np.stack(preds, axis=0)
```

**Richer metrics and latency**

In `main()`:

```python
from motionflow_mv.fusion.graph_joint_relation import MPI_INF_3DHP_28_PARENTS

# After loading model:
n_params = sum(p.numel() for p in model.parameters())

# Latency smoke (100 batches, drop first):
x0, _, K0, R0, t0 = next(iter(loader))
x0, K0, R0, t0 = x0.to(device), K0.to(device), R0.to(device), t0.to(device)
for _ in range(5):
    with torch.no_grad():
        _ = model(x0, K=K0, R=R0, t=t0)
import time
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        _ = model(x0, K=K0, R=R0, t=t0)
t1 = time.perf_counter()
latency_ms = (t1 - t0) / 100 * 1000
```

Pass `parents` to `compute_all_metrics` so the report contains `bone_length_error`:

```python
learned_report = compute_all_metrics(preds, gts, parents=MPI_INF_3DHP_28_PARENTS)
```

Also evaluate the new baselines and store per-baseline reports. The final JSON should keep scalar metrics plus `per_joint_mpjpe` and `per_view_mpjpe` arrays.

### 3.2 Create `motionflow_mv/eval/sota_reporting.py`

A small module that converts the extended JSON into a paper-ready Markdown table and a LaTeX fragment.

```python
import json
from pathlib import Path

def results_to_markdown(json_path: str, out_md: str, title: str = "SOTA Comparison"):
    data = json.loads(Path(json_path).read_text())
    lines = [f"## {title}", "", "| Method | MPJPE | PA-MPJPE | PCK@50 | AUC | Params | Latency ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in data.items():
        lines.append(
            f"| {name} | {r['mpjpe']:.2f} | {r['pa_mpjpe']:.2f} | "
            f"{r['pck@50mm']:.3f} | {r['pck_auc']:.3f} | {r.get('n_params', '-')} | {r.get('latency_ms', '-'):.2f} |"
        )
    Path(out_md).write_text("\n".join(lines))
```

### 3.3 Output artifacts

- `outputs/sota_comparison_extended.json` — scalar and array metrics for every baseline.
- `docs/tables/icra2027/sota_comparison_extended.md` — Markdown table for the paper draft.
- `docs/tables/icra2027/sota_comparison_extended.tex` — LaTeX table for the paper draft.
- (Optional) `experiments/compare_sota_baselines_h36m_smoke.py` — a 30-line wrapper that calls the same harness on `data/webbridge/h36m/h36m_test_multiview_m.npz` if the file is available.

### 3.4 No model or loss changes

This is an evaluation-only proposal. No training loss, architecture, or data loader is modified.

## 4. Smoke-Test Plan

Run the harness on a small slice of MPI-INF-3DHP clean data.

```bash
python experiments/compare_sota_baselines.py \
    --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_full_ppw005_20ep.pth \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --out_json outputs/sota_comparison_extended_smoke.json \
    --clip_len 13 \
    --batch_size 8 \
    --val_stride 200
```

Use `--val_stride 200` so only ~10 clips are evaluated; the whole smoke should finish in minutes.

**Pass/fail criteria**

- Pass: script completes without crashes in < 15 minutes on a single CPU/GPU session.
- Pass: all methods return finite MPJPE and PA-MPJPE.
- Pass: anchor MPJPE stays within 0.3 mm of 9.32 mm on the small sample.
- Pass: DLT/IRLS remain around 25 mm, and the two new baselines fall between 20 mm and 30 mm (sanity-check that they are valid geometric competitors).
- Fail: any baseline returns NaN, anchor deviates by > 0.5 mm, or runtime exceeds 15 minutes.

## 5. Evaluation Plan

1. **Full clean MPI-INF-3DHP run**
   ```bash
   python experiments/compare_sota_baselines.py \
       --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_full_ppw005_20ep.pth \
       --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
       --out_json outputs/sota_comparison_extended.json \
       --clip_len 13 --batch_size 8 --val_stride 50
   ```

2. **Report generation**
   ```bash
   python - <<'PY'
   from motionflow_mv.eval.sota_reporting import results_to_markdown
   results_to_markdown("outputs/sota_comparison_extended.json",
                       "docs/tables/icra2027/sota_comparison_extended.md")
   PY
   ```

3. **Metrics collected**
   - Scalar: MPJPE, PA-MPJPE, root-relative MPJPE, velocity MPJPE, PCK@50/100/150, AUC, bone-length error.
   - Arrays: per-joint MPJPE, per-view MPJPE.
   - Efficiency: inference latency (ms/frame) and total parameter count for the learned model.

4. **Cross-dataset sanity (optional / if data available)**
   Run the same script on the H36M test set with the 17-joint parent array to verify the harness generalizes to a different skeleton layout.

## 6. Estimated GPU/CPU Cost on RTX 4090

- **GPU**: only for the learned anchor inference; ~1–2 minutes for the full MPI-INF-3DHP test split at `val_stride=50`.
- **CPU**: DLT, IRLS, RANSAC-DLT, and top-2 DLT run on CPU; < 10 minutes for the full split.
- **Memory**: < 4 GB RAM; no training, no large batch.
- **Total**: < 15 minutes wall-clock on RTX 4090, entirely smoke-testable.

## 7. Risks & Fallback

| Risk | Mitigation |
|------|------------|
| RANSAC-DLT is too slow on CPU | Cap `max_iters=100` and `subset_size=3`; if still slow, fall back to `subset_size=2` or pre-filter views by confidence. |
| Per-joint/per-view arrays bloat the JSON | Keep arrays in `outputs/sota_comparison_extended_per_joint.json` and only scalar summary in the main JSON. |
| Anchor checkpoint path changes or is missing | Use the first matching `outputs/*pp*.pth` via glob; if none, skip learned model and run geometric baselines only. |
| H36M test data is not staged | Make the H36M wrapper read-only and skip it with a warning; the proposal remains valid on MPI-INF-3DHP alone. |
| New baselines accidentally beat the anchor | This is unlikely (they are purely geometric), but if it happens, it signals a serious bug in the learned-model eval path that must be debugged before any paper claim. |
