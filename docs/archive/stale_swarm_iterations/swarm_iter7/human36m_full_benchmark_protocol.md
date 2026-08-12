# Human3.6M Full Benchmark Protocol

## 1. Current state

- **Data converter.** `motionflow_mv/data/webbridge_loader.py:86–194` and `experiments/prepare_h36m_multiview.py` convert the Hugging Face preprocessed H36M archive (`data/h36m_hf/h36m_sh_conf_cam_source_final.pkl.zip`) into canonical multi-view `.npz` files.
- **Preprocessed data.** Only a small subset is materialized in `data/h36m_hf/`:
  - Train: `s_01_acts_02_03_..._16_multiview_m.npz` (62 k frames).
  - Val/test single-action only: `s_05_acts_02_multiview_m.npz`, `s_09_acts_02_multiview_m.npz`, `s_11_acts_02_multiview_m.npz`.
- **Best model.** `outputs/ray_attention_temporal_residual_h36m.pth` (d=64, residual_hidden=128, ~202 k params) was trained on S1 actions 2–16 and evaluated on S5 action 02, giving **MPJPE 5.74 mm**, **PA-MPJPE 3.99 mm**, **PCK AUC 0.9618** (`outputs/eval_residual_h36m_h128.json`).
- **Metrics.** `motionflow_mv/eval/metrics.py` supports MPJPE, PA-MPJPE, PCK@50/100/150, AUC, and per-joint/per-view breakdowns.
- **Scripts.** `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py` and `experiments/eval_ray_attention_temporal_residual_v1.py` accept any canonical `.npz` and are reused for H36M.

## 2. Gap / opportunity

The paper currently reports a **single-action, single-validation-subject result** (S1 actions 2–16 → S5 action 02). It does not follow the canonical Human3.6M protocol: **train on S1, S5, S6, S7, S8 (all actions); test on S9 and S11 (all actions)** with per-action and mean metrics.

Worse, the existing test-subject `.npz` files are corrupted:

- `s_09_acts_02_multiview_m.npz`: DLT triangulation vs. stored 3D GT ≈ **736 mm**.
- `s_11_acts_02_multiview_m.npz`: DLT triangulation vs. stored 3D GT **71,402 mm**.

A valid full-protocol benchmark is needed before the H36M numbers can be called publication-ready.

## 3. Concrete next step

1. **Fix the H36M test-subject preprocessing.**
   - Debug `prepare_h36m_multiview.py` / `webbridge_loader.convert_human36m` so that test-subject cameras are correctly matched (the current grouping by `ca_XX` suffix assigns wrong `camera_name` values for S9/S11).
   - Validate each generated `.npz` by triangulating the 2D points with the stored cameras and checking that DLT vs. GT MPJPE is < 10 mm.

2. **Generate the full protocol dataset.**
   - Train subjects: S1, S5, S6, S7, S8, actions 2–16.
   - Test subjects: S9 and S11, actions 2–16.
   - Keep one `.npz` per (subject, action) so per-action metrics are trivial.

3. **Add a full-benchmark runner** `experiments/run_h36m_full_benchmark.py` that:
   - Trains `RayAttentionFusionModelTemporalResidual` (d=64, residual_hidden=128, clip_len=13) on all train `.npz`s, with early stopping on S5 action 02 or a small validation split.
   - Evaluates the trained checkpoint on every S9/S11 action `.npz`, computes MPJPE / PA-MPJPE / PCK / AUC, and writes `docs/results_h36m_full_protocol.md` with per-action and mean rows.
   - Also evaluates the existing `outputs/ray_attention_temporal_residual_h36m.pth` on the full test set as a baseline without retraining.

Example generation command after the fix:

```bash
# one subject, all actions
for a in {02..16}; do
  python experiments/prepare_h36m_multiview.py --subject 9 --actions $a --split test --out_dir data/h36m_hffull
done
```

## 4. Expected success metric

- A results table for **S9 and S11 all actions (2–16)** plus mean MPJPE / PA-MPJPE / PCK / AUC.
- Target: mean test MPJPE ≤ **6 mm** and PA-MPJPE ≤ **5 mm**, matching the high quality of the single-action result.
- If the S1-only checkpoint (`ray_attention_temporal_residual_h36m.pth`) also scores ≤ 6 mm mean on the full test set, the model already generalizes; otherwise the full-train model should close the gap.

## 5. Risks / blockers

- **Corrupted test preprocessing:** S9/S11 camera mapping must be fixed first. If the Hugging Face archive is fundamentally unsalvageable, switch to the official Human3.6M release or a verified WebBridge source.
- **Storage:** the preprocessed pkl is 675 MB; generating all `.npz`s may add several GB. They are (or should be) gitignored in `data/`.
- **Compute:** local RTX 4090 only; A800-D is read-only. Schedule full-train jobs serially and use `batch_size ≤ 8`.
- **Do not commit large files:** ensure new `.npz` outputs and logs stay out of git.
