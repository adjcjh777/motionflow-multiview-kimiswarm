# Real-world GVHMR multi-view demo

**Summary**

I investigated the real-world GVHMR demo readiness. The best model (`RayAttentionFusionModelTemporalResidual`) is implemented and registered, and the GVHMR demo file `data/gvhmr_demo/hmr4d_results.pt` is real (152 frames, 667 KB). However, the existing GVHMR demo script only loads older v1/v3 plugins and never exercises the 11.17 mm residual checkpoint. The main opportunity is to add a new residual-flavoured GVHMR demo plus a GVHMR-style fine-tuning run.

Because this read-only exploration role does not have a `Write` tool, I could not create the file directly. The full report content is below; it should be saved to:

```text
docs/swarm_iter7/real-world_gvhmr_multi-view_demo.md
```

---

```markdown
# Real-world GVHMR Multi-View Demo

## 1. Current state

- **Best model:** `RayAttentionFusionModelTemporalResidual` is implemented in `motionflow_mv/fusion/ray_attention_temporal_residual_model.py:38` and wrapped as a plugin in `motionflow_mv/fusion/ray_attention_temporal_residual_module.py:13`; it is registered in `motionflow_mv/fusion/__init__.py:9-12`.
- **Best checkpoint:** `outputs/ray_attention_temporal_residual_final5.pth` (243 k params), reaching MPI-INF-3DHP S1→S2/Seq1 **MPJPE 11.17 mm**, **PA-MPJPE 8.24 mm**, **AUC 0.9256** (`docs/swarm_iter7/exploration_summary.md:7-10`).
- **GVHMR IR adapter:** `motionflow_mv/ir/gvhmr_adapter.py:26` loads a single-view `data/gvhmr_demo/hmr4d_results.pt`. The file is present (667 KB, 152 frames) and contains `smpl_params_global`, `smpl_params_incam`, `K_fullimg`, and `net_outputs`.
- **Existing GVHMR demo:** `experiments/demo_gvhmr_multiview_projection.py` (lines 123-317) only loads the older `RayAttentionFusionModelV3` and metric-normalised v1 plugin; it does **not** use the temporal-residual model.
- **Existing temporal-residual demo:** `experiments/demo_ray_attention_temporal_residual.py` runs on canonical `.npz` data with ground-truth 3D, not on real GVHMR output.

## 2. Gap / opportunity

The paper draft (`docs/paper_draft_icra_cvpr_2027.md:141`) explicitly lists “the absence of real-world GVHMR output evaluation” as a limitation. The current demo uses stale v1/v3 models, while the best residual checkpoint has never seen monocular SMPL-style errors (fitting bias, jitter, outliers). Fine-tuning the best checkpoint with GVHMR-style noise and running it on the real `hmr4d_results.pt` closes the synthetic-to-real gap and removes the inconsistency between the paper’s best numbers and the demo code.

## 3. Concrete next step

Add **new** scripts (do not modify the existing demo or trainer):

1. `experiments/demo_gvhmr_multiview_projection_residual.py` — load `RayAttentionTemporalResidualFusionModule` with `outputs/ray_attention_temporal_residual_final5.pth`, run SMPL forward on the real `data/gvhmr_demo/hmr4d_results.pt`, project the single-view world joints through 4 virtual cameras, inject optional GVHMR-style noise, fuse with the temporal-residual plugin, and compare the fused multi-view output against the original single-view GVHMR world reference.
2. `experiments/train_gvhmr_style_temporal_residual.py` — copy `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py` and add CLI flags: `--noise_std`, `--dropout_rate`, `--outlier_rate`, `--pretrained`, and a lower fine-tuning LR (1e-4). Train on MPI-INF-3DHP WebBridge data while preserving clean accuracy.

Example commands:

```bash
conda run -n mf python experiments/train_gvhmr_style_temporal_residual.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 64 --residual_hidden 128 \
    --epochs 20 --batch_size 8 --train_samples 4000 \
    --noise_std 2.0 --dropout_rate 0.1 --outlier_rate 0.05 \
    --pretrained outputs/ray_attention_temporal_residual_final5.pth \
    --lr 1e-4 --output outputs/ray_attention_temporal_residual_gvhmr.pth

conda run -n mf python experiments/demo_gvhmr_multiview_projection_residual.py \
    --input data/gvhmr_demo/hmr4d_results.pt \
    --n_views 4 --gvhmr_noise_std 2.0 --gvhmr_outlier_rate 0.05 \
    --checkpoint outputs/ray_attention_temporal_residual_gvhmr.pth
```

## 4. Expected success metric

- **GVHMR demo MPJPE** vs single-view GVHMR world reference: target **≤ 0.010 m**, ideally **≤ 0.008 m**.
- **MPI-INF-3DHP S2/Seq1 clean validation** via `experiments/eval_ray_attention_temporal_residual_mpiinf3dhp.py`: keep MPJPE in the **11.0–11.7 mm** band to ensure no catastrophic forgetting.
- **PA-MPJPE / PCK** on MPI stay within ±5 % relative to the original checkpoint.
- Optional: report mean reprojection error on the virtual cameras as a geometric sanity check.

## 5. Risks / blockers

- **No true multi-view capture:** the current `hmr4d_results.pt` is single-view; the demo simulates multi-view by re-projecting through virtual cameras. A later real-world demo needs synchronized multi-view video and calibration.
- **No 3D GT for the demo:** evaluation is relative to the single-view GVHMR world reference. Mitigate by also reporting reprojection error on the virtual cameras.
- **Catastrophic forgetting:** use a low fine-tuning LR (1e-4) and select the checkpoint with the lowest MPI validation MPJPE.
- **Per-view coordinate alignment:** if moving to real per-view GVHMR inference, each view’s gravity-aligned frame must be aligned to the calibrated rig.
- **A800-D / Docker read-only:** do not modify anything there; only copy/read data if required. WebBridge data is already local; do not commit large files.
```