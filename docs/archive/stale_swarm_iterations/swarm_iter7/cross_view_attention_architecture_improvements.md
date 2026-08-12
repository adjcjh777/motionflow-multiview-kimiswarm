# Cross-view attention architecture improvements

I investigated the cross-view attention variants in the repo. Below is the report content that should be saved at `docs/swarm_iter7/cross_view_attention_architecture_improvements.md`.

```markdown
# Cross-View Attention Architecture Improvements

## 1. Current state

Several cross-view attention variants already exist:

| Component | File | Key detail |
|---|---|---|
| Spatio-temporal cross-view + residual model | `motionflow_mv/fusion/ray_attention_temporal_crossview_residual_model.py:27` | Adds residual head on top of `RayAttentionFusionModelTemporalCrossview`; attends jointly over `(time, view)` tokens per joint. |
| Pairwise cross-view attention | `motionflow_mv/fusion/ray_attention_crossview_model.py:103` | `RayAttentionFusionModelCrossView` builds `V×V` pairwise view tokens with a bottleneck `d_cross=32` inside the per-frame encoder. |
| Training scripts | `experiments/train_ray_attention_temporal_crossview_residual_mpiinf3dhp.py`, `experiments/train_ray_attention_crossview_mpiinf3dhp.py` | Mirror the temporal-residual trainer, using `n_st_layers` / `n_crossview_layers`. |
| Test sanity check | `tests/test_ray_attention_temporal_crossview.py` | Forward/backward shape checks for the combined temporal+cross-view model. |

Full-data results (`outputs/eval_crossview_residual_d64_h128.json`) show the spatio-temporal cross-view residual model reaches **15.29 mm MPJPE / 13.49 mm PA-MPJPE / AUC 0.898** on MPI-INF-3DHP S2/Seq1, which is worse than the current best temporal-only residual (`RayAttentionFusionModelTemporalResidual`) at **11.17 mm / 8.24 mm / AUC 0.926** (`docs/paper_draft_icra_cvpr_2027.md:80`). Smoke tests had suggested the opposite (cross-view 11.56 mm vs temporal 17.01 mm), so the cross-view mechanism currently overfits on the full training set.

## 2. Gap / opportunity

The existing cross-view attention is *geometry-agnostic*: it uses learned time/view positional embeddings (`view_pos_embed`, `time_pos_embed`) and generic self-attention over `(T×V)` tokens. It therefore ignores the fact that some view pairs are geometrically more informative for triangulating a 3D joint (e.g., wide-baseline pairs with near-perpendicular rays vs. parallel or co-linear rays). Injecting an **epipolar/ray-angle attention bias** into the cross-view block would make the model explicitly geometry-aware, reduce reliance on memorized positional embeddings, and could close the gap to the temporal-only baseline.

## 3. Concrete next step

Implement a geometry-aware cross-view attention variant and train it on full MPI-INF-3DHP:

1. Add `motionflow_mv/fusion/ray_attention_temporal_crossview_geo_model.py` that extends `RayAttentionFusionModelTemporalCrossview` (or `RayAttentionFusionModelCrossView`) by:
   - Computing per-joint ray directions for each view from the 2D keypoints and calibrated cameras.
   - Building a `(V, V)` pairwise geometric bias matrix from the angle between rays / camera baselines (e.g., `b_ij = softplus(1 − |ray_i · ray_j|)`).
   - Adding this bias to the attention scores before softmax in the spatio-temporal (or pairwise) transformer, instead of or in addition to the learned `view_pos_embed`.
2. Add `experiments/train_ray_attention_temporal_crossview_geo_mpiinf3dhp.py` mirroring `train_ray_attention_temporal_crossview_residual_mpiinf3dhp.py` but loading the new model.
3. Train with the standard full MPI-INF-3DHP protocol: train S1 Seq1+Seq2, validate S2 Seq1, `clip_len=13`, `d=64`, `residual_hidden=128`, 5–10 epochs.

## 4. Expected success metric

Run `experiments/eval_ray_attention_temporal_residual_v3.py` (adapted to the new model) on `data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz` and report:

- MPI-INF-3DHP cross-subject **MPJPE** and **PA-MPJPE** in mm.
- **PCK@50/100/150 mm** and **AUC (0–150 mm)**.

Target: beat the current cross-view residual baseline of **15.29 mm MPJPE / 13.49 mm PA-MPJPE**, and ideally approach or surpass the temporal-only best of **11.17 mm MPJPE / 8.24 mm PA-MPJPE / AUC 0.926**.

## 5. Risks / blockers

- **Memory:** The spatio-temporal transformer already attends over `T×V` tokens; adding a computed `(V, V)` bias is cheap, but full attention over `T×V` remains memory-heavy on the RTX 4090.
- **No guaranteed gain:** Cross-view attention has not yet beaten the temporal-only model on full data; geometry-aware bias is a hypothesis, not a proven fix.
- **A800-D and Docker are read-only** — cannot modify containers or launch training there; use the local RTX 4090 or WSL environment.
- **WebBridge data:** `data/webbridge/mpi_inf_3dhp/*.npz` must be present locally; do not commit large files.
- **No git commits/pushes** for this exploration step.
```

**Summary:** The current cross-view attention variants exist and are trainable, but on full MPI-INF-3DHP they underperform the temporal-only residual model (15.29 mm vs 11.17 mm MPJPE). The actionable next step is to replace generic learned view positional embeddings with an epipolar/ray-angle attention bias, train the new geometry-aware cross-view variant, and re-evaluate. I could not write the file to disk because this sub-agent role is read-only, but the markdown above is ready to be saved at `docs/swarm_iter7/cross_view_attention_architecture_improvements.md`.