# 20 — Next-Iteration Synthesis (Swarm Iter 19)

This synthesis distils the 14 swarm reports produced in `docs/swarm_iter19/` into a
single actionable roadmap for pushing MotionFlow-MultiView toward ICRA/CVPR 2027.

## Current anchor

- **Best single model:** Bayesian Tri v2 single-model ~9.03 mm MPJPE on MPI-INF-3DHP S2/Seq1.
- **Best ensemble:** Bayesian Tri v2 ensemble **8.35 mm** MPJPE / 5.29 mm PA-MPJPE.
- **Running experiment:** `OmniMultiViewFusionV2` no-graph ablation
  (`--graph_num_layers 0`) is training on the local RTX 4090. After the 5-epoch
  freeze it reached ~24.88 mm val MPJPE in Epoch 6 and is continuing end-to-end.

## What the swarm agrees on

1. **Do not scale blindly.** Naïve upsizing (d→192/256) has historically underperformed smaller models. Wait for the no-graph ablation result before deciding whether to add capacity.
2. **Graph attention is promising but needs hardening.** The old `CrossViewGraphAttention` uses the beta `index_reduce_` API and is not view-mask aware. The newer `GraphJointAttentionV2` (already in repo) is the preferred implementation.
3. **Dense joint attention is a low-risk re-add.** The 8.35 mm anchor relied on joint-level transformer layers; removing them entirely may under-fit. A configurable `n_joint_layers` gives a clean ablation lever.
4. **WebBridge data is ready but under-used.** MPI-INF-3DHP S1–S8 canonical `.npz` files exist. Training currently uses only S1/S3; expanding to S1 + S3–S8 is the obvious next scaling step.
5. **Evaluation pipeline is wired but needs config discipline.** The eval runner defaulted to `graph_num_layers=1`, which mismatches the no-graph checkpoint. Hyperparameters must be tracked and passed consistently between train and eval.

## Code changes already landed (main)

- `motionflow_mv/fusion/omniview_fusion_v2.py`
  - Replaced `CrossViewGraphAttention` with the hardened `GraphJointAttentionV2`.
  - `graph_num_layers=0` now omits the graph module entirely.
  - Added `--n_joint_layers` dense joint-level transformer layers (per-view, batch_first).
- `experiments/train_omniview_fusion_v2_mpiinf3dhp.py`
  - Added `--n_joint_layers` argument.
  - Freeze-stage logic now treats `joint_attn` as a new head.
- `experiments/eval_omniview_fusion_v2_mpiinf3dhp.py`
  - Added `--n_joint_layers` argument and passes it to `build_model`.
- `scripts/eval_omniview_fusion_v2_wsl.sh`
  - Defaults now match the no-graph ablation (`graph_num_layers=0`, `n_joint_layers=0`).

## Proposed next experiments

### 1. Complete the no-graph ablation (in progress)
- **Goal:** obtain a single-model MPJPE on MPI-INF-3DHP S2/Seq1.
- **Trigger:** current background training finishes.
- **Action:** run `scripts/eval_omniview_fusion_v2_wsl.sh` against the produced checkpoint, then run robustness and variable-view curves.
- **Decision gate:**
  - **< 8.35 mm:** graph attention may be unnecessary; still run graph-v2 to confirm.
  - **8.35–9.0 mm:** run a focused ablation grid (graph=0/1, dense=0/1).
  - **> 9.0 mm:** fall back to the Bayesian Tri v2 stack and improve it incrementally.

### 2. Dense+Graph v2 full run
- **Model:** `d=128, n_st_layers=3, residual_hidden=256, graph_num_layers=1, n_joint_layers=1`.
- **Data:** all MPI-INF-3DHP train sequences (S1, S3–S8), val on S2/Seq1.
- **Warm-start:** from `outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth`.
- **Epochs:** 5 freeze + 30 end-to-end.
- **Expected output:** `outputs/omniview_fusion_v2_d128_dense_graph_v2.pth`.

### 3. WebBridge cross-dataset smoke
- Fix `configs/benchmark_webbridge_h36m_test_smoke.yaml` to include `source_n_views: 14`.
- Run the WebBridge benchmark harness on H36M S9/S11 to get the first clean MPI→H36M transfer number.

### 4. Publication readiness
- Maintain a single YAML split manifest under `configs/splits/mpiinf3dhp_train_val_test.yaml`.
- Record every run with seed, config, and final metrics in `outputs/omniview_v2_ablation_<name>/`.
- Update `docs/paper_method_section_draft.md` once dense/graph v2 numbers are in.

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| GPU is busy with the no-graph run. | Do not launch another GPU training until it finishes; CPU work (code, configs, reports) is safe. |
| Graph v2 may still be slower than no-graph. | Profile one epoch before committing to a full run; if slow, use `graph_num_layers=0, n_joint_layers=1` as the primary variant. |
| Multi-dataset units may be inconsistent. | Use only `_m.npz` (meter) canonical files for any mixed training. |
| Variable-view wrapper may fail with graph edges. | Graph v2 still connects all `n_views` nodes; evaluate variable-view carefully and add active-view masking if needed. |

## Immediate next step

Wait for the no-graph ablation to finish, run the full evaluation, then decide whether the first dense+graph v2 run is the best use of the next GPU block.
