# Ablation study design for paper

I investigated the ablation landscape and prepared the report below. I do **not** have file-editing tools in this read-only exploration role, so I could not write it to disk. The report content is provided here for the parent agent to place at the requested path.

**Intended path:** `docs/swarm_iter7/ablation_study_design_for_paper.md`

---

# Ablation Study Design for ICRA/CVPR 2027 Paper

## 1. Current state

The paper draft (`docs/paper_draft_icra_cvpr_2027.md`) presents a single ablation row for the residual head:

| Model | MPJPE (mm) | PA-MPJPE (mm) | Params |
|---|---:|---:|---:|
| Raw DLT | 25.21 | — | — |
| Temporal ray-attention (no residual) | 25.21 | 24.14 | 218 k |
| **Residual full 5-epoch (d=64, h=128)** | **11.17** | **8.24** | 243 k |

Existing ablation scripts already in the repo:
- `experiments/ablate_residual_hidden_mpiinf3dhp.py` — varies `residual_hidden ∈ {32,64,128,256}`; best is `h=128` at 12.99 mm (smoke run) (`docs/swarm_iter6/ablate_residual_hidden_report.md`).
- `experiments/ablation_v1_v2_h36m.py` — compares view-only (v1) vs. view+joint attention (v2) on a 500-frame H36M subset.
- `experiments/run_ablations.py` — generic v3 ablation harness, but it targets the outdated `RayAttentionFusionModelV3` and a synthetic/H36M stub, not the current temporal-residual model.
- `motionflow_mv/fusion/ray_attention_temporal_residual_model.py` — the current best model (`RayAttentionFusionModelTemporalResidual`).
- Verified results: `docs/swarm_iter7/verified_results.json` tracks the 11.17 mm checkpoint.

## 2. Gap / opportunity

There is **no unified paper ablation** that isolates each design choice on the same MPI-INF-3DHP split with identical training/evaluation settings. The existing ablations are scattered across different scripts, datasets, and model versions, so the paper cannot yet claim that the residual head, temporal attention, and non-linear MLP each independently contribute to the 11.17 mm result.

Specific missing ablations:
1. **Residual head on/off** — run with the exact same temporal-attention backbone.
2. **Temporal context on/off** — single-frame residual vs. temporal residual.
3. **Linear vs. non-linear residual head** — 1-layer linear vs. the current 2-hidden-layer MLP.
4. **Capacity vs. accuracy** — already partially covered by `residual_hidden` ablation, but not in the final full-data run.

## 3. Concrete next step

Create and run `experiments/run_paper_ablations_mpiinf3dhp.py`, a single script that trains and evaluates the following five fixed variants on the standard MPI-INF-3DHP split (train S1 Seq1+2, val S2 Seq1) with the same hyperparameters (`d=64`, `residual_hidden=128`, `clip_len=13`, 5 epochs, batch=8, 4000 train clips):

| # | Variant | What it tests |
|---|---|---|
| A | DLT baseline | Geometric lower bound without learning |
| B | Temporal ray-attention, **no residual** | Value of the residual head alone |
| C | **Temporal ray-attention + residual** (current best) | Full model |
| D | Single-frame ray-attention + residual | Value of temporal attention |
| E | Temporal ray-attention + **linear** residual | Whether non-linearity is needed |

Implementation notes:
- Variant B is obtained by setting `residual_hidden=0` or by subclassing `RayAttentionFusionModelTemporalResidual` and skipping the residual MLP.
- Variant D sets `clip_len=1` in the training/evaluation dataset.
- Variant E replaces the 2-hidden-layer MLP with a single `nn.Linear(d+3, 3)`.
- The script reuses `TemporalClipDataset`/`RandomClipDataset` from `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py` and writes a Markdown table to `docs/swarm_iter7/paper_ablation_mpiinf3dhp.md` plus a JSON file to `outputs/paper_ablation_mpiinf3dhp.json`.

## 4. Expected success metric

After running the script we should have:

| Variant | Expected MPJPE (mm) | Success criterion |
|---|---|---|
| A DLT | ~25 | Reproduces baseline |
| B no residual | ~25 | Confirms residual head is necessary for the large gain |
| C full | **~11** | Matches current best |
| D single-frame residual | 13–16 | Shows temporal attention helps |
| E linear residual | 14–18 | Shows non-linear MLP is useful |

Success is defined as a **clear ordering C < D, E < B ≈ A** on the same validation split, giving the paper a strong, self-contained ablation table.

## 5. Risks / blockers

| Risk | Likelihood | Mitigation |
|---|---|---|
| **A800-D / Docker are read-only** — cannot train there | High | Run on the local RTX 4090; keep batch size and epochs modest |
| **Long wall-clock time** — 5 full trainings ≈ 1–2 days | Medium | Run smoke-test first (`--epochs 1`) to validate the harness, then launch the full run overnight |
| **WebBridge data not downloaded** | Medium | Check `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz`; if missing, run the WebBridge download script (do not commit large files) |
| **Class-name collision** between `ray_attention_temporal_residual_v3_model.py` and `ray_attention_temporal_residual_model_v3.py` | Low | Only import from the base `ray_attention_temporal_residual_model.py` for variants B–E |
| **Variant B diverges / high loss** | Low | Use the same weight head and DLT layer as the residual model; just zero or skip the residual MLP |

## 6. Suggested file naming

- Script: `experiments/run_paper_ablations_mpiinf3dhp.py`
- Report output: `docs/swarm_iter7/paper_ablation_mpiinf3dhp.md`
- Metrics output: `outputs/paper_ablation_mpiinf3dhp.json`
- Checkpoints: `outputs/paper_ablation_mpiinf3dhp_{variant}.pth`

---

**Summary:** The key gap is a unified, paper-quality ablation harness for the current best `RayAttentionFusionModelTemporalResidual` on MPI-INF-3DHP. The proposed next step is to implement `experiments/run_paper_ablations_mpiinf3dhp.py` comparing DLT, no-residual, full, single-frame, and linear-residual variants on the same split, producing a single self-contained table for the paper.