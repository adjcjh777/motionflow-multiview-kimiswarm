# CVPR 2027 Risk Register — MotionFlow-MultiView

> **Scope:** All work required to submit a competitive CVPR 2027 paper.  
> **Target deadline:** ~2026-11-13 (CVPR 2027; confirm when official).  
> **Last updated:** 2026-08-11  
> **Source of truth for details:** `docs/cvpr2027_status.md`, `docs/open_blockers.md`, `docs/roadmap_cvpr2027.md`, `docs/results_true_gt_h36m.md`.

## How to read this register

- **Likelihood / Impact:** Low / Medium / High.
- **Status:**
  -  **Open** — not yet mitigated.
  - 🟡 **Monitoring** — mitigation in progress or risk partially controlled.
  - 🟢 **Resolved** — no longer threatens submission.
- **Owner:** The role or agent responsible for driving the mitigation (not the only person who can help).

---

## Active risks

| ID | Risk | Likelihood | Impact | Owner | Mitigation | Status | Contingency / Trigger |
|---|---|---|---|---|---|---|---|
| R1 | **H36M true-GT learned models diverge / overfit** (v25, v80, v57 all trail Iskakov/DLT; v25 spikes to 207.62 mm by epoch 8). | High | High | Algorithm lead (currently `qwen3.8max`) | Run v25 true-GT ablation queue (`v25_true_gt_baseline_fix` → geometry reg → mixed dataset); add weight decay / LR reduction / early stopping / augmentation clamp; re-run v80/v57 with the same regularisation once a recipe works. | 🟡 Monitoring | If no recipe closes gap by 2026-09-01, pivot story: position v25 as a strong-but-not-record baseline and emphasise sparse-view / cross-domain robustness instead of absolute MPJPE. |
| R2 | **MPI-INF-3DHP detected-2D / camera / label alignment** (16 `_m.npz` files ready, but DLT baseline ~326–400 mm). | Medium | High | Data lead | Diagnose camera/label coordinate-frame mismatch; fix MPI loader/canonical builder; re-run DLT until ~20–30 mm. | 🟡 Monitoring | If alignment cannot be fixed by 2026-09-01, drop MPI main results and cite only the GT-projected 2D baseline (~23.8 mm) with a protocol caveat. |
| R3 | **AIST++ full medium validation missing** — only 3-epoch smoke exists; v25/v80 far behind DLT/Iskakov. | Medium | Medium | `agent-67` (when GPU free) | Run v25/v80 medium on `configs/splits/aist_only_smoke.yaml`; port regularisation fixes from R1. | 🟡 Monitoring | Drop AIST++ from the main leaderboard and cite smoke numbers only as preliminary cross-domain evidence. |
| R4 | **v25 training crash on Shelf/Campus non-circular smoke** (`CUDA assert` in `epipolar_attention_bias.py`). | Medium | Medium | Geometry-fusion lead | Re-run with `CUDA_LAUNCH_BLOCKING=1`; bisect cross-view modules; evaluate Shelf/Campus via `experiments/eval_shelf_campus_standard.py` as a workaround. | 🟡 Monitoring | If training cannot be fixed, compare methods only on H36M/MPI and describe Shelf/Campus results from eval-only checkpoints. |
| R5 | **Cross-domain training manifest / recipe not created** — no mixed-dataset experiment supports the robustness claim. | Medium | High | Data + modeling lead | Create `configs/splits/h36m_aist_shelf_campus_mix.yaml`; resolve domain-embedding and view-count mismatches; run a short smoke measuring per-domain val MPJPE. | 🔴 Open | Limit cross-domain claim to transfer from H36M→MPI or H36M→AIST++ with separate manifests, avoiding a single mixed loader. |
| R6 | **SOTA baselines (VoxelPose / MVPose) not reproduced** — related-work table is thin. | Medium | Medium | Baselines lead | Identify compatible code/checkpoints; add scripts under `scripts/sota_baselines/`; run on H36M true-GT and Shelf/Campus detected. | 🔴 Open | Cite published numbers on identical splits and run only Iskakov + one other baseline if time is short. |
| R7 | **Paper rewrite around sparse-view / cross-domain robustness not complete.** | Medium | High | Paper lead | Update `docs/paper_draft_icra_cvpr_2027.md` abstract/intro; replace old absolute-MPJPE narrative; add real citations; rebuild tables with true-GT numbers. | 🟡 Monitoring | Write the paper as a focused robustness paper from the start; treat old draft sections as deprecated. |
| R8 | **Single local GPU bottleneck** (RTX 4090 runs one training job at a time; A800 read-only). | High | High | Infra / compute lead | Queue GPU work serially; run CPU-only tasks (detection, plotting, data prep) in parallel; verify `nvidia-smi` before every launch. | 🟡 Monitoring | Re-prioritise: run only divergence-fix and one cross-domain medium before deadline; defer non-essential ablations. |
| R9 | **Timeline compression / missed deadline** (only ~13 weeks remain; six-week sprint is already aggressive). | High | High | Project lead | Re-scope to a minimal viable submission: H36M true-GT leaderboard + one cross-dataset result + robustness curves + rewritten paper. | 🟡 Monitoring | If P0 divergence is unresolved by 2026-09-01, pivot to a diagnostic/lessons-learned workshop paper or target a later venue. |
| R10 | **Reproducibility / environment / citations not locked** (placeholder refs, unfrozen dependencies, missing READMEs). | Medium | Medium | Paper lead + infra lead | Freeze `requirements.txt`/conda env; replace placeholder citations; add one-command training/eval README. | 🔴 Open | For submission, include an appendix with exact commands and checkpoint naming; fix dependency pins after deadline. |
| R11 | **Residual circular-label leakage** (old `data/h36m_hf` or `data/webbridge/h36m*.npz` accidentally used). | Low | High | Data lead | Use only `data/h36m_true_gt/` and detected-2D `.npz`; enforce manifest-level checks; re-run `scripts/check_true_gt_reprojection.py` before final numbers. | 🟡 Monitoring | If leakage is found, regenerate affected `.npz` and re-run all baselines before paper finalisation. |
| R12 | **A800-D / Docker `motionflow` treated as writeable** (policy violation / accidental writes). | Low | Medium | Infra lead | Treat A800-D and the Docker service as read-only; keep all writes in local repo. | 🟢 Resolved | Document guardrail in `AGENTS.md`; audit any script that touches remote paths. |

---

## Risk heat map

| Impact | Likelihood |
|---|---|
| **High** | R1, R8, R9 |
| **Medium** | R2, R3, R4, R5, R6, R7, R10, R11 |
| **Low** | R12 |

*(R12 is shown for completeness; it is resolved.)*

---

## Risk-to-blocker mapping

| Risk ID | Maps to blocker / issue | Location |
|---|---|---|
| R1 | P0-4 H36M true-GT learned-model divergence / overfitting | `docs/open_blockers.md` |
| R2 | P1-1 MPI-INF-3DHP detected-2D / camera / label alignment | `docs/open_blockers.md` |
| R3 | P1-3 AIST++ full medium validation | `docs/open_blockers.md` |
| R4 | P1-2 v25 crash on Shelf/Campus non-circular smoke | `docs/open_blockers.md` |
| R5 | P1-5 Cross-domain training manifest and recipe | `docs/open_blockers.md` |
| R6 | P1-4 Additional SOTA baselines | `docs/open_blockers.md` |
| R7 | P1-6 Paper rewrite | `docs/open_blockers.md` |

---

## Mitigation watch list

| Watch item | Next check | Expected signal |
|---|---|---|
| v25 true-GT ablation 1 (`v25_true_gt_baseline_fix`) complete | 2026-08-12 | Log file shows early-stopped best epoch and val MPJPE. |
| MPI S2/Seq2 detection finished | 2026-08-12 | `outputs/mpi_s2_seq2_detected_cpu_20260811.log` reaches end of chunk 6081. |
| v80/v57 regularisation recipe decided | 2026-09-01 | New medium run on H36M true-GT no longer diverges after best epoch. |
| Cross-domain smoke manifest created | 2026-09-01 | `configs/splits/h36m_aist_shelf_campus_mix.yaml` exists and runs end-to-end. |
| Paper draft v0.9 | 2026-10-31 | `docs/paper_draft_icra_cvpr_2027.md` has no placeholder citations and matches true-GT tables. |

---

## Related documents

- `docs/cvpr2027_submission_checklist.md` — full submission checklist with owners and deadlines.
- `docs/cvpr2027_status.md` — day-to-day status and next actions.
- `docs/open_blockers.md` — P0/P1 blocker details and next steps.
- `docs/roadmap_cvpr2027.md` — strategic roadmap and revised contribution.
- `docs/results_true_gt_h36m.md` — current true-GT numbers that anchor the risks above.
