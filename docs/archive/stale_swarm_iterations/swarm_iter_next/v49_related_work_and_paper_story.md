# v49: Related Work & Paper Story Packaging

**Status:** Proposal / ready for design review  
**Labels:** `paper`, `P1-next`  
**Tracking issue:** #166 (proposed)  
**Depends on:** v46-SVG (#160), v47-temporal (#162), v48-domain (#164), v49-realtime-streaming (#165)  

---

## 1. Problem Statement

By v49 the project has accumulated four orthogonal technical directions—**sparse-view robustness** (v46), **temporal aggregation** (v47), **domain generalization** (v48), and **real-time streaming** (v49). Each has its own proposal, but there is no single coherent narrative that explains *why* these four pieces belong together, how they relate to the 2025–2026 literature, and how the ICRA/CVPR 2027 paper story should be told.

Specifically, the current paper story risks:

1. **Looking incremental.** Reviewers may see v46–v49 as four disconnected add-ons rather than a unified path toward practical multi-view pose estimation.
2. **Missing self-evolution framing.** The v37 self-critique reliability loop and the project-wide failure-analysis-driven iteration are not yet presented as a methodological contribution.
3. **Weak related-work positioning.** We cite papers, but we do not explicitly map each v46–v49 component onto the literature gaps it fills.
4. **Unclear hierarchy of claims.** It is not obvious which result is the primary claim (geometry fusion), which is a secondary robustness claim (sparse views + temporal), and which is a deployment claim (domain generalization + streaming).

This design note proposes a v49 **paper-story and related-work packaging** deliverable that resolves the above without adding new neural modules.

---

## 2. Proposed Approach

### 2.1 Single-sentence paper claim

> MotionFlow-MultiView is a self-evolving multi-view 3D human pose system that uses geometric triangulation as a foundation, learns per-view reliability from its own reprojection residuals, and then generalizes the resulting pose estimator across sparse views, time, domains, and real-time streaming constraints.

### 2.2 Story arc

The paper is told in four acts, each anchored to a technical module:

| Act | Technical content | What it answers | Related-work gap |
|---|---|---|---|
| **1. Geometry foundation** | v25 multi-view geometry fusion + v45 adaptive geometry weights | Why triangulation is still the strongest signal | End-to-end transformers often ignore projective geometry (Liao & Zhu 2023, Moliner & Huang 2024) |
| **2. Self-critique reliability** | v37 self-critique view reliability, v39 reliability-coupled refinement, v43 adaptive residual | How the model learns to trust its own views | Most fusion methods use hand-tuned confidence or hard outlier rejection (Bragagnolo et al. 2024; Davoodnia et al. 2024) |
| **3. Robustness in the wild** | v46 sparse-view generalization, v47 temporal aggregation, v48 domain generalization | How the same model works with missing views, motion blur, and in-the-wild video | Prior work trains for fixed camera rigs and studio data (Ghasemzadeh & Alahi 2024; Choudhury & Kitani 2023) |
| **4. Deployment** | v49 real-time streaming, dynamic view budget, causal temporal smoother | How the model runs live on constrained hardware | State-space (MV-SSM, Chharia et al. 2025) and ray-based lifters (RUMPL 2025) are accurate but not latency-aware |

### 2.3 How it fits with v46–v49 and the overall pipeline

```text
2D keypoints + cameras (any V >= 2)
        |
        v
[v25/v45 geometry fusion + v37/v39 self-critique reliability]
        |    <-- self-evolution feedback: reprojection residual -> reliability gate -> refined 3D
        v
[v46 sparse-view generalization]
        |
        v
[v47 temporal aggregation]
        |
        v
[v48 domain-invariant refinement]
        |
        v
[v49 real-time streaming head]
        |
        v
3D pose, per-joint uncertainty, and runtime budget
```

The related-work packaging is not a separate algorithmic module; it is the **narrative glue** that turns the v46–v49 stack into a single paper contribution.

---

## 3. Concrete Code-Level Changes

Only documentation and lightweight tooling are added. No existing training code, model, or config is modified.

### 3.1 New files

| File | Purpose |
|---|---|
| `docs/swarm_iter_next/v49_related_work_and_paper_story.md` | This design note. |
| `docs/paper_story_v49.md` | Single-page narrative with figure captions, claim hierarchy, and result tables. |
| `docs/related_work_mapping_v49.md` | Table mapping each v46–v49 module to specific literature gaps and papers. |
| (optional) `scripts/generate_related_work_table_v49.py` | Reads `docs/literature_review_multiview_pose.md` and emits the mapping table in LaTeX/Markdown for the paper. |

### 3.2 Files to update when the story is finalized

These are listed here for reference; they are **not** modified by the present proposal.

| File | Update |
|---|---|
| `docs/literature_review_multiview_pose.md` | Add a “v49 positioning” subsection for each of v46–v49. |
| `docs/results_snapshot_2026_08_09.md` | Populate the paper-story results table. |
| `README.md` (or a future `PAPER.md`) | Add the four-act story arc. |

### 3.3 No new training flags

This deliverable is purely paper/related-work packaging; no model or trainer flags are introduced.

---

## 4. Risks / Failure Modes

| Risk | Mitigation |
|---|---|
| Story overstates contribution relative to v25 baseline | Anchor every claim to a measured metric (MPJPE@k, domain gap, latency) and keep v25 as the primary baseline. |
| Related-work mapping is too generic | Cite the exact modules/papers already collected in `docs/literature_review_multiview_pose.md` and explain the specific gap each v46–v49 module fills. |
| Self-evolution loop is described too abstractly | Tie it to concrete mechanisms: v37 reprojection-residual reliability, v39 gating, v43 adaptive residual, and v46–v48 failure-driven redesign (dropout, temporal smoothing, domain weights). |
| v49 streaming results not ready in time | Treat v49 as a deployment extension; the core paper story is v45/v25 foundation + v37 self-critique + v46–v48 robustness. |
| Reviewers see v46–v48 as ablations, not a pipeline | Use the v31 paper-story structure (`docs/proposals/v31_paper_story_multiview_video_pipeline.md`) as a template: pipeline diagram, per-module claim, and a single end-to-end result table. |

---

## 5. Success Metrics and Experiments

Because this deliverable is documentation, the “smoke” and “full” experiments are validation of the paper narrative rather than GPU runs.

### 5.1 Smoke test

- **Hardware:** local WSL / RTX 4090 (no GPU training; only doc generation).
- **Config:** Run the optional `scripts/generate_related_work_table_v49.py`.
- **Expected outcome:**
  - Generated table references every v46–v49 module.
  - Every literature entry maps to at least one concrete technical mechanism.
  - No broken links or missing citations.

### 5.2 Full experiment

- **Hardware:** N/A (writing / review).
- **Config:** Produce `docs/paper_story_v49.md` with:
  - One figure showing the full v25 → v49 pipeline.
  - A results table containing the latest v25/v42/v43/v45/v46/v47/v48 numbers from `AGENTS.md` and `docs/results_snapshot_2026_08_09.md`.
  - A “self-evolution feedback loop” box explaining v37 → v39 → v43 → v46–v48 iteration.
- **Expected outcome:**
  - A reviewer can read the paper story and understand why v46–v49 are a single coherent contribution.
  - Each act is supported by at least one quantitative result already in the repo.

---

## 6. Self-Evolution Feedback Loop

The v49 paper story explicitly frames the project as a **self-evolving system** at two levels:

1. **Model-level self-evolution (v37, v39, v43):**
   - The model predicts per-view reliability from its own reprojection residuals (v37).
   - This reliability gates the iterative graph refinement (v39) and scales the per-node residual update (v43).
   - In v46, the same reliability head is reused to decide which views to drop during training and which to retain at inference.

2. **Project-level self-evolution (v46–v49):**
   - Failure analysis on sparse-view validation (v44/v45 A800 runs) identified that the heavy v31–v43 stacks were not improving over v25.
   - v46–v49 therefore strip the architecture back to the v25 geometric foundation and add only lightweight, targeted robustness modules.
   - v49 real-time streaming is the next natural step once v46–v48 show that the simplified model is robust enough for deployment.

This loop—**observe reprojection/failure → update reliability/architecture → retrain → redevaluate**—is the unifying methodological narrative for the paper.

---

## 7. Next Steps

1. Wait for v46–v48 smoke/full results and the v44/v45 A800 decision.
2. Draft `docs/paper_story_v49.md` using the four-act structure above.
3. Populate `docs/related_work_mapping_v49.md` with citations from `docs/literature_review_multiview_pose.md` and the v46–v49 proposals.
4. (Optional) Implement `scripts/generate_related_work_table_v49.py` to keep the LaTeX/Markdown table in sync with the literature review.
5. Review the story with the team before finalizing the ICRA/CVPR 2027 submission outline.
