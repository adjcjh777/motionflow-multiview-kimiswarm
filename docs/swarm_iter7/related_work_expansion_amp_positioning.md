# Related work expansion &amp; positioning

I investigated the related-work landscape for the MotionFlow-MultiView paper. My findings and the report content are below. Note that this subagent role is read-only and does not include file-editing tools, so I could not write the file directly; the markdown report should be saved to `docs/swarm_iter7/related_work_expansion_positioning.md`.

---

## Summary

- **Current state:** `docs/paper_draft_icra_cvpr_2027.md` has a very thin Related Work section (lines 19–26, only 4 paragraphs and 7 references, several placeholder/incorrect). Rich prior surveys exist in `docs/swarm_iter4/` and `docs/swarm_iter5/` but are not integrated.
- **Gap:** The paper lacks a structured positioning against classical triangulation, recent learned multi-view fusion (MV-SSM, RUMPL, COMPOSE, etc.), temporal/residual pose models, and robotics-oriented systems.
- **Next step:** Expand the Related Work section into four subsections with 15+ properly cited references, replacing placeholder citations.
- **Success metric:** A ≥1-page, well-cited Related Work section with clear one-sentence positioning for each cited method.
- **Risk:** Need to verify arXiv pre-print venues before camera-ready.

---

### Report content to save as `docs/swarm_iter7/related_work_expansion_positioning.md`

```markdown
# Related Work Expansion & Positioning

## 1. Current state

The paper draft at `docs/paper_draft_icra_cvpr_2027.md` contains a minimal Related Work section (lines 19–26): only four short paragraphs and seven references, several of which are placeholder or incorrectly formatted (e.g. “Ray-attention multi-view pose. CVPR, 2022”). It does not yet position the project against the 2025–2026 multi-view pose literature.

Existing literature surveys in the repo that can be mined for this expansion include:

- `docs/swarm_iter4/related_work_voxelpose_rpsnet_mvmesh_epipolarpose_metrabs.md` — VoxelPose, EpipolarPose, MeTRAbs positioning.
- `docs/swarm_iter5/literature_gap.md` — CVPR/ICRA 2025–2026 gap analysis (MV-SSM, RUMPL, DisPOSE, COMPOSE, etc.).
- `docs/swarm_iter5/literature_gap_summary.md` — condensed take-aways from the above.
- `docs/swarm_iter5/literature_survey.md` — self-evolution/reward-weighted angle and concrete experiment ideas.
- `docs/swarm_iter4/cvpr_2027_positioning.md` — positioning as a systems / modular fusion paper.

The codebase also has the technical artefacts needed to anchor citations:

- `motionflow_mv/fusion/ray_attention_model.py` — ray-aware attention + weighted DLT.
- `motionflow_mv/fusion/ray_attention_temporal_residual_model.py:38` — `RayAttentionFusionModelTemporalResidual`, the residual refinement head.
- `motionflow_mv/fusion/ray_attention_temporal_uncertainty_model.py` — uncertainty-aware variant.
- `motionflow_mv/fusion/ray_attention_temporal_crossview_model.py` — cross-view variant.

## 2. Gap / opportunity

The Related Work section needs to be expanded from a placeholder into a structured, one-page narrative that situates the project in four strands:

1. **Classical triangulation** (DLT, RANSAC, confidence-weighted DLT) — establishes the geometric baseline this project builds on.
2. **Learned multi-view fusion** (Iskakov et al. ICCV 2019; MV-SSM, RUMPL, COMPOSE, etc.) — shows the shift from direct 3D regression to geometry-aware learned fusion.
3. **Temporal and residual pose models** (MotionBERT, SmoothNet, residual refiners) — justifies the temporal + residual decomposition.
4. **Systems / robotics-oriented pose pipelines** (MotionFlow, HumanMotionIR, VoxelPose-style volumetric methods) — supports the ICRA plug-in / metric-scale argument.

This expansion is an opportunity to sharpen the paper’s contribution claim: *post-triangulation residual refinement on top of temporal ray-attention fusion*, rather than “another attention fusion model.”

## 3. Concrete next step

**Rewrite and expand the Related Work section in `docs/paper_draft_icra_cvpr_2027.md` (currently lines 19–26) into a full ~1-page section with the four subsections above, using 15+ properly cited references drawn from the existing swarm literature survey files.**

Specific deliverables:

- Replace placeholder citations (e.g. “Ray-attention multi-view pose. CVPR, 2022”) with real references.
- Add one sentence of positioning per cited method explaining the difference from this work.
- Cite the project’s own prior survey files where appropriate so reviewers can trace the literature audit.
- Add a short paragraph explicitly stating the paper’s novelty relative to the four strands (geometric triangulation + learned fusion + temporal/residual + systems).

Optionally, create `docs/references.bib` with the same 15+ entries so the draft can be compiled to a LaTeX bibliography later; this avoids duplicate entry typing and keeps the draft and any future LaTeX version synchronized.

## 4. Expected success metric

- The Related Work section is at least one page long and contains ≥15 unique, correctly formatted citations.
- Each of the four strands above is represented.
- No placeholder references remain.
- A reviewer can read the section and clearly state what is novel about this paper (temporal ray-attention + per-joint residual refinement + plug-in metric HumanMotionIR output).

## 5. Risks / blockers

- **Citation verification:** several 2025–2026 papers listed in `literature_gap.md` are arXiv pre-prints (RUMPL, DisPOSE, COMPOSE, etc.). Before camera-ready, each must be verified against its final venue or replaced with a peer-reviewed equivalent.
- **Scope creep:** expanding Related Work can turn into a full literature review. Keep it tightly focused on the four strands that directly justify the paper’s design choices.
- **No A800-D / Docker impact:** this is a documentation task; read-only A800-D and Docker constraints do not apply.
- **No large-file commits:** this work only touches Markdown/BibTeX files; no data downloads or large binary commits are needed.
```

**Path to report:** `docs/swarm_iter7/related_work_expansion_positioning.md` (needs to be written by the main agent; I do not have file-editing tools in this read-only role).