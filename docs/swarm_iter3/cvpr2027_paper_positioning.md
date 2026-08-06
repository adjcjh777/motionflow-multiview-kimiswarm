Completed. I researched the repository and wrote the CVPR/ICRA 2027 positioning report to:

**`docs/swarm_iter3/cvpr2027_paper_positioning.md`**

**Key findings:**
- The project has a strong modular skeleton (`HumanMotionIR`, `FusionModule`, `MultiViewAdapter`, GVHMR adapter) and a robust DLT baseline (~9.9 px mean reprojection on Shelf 300–600).
- All learned variants trained on reprojection loss or DLT pseudo-GT only match DLT; they do not beat it.
- The literature (MVGFormer, MV-SSM, ScoreHMR, Bragagnolo et al., COMETH) points toward **hybrid geometry-learning fusion**, **3D-supervised training**, and **physics-aware refinement** as the path forward.

**Main recommendation:**
Frame the 2027 paper as a **modular, uncertainty-aware, geometry-learning fusion framework** rather than a “better triangulator.” Critical next steps are:
1. Obtain real 3D GT (Human3.6M) or high-quality synthetic AMASS data.
2. Train with combined 3D + reprojection + bone-length + temporal losses.
3. Upgrade `AttentionFusionModelV2` with ray/plane geometry encoding (not flattened projection matrices).
4. Integrate ScoreHMR as a per-view plugin and pseudo-3D label source.
5. Populate `HumanMotionIR.uncertainty` fields and add downstream robot validation for ICRA.

The full report includes a problem statement, related-work table, codebase mapping, implementation priorities, datasets/baselines, and a risk register.