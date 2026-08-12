# Task Snippet: Paper Outline and Contribution Framing

**Agent task:** Write `docs/swarm_iter5/paper_outline_icra_cvpr.md` with a 6-page outline, key claims, and expected tables/figures for the ICRA/CVPR 2027 submission.

**What was produced:**

- `docs/swarm_iter5/paper_outline_icra_cvpr.md`
  - 6-page camera-ready-style outline (Introduction → Related Work → Method → Experiments → Results → Conclusion).
  - Abstract (~150 words) framing the ray-aware attention fusion contribution.
  - Section-by-section narrative grounded in `motionflow_mv/fusion/ray_attention_v3_model.py` and `experiments/train_ray_attention_v3_h36m.py`.
  - Five key claims: geometry-aware attention beats direct regression, camera-conditioned embeddings improve cross-dataset generalization, per-view weights provide interpretable robustness, the module is a drop-in DLT replacement, and large 3D-supervised training is necessary to beat DLT on clean real data.
  - Four expected tables (H36M main results, synthetic robustness, cross-dataset zero-shot, architectural ablations).
  - Six expected figures (architecture, ray geometry + weights, synthetic robustness bar chart, H36M training curves, MotionFlow pipeline qualitative, ablation figure).
  - Submission venue/timeline and risk register.

**Important findings / constraints surfaced:**

- The current best model is `ray_attention_v3` (camera-conditioned embeddings + view/joint attention + differentiable weighted DLT).
- Training is running on 62 k frames of Human3.6M (`s_01_acts_02_..._16_multiview.npz`) with DLT pseudo-GT.
- The paper must be framed around **robustness** and **modular integration**, not just raw accuracy, because DLT is already near-optimal on clean real data.
- Real 3D GT (true Human3.6M) and raw Shelf/Campus are still a data-access risk; the outline includes a fallback to synthetic + pseudo-GT experiments and a modular-systems contribution.
- A800-D remains read-only; all training/experiments stay on the local WSL 4090.

**Not started / next steps:**

- Populate Table 1 and Figure 4 with actual H36M training results once `experiments/train_ray_attention_v3_h36m.py` finishes.
- Run `ray_attention_v3` on Shelf/Campus to fill Table 3 (cross-dataset generalization).
- Generate the architecture and ablation figures (matplotlib or TikZ).
