# Longer temporal context (&gt;13 frames)

**Summary**

Investigated the “Longer temporal context (>13 frames)” direction for `RayAttentionFusionModelTemporalResidual`.

- **Current state:** The best model uses `clip_len=13` and reaches 11.17 mm / 8.24 mm / AUC 0.9256 on MPI-INF-3DHP. The architecture already supports up to `max_temporal_len=256` (`ray_attention_temporal_model.py:110`). A 1-epoch `clip_len=27` smoke run produced 27.15 mm and is not representative.
- **Gap:** No full 5-epoch training/evaluation has been run for `clip_len > 13`.
- **Opportunity:** A properly trained 27-frame model could capture more motion context, reducing jitter and occlusion errors.
- **Next step:** Run the full `clip_len=27` train/eval using the existing scripts and model; no core code changes needed.
- **Expected success:** MPI-INF-3DHP MPJPE < 11.17 mm, ideally 9.5–10.5 mm; PA-MPJPE < 8.24 mm; AUC > 0.9256.
- **Risks:** O(T²) memory (mitigate with `batch_size=4`), diminishing returns, overfitting on small MPI-INF-3DHP, and read-only A800-D constraints.

**Report written to:** `docs/swarm_iter7/longer_temporal_window.md`