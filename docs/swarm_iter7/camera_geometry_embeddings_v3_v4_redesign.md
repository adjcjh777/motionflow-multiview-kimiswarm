# Camera-geometry embeddings (v3/v4 redesign)

Investigation complete. I traced the camera-embedding story from the raw `K/R/t` token in the current-best temporal-residual model through the v3 domain-agnostic and v4 normalised variants, and found that none has been shown to actually improve over the baseline on real data. The report proposes a redesign around pairwise camera-geometry embeddings with a controlled ablation.

Report written to: `docs/swarm_iter7/camera_geometry_embeddings_v3_v4_redesign.md`