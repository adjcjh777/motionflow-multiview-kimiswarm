# Bone-length &amp; skeleton consistency losses

**Summary**

Bone-length and skeleton-consistency losses are already implemented in `experiments/train_utils.py`, but they are not used by the current best model `RayAttentionFusionModelTemporalResidual`, which was trained without them and holds the 11.17 mm MPI-INF-3DHP record. The opportunity is to add a properly weighted supervised bone-length and an unsupervised temporal bone-length consistency loss to the residual trainer, using the existing MPI-INF-3DHP 28-joint parent array.

**Report written:** `docs/swarm_iter7/bone_length_skeleton_consistency_loss.md`

The report includes: current state, gap/opportunity, a concrete next step (create `experiments/train_ray_attention_temporal_residual_aux_mpiinf3dhp.py` and run a 2-epoch smoke), expected success metrics (MPJPE < 11.17 mm), and risks/blockers (A800-D read-only, weight sensitivity, skeleton layout mismatch).