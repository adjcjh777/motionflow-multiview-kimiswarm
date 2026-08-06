## Summary

I investigated **outlier-robust training losses** for the current best `RayAttentionFusionModelTemporalResidual` architecture.

**Key findings:**
- Every residual trainer (`experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py`, `..._curriculum.py`, H36M variants) still uses plain `nn.MSELoss()` for 3D pose regression.
- Robustness is currently handled only by data augmentation (`augment_clip` adds noise/dropout/outliers) and the attention weighting, not by the loss function itself.
- Auxiliary losses exist in `experiments/train_utils.py` (bone-length, skeleton consistency) and the uncertainty model has a reprojection NLL loss, but there is no robust 3D loss module.

**Recommendation:** Add a reusable `motionflow_mv/eval/robust_losses.py` module and fork the MPI-INF-3DHP trainer to compare MSE vs. Huber/Charbonnier/Geman–McClure losses, starting with Huber because it is a one-line PyTorch replacement. Run a smoke comparison, then a full 5-epoch benchmark if it helps.

**Report written:** `docs/swarm_iter7/outlier_robust_training_losses.md`