# v53 Multi-Scale Geometry Fusion — Risk Report

## 1. Dataset-specific skeleton groupings

**Risk:** The limb-scale token builder needs a fixed limb grouping. H36M (17 joints) and MPI-INF-3DHP (28 joints) have different skeletons; a hard-coded H36M grouping will fail or degrade on MPI and 3DPW.

**Mitigation:**
- Provide at least two groupings (`h36m_17_limbs`, `mpi_28_limbs`) selected by config flag.
- Add a fallback universal grouping based on a generic 17-joint CMU skeleton and a nearest-joint map.
- Unit-test limb pooling on both H36M and MPI sample shapes before training.

## 2. Multi-scale module overfits to the training skeleton

**Risk:** Even with the correct grouping, the cross-scale attention may learn pose-specific shortcuts that do not transfer across datasets (e.g., H36M poses are more upright than in-the-wild 3DPW).

**Mitigation:**
- Use dropout on the cross-scale attention (`v53_msgf_dropout` default 0.1).
- Freeze the module for the first epoch or use loss weight warmup (`v53_msgf_loss_warmup_epochs`).
- Evaluate on 3DPW actual-mode validation immediately after smoke to catch over-transfer.

## 3. Memory and compute overhead from multiple attention layers

**Risk:** Running separate cross-view attention for joint, limb, and body scales multiplies memory and runtime, especially for high `T` or many views. This can OOM on the RTX 4090 smoke or slow A800 training.

**Mitigation:**
- Share the same multi-head attention parameters across scales (joint/limb/body tokens differ, but the attention block is shared).
- Default to small hidden sizes (`v53_msgf_hidden=64`, `v53_msgf_n_layers=2`).
- Benchmark memory with `torch.cuda.memory_summary` in smoke and cap `clip_len` if needed.

## 4. Identity-at-init can mask a broken module

**Risk:** If the final layers are zero-initialized and the residual gate starts at 0, the module passes the v52 result unchanged. A bug in the cross-scale attention or a missing connection may therefore not be detected until training diverges.

**Mitigation:**
- Add a dedicated unit test that disables zero-initialization, runs a small forward pass, and checks the gradients flow back through the limb/body branches.
- Smoke train for 10 steps and verify the loss decreases and the refined weights differ from `w_in` by more than 1e-4.
- Compare the v53 output against a hand-computed two-view toy example.

## 5. Weight refinement destabilizes v52 triangulation

**Risk:** v52 already learns per-view precision weights. Adding another learned multiplier on top can create a feedback loop: v53 may suppress a view that v52 already down-weighted, or amplify a noisy view during early training, hurting triangulation and causing NaN weights.

**Mitigation:**
- Clamp the multiplier: `1 + tanh(g)` is bounded in `[0, 2]`, so each view can at most double or zero its weight.
- Add a small entropy term in the auxiliary loss to discourage degenerate all-zero weights per joint.
- Use `v53_msgf_loss_weight` warmup so the module has little influence until after v52 has stabilized.
