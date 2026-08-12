# v54 Multi-Scale Geometry Fusion — Risk Report

## 1. Scale-token builders collapse to a single scale

**Risk:** The cross-scale attention gate may learn to ignore one or more scales, or the limb/body/scene MLPs may dominate and wash out joint-level detail, degrading accuracy rather than improving it.

**Mitigation:** Initialize all cross-scale mixing weights to zero (identity at init) so the model starts from the v53 baseline and gradually learns to add scales. Add a small per-scale KL/sparsity penalty in the auxiliary loss to keep the gate distribution informative, and run ablations that drop each scale individually before committing to the full model.

## 2. v52 weight drift breaks triangulation stability

**Risk:** v54 rewrites the v52 uncertainty weights `w^in` into `w^out`. If the weight-refinement head learns to zero out some views or over-weight others, the weighted-DLT stage after v54 could become ill-conditioned, causing NaN/Inf gradients or large MPJPE spikes.

**Mitigation:** Constrain the refined weights with `w^out = w^in * (1 + tanh(g))`, which keeps the multiplier in `[0, 2]`. Enforce a minimum weight `v54_msgf_min_weight=0.05` and add the KL term `KL(w^out || w^in)` so the refined weights stay close to the validated v52 weights. Clip gradients through the weight branch to 1.0.

## 3. Dataset-specific limb grouping harms cross-domain generalization

**Risk:** `v54_msgf_limb_grouping` defaults to H36M's 17-joint layout, but MPI-INF-3DHP (28 joints) and WebBridge/3DPW have different topologies. A hard-coded grouping can map joints to wrong limbs and hurt cross-domain performance.

**Mitigation:** Keep the assignment matrix `A_lj` as a learnable soft mask initialized from the dataset-specific grouping, allowing the model to adapt. Provide a `universal_16` fallback that groups by anatomical name (head, torso, upper/lower arms/legs) rather than joint index, and validate on 3DPW actual-mode before the A800 full run.

## 4. Scene/floor scale over-constrains non-upright captures

**Risk:** The scene token consumes `floor_height` from v53. If the dataset contains non-upright poses, stairs, or sitting subjects, tying the model to the floor plane can push predictions toward an incorrect upright plane and increase MPJPE.

**Mitigation:** Make `v54_msgf_use_v53_floor` optional and default it to `True` only on datasets known to be upright (H36M, MPI-INF-3DHP). Add a per-sample confidence score for the floor estimate and gate the scene-scale contribution with `σ(confidence)`. Smoke-test with floor disabled to establish an upper bound.

## 5. Cross-scale attention adds memory and latency

**Risk:** Running separate cross-view attention for joint, limb, body, and scene scales increases memory by roughly `4×` the base attention footprint and can OOM on A800 with full-batch training.

**Mitigation:** Share MLP parameters across scales where possible, use single-layer per-scale attention (`v54_msgf_n_layers=1`), and pool tokens over the temporal dimension `T` before cross-view attention. Apply stochastic depth with probability `v54_msgf_stochastic_depth=0.1` during training. If OOM persists, drop the scene scale first; it is the cheapest to remove while retaining most of the gain.
