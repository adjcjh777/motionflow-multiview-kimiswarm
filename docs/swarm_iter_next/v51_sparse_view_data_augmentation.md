# v51 Reliability-Guided Sparse-View Mixup (RGSVM)

## Motivation
v46 uniform view dropout exposes the model to 2-, 3-, and 4-view subsets, but every subset is sampled with equal probability. v50 SEFH learns per-view reliability from reprojection, temporal, and epipolar residuals. v51 closes the loop by using those reliability scores to guide sparse-view data augmentation: generate harder, more informative view subsets and reliability-weighted synthetic poses, pushing sparse-view robustness without touching inference architecture.

## Architecture
Add a training-only module `ReliabilityGuidedSparseViewMixupV51` in `motionflow_mv/data/sparse_view_reliability_mixup_v51.py`. After standard v46 dropout, with probability `v51_rgsvm_mixup_prob`:

1. Select two batch samples `A` and `B` sharing the same camera rig.
2. Apply the same sparse subset `S` (size `k`) to both.
3. Triangulate sparse 3D poses `P_A`, `P_B` through the existing v45/v46 triangulation path.
4. Obtain per-view reliability scores `r_A`, `r_B` from the v50 SEFH head (fallback to v46 reliability if v50 is disabled).
5. Compute a reliability-weighted mix weight:
   `w = mean_v σ(r_A,v) / (mean_v σ(r_A,v) + mean_v σ(r_B,v))`.
6. Form `P_mix = w·P_A + (1-w)·P_B`, clamp `w ∈ [0.2, 0.8]`, and reproject `P_mix` to all cameras to synthesize 2D keypoints `K_mix`.
7. Feed `K_mix` into the pipeline; the supervised target is `P_mix`.

The module is identity-at-init: when disabled or when reliability is unavailable, it falls back to standard v46 dropout.

## New config flags

| Flag | Type | Default |
|---|---|---|
| `use_v51_reliability_guided_sparse_view_mixup` | bool | `False` |
| `v51_rgsvm_mixup_prob` | float | `0.25` |
| `v51_rgsvm_min_views` | int | `2` |
| `v51_rgsvm_max_views` | int | `4` |
| `v51_rgsvm_reliability_source` | str | `"v50"` |
| `v51_rgsvm_mix_temperature` | float | `1.0` |
| `v51_rgsvm_hard_subset_boost` | float | `2.0` |
| `v51_rgsvm_consistency_weight` | float | `0.01` |

## Loss term
In addition to the standard supervised pose loss on `P_mix`, add a reliability-consistency term that teaches the model to rank mixed views by their true reprojection error:

`L_rgsvm = λ · (1/|S|) Σ_{v∈S} | r_v - exp(-α · e_v) |^2`

where `e_v` is the reprojection error of `P_mix` in view `v`, `α` is a fixed scale, and `λ` is `v51_rgsvm_consistency_weight`. Only active when `use_v51_reliability_guided_sparse_view_mixup=True`.

## Evaluation metric
Report `MPJPE@k` for `k = 2, 3, 4, full` via `experiments/eval_variable_views.py`. Also track Spearman(predicted reliability, reprojection error) on mixed samples and the fraction of "hard" subsets whose `MPJPE@2` improves after mixing.

## Expected MPJPE impact
- `MPJPE@2`: −3 to −5 mm vs. v46 baseline.
- `MPJPE@3`: −2 to −3 mm.
- `MPJPE@full`: ±0.5 mm (augmentation is training-only).
- 3DPW actual `MPJPE@2`: −4 to −6 mm by densifying exposure to rare sparse subsets.

## Main risk
Mixing two arbitrary poses can yield geometrically implausible `P_mix` and noisy gradients if `A` and `B` differ widely. Mitigation: clamp `w` to `[0.2, 0.8]`; skip mixup when `MPJPE(A,B) > 200 mm`; start smoke with `v51_rgsvm_mixup_prob=0.1`; and freeze base model weights for the first epoch so only the augmentation distribution updates.
