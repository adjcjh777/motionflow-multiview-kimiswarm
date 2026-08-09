# v45 Next Iteration Plan

This document proposes three candidate multi-view architecture variants for **v45**, conditionally following the v44 decision outcome documented in `docs/v44_decision_plan.md`. It is intended for the ICRA/CVPR 2027 multi-view pose paper story.

## Context and assumptions

Per `docs/results_snapshot_2026_08_09.md` and `docs/v44_decision_plan.md`:

- **v25 geometry fusion** is the strongest known baseline (17.17 mm on A800).
- **v31-v34 complex stacks** (graph, attention, temporal, outlier rejection, etc.) are currently 26-37 mm and lag v25.
- **v44 is most likely** a v25-based architecture, either:
  - **v44a**: plain v25 with larger capacity / longer training / SWA, or
  - **v44b**: v25 + skeleton-aware physical loss + domain-balanced training (already queued).

All v45 variants below assume v44 is a **v25-based baseline**. If v44 instead refines a v42/v43 complex stack, the same ideas can be transplanted onto that stack.

---

## Variant 1: v45-adaptive-geometry-fusion (v45-AGF)

### Motivation

Current v25 uses a fixed geometry-fusion weighting for multi-view triangulation. v45-AGF introduces **learnable, data-driven per-joint / per-view weights** for the triangulation step, so the model can down-weight noisy views and up-weight reliable ones without committing to the heavy graph stacks of v33-v34.

### Key changes

1. Add a lightweight **view-reliability head** (one MLP per view) operating on v25 fused features.
2. Use the predicted reliabilities as weights in a weighted DLT / triangulation step.
3. Keep the rest of the v25/v44 architecture unchanged to isolate the triangulation contribution.

### Expected experiments

- Smoke on RTX 4090: clip_len=9, d=64, train_samples=500, 2 epochs.
- Full A800 run: d=128, n_st_layers=3, full WebBridge/H36M/MPI manifest.
- Ablations:
  - per-joint weights vs. per-view weights vs. per-(view,joint) weights;
  - learned weights vs. vanilla uncertainty-weighted triangulation (v33 baseline);
  - with and without physical/domain additions from v44.

### Success criterion

Beat v44 (or v25) by >3% relative in MPJPE, or match it with fewer views / stronger robustness to missing views.

---

## Variant 2: v45-sparse-view-generalization (v45-SVG)

### Motivation

All current runs train with a fixed number of camera views. v45-SVG explicitly trains the model on **variable, sometimes sparse, camera subsets**, so the final system generalizes to fewer or different camera configurations. This is high-value for practical deployment and complements v25's strong geometry-fusion mechanism.

### Key changes

1. Data loader augmentation: randomly drop 1-2 views per training clip with probability p_drop (e.g., 0.3).
2. Add a **view-agnostic token / mask** so the transformer encoder handles varying view counts.
3. Use the same triangulation objective but with adaptive normalization by the number of available views.

### Expected experiments

- Smoke on RTX 4090 with variable-view training (drop 0-2 views).
- Full A800 run with the same config as v44.
- Robustness evaluation:
  - fixed 2-view, 3-view, and 4-view subsets on the standard val set;
  - compare against the same v44 model trained without view dropout.

### Success criterion

No worse than v44 at full views, and at least 10% better MPJPE on 2-view/3-view subsets.

---

## Variant 3: v45-temporal-geometry-aggregation (v45-TGA)

### Motivation

v25 fuses geometry within a single frame. v45-TGA extends this to **short temporal clips** by aggregating triangulated 3D joints across frames, exploiting temporal smoothness without the expensive per-frame graph stacks that hurt v31-v34. This is a minimal addition to the v25 architecture.

### Key changes

1. Extend the temporal receptive field to clip_len=13 (from current 9).
2. Add a small **temporal geometry attention module** (2-4 layers, shared across joints) that refines triangulated 3D positions using neighboring frames.
3. Optional: add a lightweight temporal physical consistency loss (bone length smoothness + anti-jitter) from v40.

### Expected experiments

- Smoke on RTX 4090: clip_len=13, d=64, train_samples=500.
- Full A800 run: d=128, n_st_layers=3, clip_len=13.
- Ablations:
  - temporal module depth (1, 2, or 4 layers);
  - with and without temporal physical loss;
  - causal vs. non-causal temporal aggregation.

### Success criterion

Improve over v44 in MPJPE, or maintain MPJPE while reducing per-frame jitter (e.g., 10% lower acceleration error).

---

## Recommended order

1. **First**: v45-AGF — smallest change, directly addresses a v25 limitation, and can borrow uncertainty ideas from v33 without the full stack.
2. **Second**: v45-SVG — high practical value, orthogonal to AGF, and could be combined with AGF in a follow-up.
3. **Third**: v45-TGA — more involved; run only if temporal smoothness is identified as a remaining gap in v44/v45-AGF.

## Follow-up combinations

If individual v45 variants succeed, a **v46 combined** architecture would stack:

- v25/v44 base geometry fusion
- v45-AGF learned triangulation weights
- v45-SVG variable-view training
- v45-TGA temporal aggregation + temporal physical loss

This combined system should be the target end-to-end architecture for the paper.
