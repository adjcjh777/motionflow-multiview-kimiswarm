# v54 Physical-Space Calibration v2 — Risk Report

This document lists the main technical risks for the proposed `physical_space_calibration_v2_v54` module and concrete mitigations for each.

## Risk 1: Skeleton-graph refiner over-smooths anatomical detail

**Description:** The GNN residual refiner propagates information along bone edges. If the edge attention or gate grows too large, the network may collapse distinct joint positions toward a smoothed skeleton, hurting wrists/ankles and raising MPJPE on articulated poses.

**Mitigation:**
- Keep the graph shallow (`v54_psc2_gnn_layers=1`) and hidden dim small (`v54_psc2_hidden=64`).
- Zero-initialize the final output projection and initialize the residual gate to `-6.0` (`σ ≈ 0.0025`), so the correction starts at zero and grows only as the loss gradients demand it.
- Clamp the per-joint residual to a small bounding box (e.g., `±200 mm`) during the first epoch as an additional safety rail.

## Risk 2: Floor/contact constraints hurt non-upright or airborne motion

**Description:** PSC-v2 assumes that feet near the floor should touch it. For jumping, lying, stairs, or sitting sequences, forcing floor contact can pull valid poses toward a ground plane and increase error.

**Mitigation:**
- Use a **soft** floor loss that penalizes feet only below the estimated plane, not above it.
- Gate the contact loss by foot velocity (`v54_psc2_contact_velocity_thresh`) so static frames are regularized while fast-moving frames are left alone.
- Make the floor head optional via `v54_psc2_use_floor` and start with a low weight (`v54_psc2_floor_weight=0.01`).

## Risk 3: Canonical bone scales conflict across datasets

**Description:** v54 learns per-domain canonical bone log-scales. If a dataset has a different skeleton (e.g., MPI-INF-3DHP 28 joints vs. H36M 17 joints) or unusual subject proportions, the learned scale can bias the pose in the wrong direction.

**Mitigation:**
- Initialize scales to zero (identity) and keep the bone loss weight small at first; let the network ignore the prior when residuals are large.
- Use `domain_id` to select per-domain scales only when `v48_domain_generalization` is enabled; otherwise share a single global scale.
- In `forward`, mask out bones whose parent or child joint is not visible in at least `v54_psc2_min_visible_views` views.

## Risk 4: Identity-at-init fails and v53 checkpoints regress

**Description:** If the GNN output layer, the residual MLP, or the bone log-scales are not initialized correctly, loading a v53 checkpoint with v54 enabled could change predictions by more than the targeted `< 0.1 mm`, breaking warm-start compatibility.

**Mitigation:**
- Zero-initialize the final linear layer of the graph refiner and the bone-scale MLP output.
- Initialize the bone log-scales to zeros and the residual gate logit to `-6.0`.
- Add a unit test that loads a v53 checkpoint, enables v54, runs one forward pass, and asserts `||pred_v54 - pred_v53||_∞ < 1e-4 mm`.

## Risk 5: Extra compute from the GNN and temporal terms hurts throughput

**Description:** PSC-v2 adds a skeleton-graph convolution over `J` joints across `T` time steps plus extra physical losses. On the local RTX 4090 smoke run this could cause OOM or slow iterations, especially when stacked with v50/v51/v52/v53.

**Mitigation:**
- Keep the graph operation sparse: only parent-child edges are used, so the adjacency matrix has at most `J-1` non-zero entries.
- Compute floor/bone/reprojection losses on a down-sampled temporal stride during training if memory becomes tight; the full sequence is still used at inference.
- Set `v54_psc2_use_gnn=false` as a fallback to a simple per-joint MLP refiner for quick ablations.
