# GPU Queue Manifest (single RTX 4090, sequential)

Generated after Swarm Iteration 7. The current curriculum training must finish before any item below starts.

## Running now

1. **Cross-view PP curriculum + view dropout** (`scripts/run_crossview_pp_curriculum_wsl.sh`)
   - Output: `outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth`
   - On completion: run clean + robustness final eval, then start item 2.

## Queued (already launched)

2. **Visibility-gated fusion v2** (`scripts/run_crossview_residual_visibility_v2_wsl.sh`)
   - Warm-start from best PP checkpoint with `view_dropout_rate=0.2` and `visibility_loss_weight=0.1`.
   - Goal: clean ≤ 9.6 mm, ≥ 10% relative gain at 30% occlusion.

3. **SSL pre-training on H36M** (`scripts/run_ssl_pretrain_h36m_full_wsl.sh`)
   - Masked-view reprojection pre-training; then fine-tune on MPI.
   - Goal: quantify pre-training → fine-tuning data efficiency.

4. **Spatiotemporal (T×V×J) PP model** (`scripts/run_spatiotemporal_principal_point_wsl.sh`)
   - Factorized T×V×J attention over principal-point-corrected rays.
   - Goal: clean < 9.0 mm with moderate compute increase.

## Proposed next (add to queue after item 4)

5. **Focal self-calibration + stronger extrinsic curriculum**
   - Launcher: `scripts/run_focal_selfcalib_0p02_wsl.sh` (from Swarm Iteration 7, direction 11).
   - Goal: rot_0.5° < 12 mm, focal_1% < 14 mm.

6. **Temporal consistency / velocity loss**
   - Launcher: `scripts/run_temporal_velocity_longclip_wsl.sh` (from Swarm Iteration 7, direction 5).
   - Goal: smooth predictions without clean-MPJPE regression.

## Optional / lower priority

7. **Cross-dataset domain adaptation** (H36M → MPI / MPI → H36M).
8. **Action-aware PP model** (H36M action labels).
9. **Graph joint-relation fusion** (drop-in GNN skeleton module).
10. **Multi-scale spatial pyramid PP model**.

## Notes

- Do not start items 5+ until items 1–4 have completed; GPU is single-threaded.
- Each experiment must produce `manifest.json` and update `docs/tables/results_generated.md`.
- Results feed back into the next swarm iteration.
