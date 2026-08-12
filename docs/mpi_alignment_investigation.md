# MPI-INF-3DHP Detected-2D Alignment Investigation (S1/Seq1)

## Summary

The MPI-INF-3DHP `detected_2d` .npz files share **identical camera parameters and 3D labels** with the original `data/webbridge/mpi_inf_3dhp/` canonical files, so the 326–400 mm DLT error is not a camera or label mismatch. The error is caused by the 2D detections themselves: the current MediaPipe Pose-based detections use an approximate 28-joint → MPI skeleton mapping and are run at a low input resolution (384×384) on 2048×2048 wide-angle video frames. This produces large reprojection errors, especially for the lower body and hands, which triangulates to ~330–460 mm MPJPE.

| Dataset | S1/Seq1 DLT MPJPE (unweighted) | S1/Seq1 DLT MPJPE (confidence-weighted) |
| --- | --- | --- |
| `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz` | **34.8 mm** | — |
| `data/webbridge/mpi_inf_3dhp_detected_2d/s_01_seq_01_v14_multiview_m.npz` | **330.7 mm** | **462.6 mm** |

## Files Compared

- **Original:** `data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz`
- **Detected-2D:** `data/webbridge/mpi_inf_3dhp_detected_2d/s_01_seq_01_v14_multiview_m.npz`

Both files have shape `(T=6416, V=14, J=28)` and the canonical WebBridge keys: `points_2d`, `confidences`, `joints_3d`, `camera_K`, `camera_R`, `camera_t`.

## Camera Parameters and 3D Labels

The camera matrices and 3D labels are **byte-for-byte identical** between the two files:

| Array | Max absolute difference | Notes |
| --- | --- | --- |
| `camera_K` | `0.0` | 14 intrinsics, identical |
| `camera_R` | `0.0` | 14 rotations matrices, identical |
| `camera_t` | `0.0` | 14 translation vectors, identical |
| `joints_3d` | `0.0` | Ground truth unchanged |
| `confidences` | `1.0` | Detected file uses MediaPipe visibility for mapped joints and `0.0` for unmapped joints (original is all `1.0`) |

Conclusion: the 326–400 mm triangulation error is not caused by different calibration or labels.

## 2D Point Differences

```text
points_2d max abs diff : 4061.96 px
points_2d mean abs diff:   86.04 px
points_2d median diff  :   18.43 px
fraction exactly equal :    0.441
```

- 44.1% of the 2D points are exactly equal to the original GT points. These are the unmapped joints (spine, neck, clavicles) plus failed detections, for which the generator falls back to GT.
- The remaining 55.9% are MediaPipe detections for the mapped joints, and they differ from the GT projection of the 3D labels.

## Per-Joint Reprojection Error

Projecting the true 3D labels through the shared cameras and comparing to the detected 2D points (sample: first 500 frames) gives the following mean per-joint reprojection error:

| Joint | Mean px error | Joint | Mean px error |
| --- | --- | --- | --- |
| `left_ankle`  (20) | **307.8** | `right_ankle`  (25) | **156.1** |
| `left_knee`   (19) | **248.4** | `right_wrist`  (16) | **153.5** |
| `right_knee`  (24) | **238.5** | `left_wrist`   (11) | **147.4** |
| `left_hip`    (18) | **187.6** | `left_elbow`   (10) | **117.3** |
| `right_hip`   (23) | **184.3** | `head_top`     (7)  | **82.8** |
| `left_foot`   (21) | **175.1** | `right_shoulder`(14)| **70.4** |
| `left_hand`   (12) | **165.3** | `left_shoulder`(9)  | **76.4** |
| `right_hand`  (17) | **165.0** | `head`         (6)  | **29.9** |

Unmapped torso joints (`spine*`, `pelvis`, `neck`, `clavicles`) have <10 px error because they are copied from GT.

## Root Cause

The large DLT error has three contributing factors:

1. **Approximate skeleton mapping.** `scripts/generate_mpi_detected_2d_from_avi.py` maps MediaPipe’s 33 landmarks to MPI’s 28 joints through a hand-coded dictionary (`MEDIAPIPE_TO_MPI`). Distal joints — especially ankle/foot/toe and hand/wrist — do not map cleanly, so the same MediaPipe landmark is reused for multiple MPI joints or mapped to an anatomically different location.
2. **Low detection resolution.** MediaPipe Pose is run on a 384×384 resized image. The original video frames are 2048×2048. Fine-grained joints such as ankles, toes, and wrists are localized imprecisely after the heavy downsample.
3. **Confidence weighting exposes the mismatch.** The detected file stores high confidence for the mapped joints and `0.0` for unmapped/fallback joints. When DLT uses those confidences, it trusts the noisy mapped detections and ignores the accurate GT fallback joints, pushing the triangulation error above 460 mm. Unweighted DLT is lower (330 mm) because it also benefits from the copied GT torso points.

## Fix Plan

### Immediate (diagnostic / protocol)

1. **Do not use the current `detected_2d` .npz for model selection.** Treat it as a protocol-development placeholder only.
2. **Use the confidence-weighted DLT baseline as the gate.** A usable real-2D MPI-INF-3DHP dataset should achieve per-sequence DLT MPJPE below ~100 mm (comparable to H36M detected-2D baselines).
3. **Add a per-joint reprojection check to the generation pipeline.** Reject or flag any sequence whose detected 2D points reproject with >50 px per-joint mean error before writing the .npz.

### Short-term (data regeneration)

4. **Regenerate with an MPI-INF-3DHP-specific detector.** The standard literature uses CPN or HRNet trained/evaluated on MPI-INF-3DHP. These detectors output the dataset’s native 28-joint skeleton, avoiding the MediaPipe mapping mismatch. The project already documents this in `docs/mpi_detected_2d_protocol.md`.
5. **If MediaPipe must be used:**
   - Increase `detect_size` to the full frame resolution (2048) or at least 1024, so landmarks are not downsampled.
   - Revise `MEDIAPIPE_TO_MPI` using the official MPI-INF-3DHP joint definitions; map only landmarks that have a clear anatomical correspondent and leave the rest as GT with confidence `0.0`.
   - Replace the current `l.x * w` scaling with `l.x * detect_size` if the MediaPipe tasks API returns normalized coordinates relative to the resized input; verify by checking the ratio against GT on a few frames.

### Medium-term (model training)

6. **Use confidence-aware triangulation/loss.** Pass the generated `confidences` array through the training pipeline so low-confidence/unmapped joints are not treated as reliable observations.
7. **Add a 2D reprojection consistency loss** during training so the network learns to recover 3D poses that are consistent with the detected 2D points, even when those detections are noisy.

## Commands for Reproduction

```bash
# Original DLT baseline
python scripts/run_mpi_dlt_baseline.py \
    --glob "data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz" \
    --output tmp/mpi_orig_dlt.json

# Detected-2D DLT baseline (unweighted gives the 326-400 mm range)
python scripts/run_mpi_dlt_baseline.py \
    --glob "data/webbridge/mpi_inf_3dhp_detected_2d/s_01_seq_01_v14_multiview_m.npz" \
    --unweighted \
    --output tmp/mpi_det_dlt_unw.json

# Detected-2D DLT baseline (confidence-weighted)
python scripts/run_mpi_dlt_baseline.py \
    --glob "data/webbridge/mpi_inf_3dhp_detected_2d/s_01_seq_01_v14_multiview_m.npz" \
    --output tmp/mpi_det_dlt.json
```

## Recommended Next Step

Start item 4: re-run `scripts/fetch_mpi_real_2d.sh` after replacing the MediaPipe backend with an MPI-INF-3DHP-trained CPN/HRNet detector that emits the exact 28-joint layout. Until that is available, keep the current `detected_2d` files for CPU-only smoke tests only.
