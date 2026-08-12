# Multi-person association

## Problem statement

The MotionFlow-MultiView pipeline is currently single-person: loaders such as
`motionflow_mv/data/shelf_loader.py` and `webbridge_loader.py` take a single
`person_id`, and `motionflow_mv/fusion/ray_attention_model.py` consumes a tensor
of shape `(B, V, J, 3)` for one skeleton. Real multi-camera scenes (Shelf,
Campus, Panoptic) contain several people, so a practical system must group
per-view detections into per-person tracks before/while triangulating them. The
current gap is not the fusion head itself, but **data loading, cross-view
association, and multi-person evaluation**.

## Simplest concrete next step

Run and extend the existing CPU smoke test
`experiments/associate_multi_person_synthetic.py`. It already generates two
synthetic people, shuffles the per-view detections, and recovers the correct
identity labeling by minimizing multi-view reprojection error. The next step is to
turn this 2-person brute-force proof into an **M-person Hungarian-assignment
baseline** that can be evaluated under noise, occlusion, and false positives.

No GPU or training is required; the work is purely CPU geometry and synthetic
benchmarking.

## Files to touch and rough diff/sketch

1. `experiments/associate_multi_person_synthetic.py` – generalize from `M=2` to
   variable `M`, replace the `2^V` label brute-force with:
   - a per-frame cost between two detections in different views (mean epipolar
     distance over all joints, or reprojection residual after triangulating a
     probe pair);
   - a greedy/hungarian solver that picks `M` consistent tuples of detections, one
     per person per view.

2. `motionflow_mv/data/synthetic_3d_dataset.py` – re-use the camera/projection
   helpers already imported by the script; no change needed unless SMPL motion
   for multiple people is required.

3. (Optional) `motionflow_mv/eval/metrics.py` – add an simple
   `association_accuracy(pred_labels, gt_labels)` helper that allows a global
   identity swap.

Sketch of the new association core:

```python
from scipy.optimize import linear_sum_assignment
import itertools

def associate_detections_m_person(detections, proj_matrices, n_people):
    """
    detections: (T, V, M, J, 2) unordered detections.
    proj_matrices: (V, 3, 4)
    Returns: labels (T, V, M) assigning each view slot to a global person id.
    """
    T, V, M, J, _ = detections.shape
    labels = np.zeros((T, V, M), dtype=int)
    # Cost matrix: for each pair of view slots, score by triangulation residual.
    for t in range(T):
        # Simplified: use first two views to seed tracks, then assign remaining
        # views with Hungarian over reprojection residuals.
        ...
    return labels
```

The existing single-person fusion model (`RayAttentionFusionModel`) can then be
applied to each associated person independently; no model retraining is needed for
this CPU association step.

## Expected success metric

- Association accuracy ≥ 90 % on clean synthetic 2-person scenes (already met by
  the smoke test).
- Maintain ≥ 75 % accuracy when adding 20 % joint occlusion or 5 % 2D outliers.
- Scale to `M = 3..5` people with ≥ 80 % frame-wise association accuracy on
  synthetic data before moving to real Shelf/Campus sequences.

## GPU / CPU classification

**CPU-only.** No training, no GPU. The script runs in milliseconds to seconds
per frame on the WSL host.

## Command and result

Run the existing smoke test:

```bash
python experiments/associate_multi_person_synthetic.py --smoke
```

Output:

```
Running CPU smoke test...
Generating 2 clips x 5 frames x 4 views

=== Synthetic Two-Person Association Results ===
Sequences: 2, Frames/seq: 5, Views: 4
Person separation: 1.5 m, Noise std: 1.0 px
Overall frame-wise association accuracy: 100.00%
Mean per-frame association time: 150.55 ms

Per-sequence metrics:
  seq 00: frame_acc=100.00% (raw=80.00%, swap=20.00%)
  seq 01: frame_acc=100.00% (raw=60.00%, swap=40.00%)
```

A slightly larger CPU-only run confirms the same performance:

```bash
python experiments/associate_multi_person_synthetic.py --n_sequences 10 --n_frames 10 --n_views 4 --noise_std 1.0
```

Output:

```
Generating 10 clips x 10 frames x 4 views

=== Synthetic Two-Person Association Results ===
Sequences: 10, Frames/seq: 10, Views: 4
Person separation: 1.5 m, Noise std: 1.0 px
Overall frame-wise association accuracy: 100.00%
Mean per-frame association time: 177.85 ms
```

## Notes

- No existing training runners were modified, so the running cross-view PP
  curriculum on the RTX 4090 is unaffected.
- The current brute-force search is exponential in the number of views and only
  works for `M=2`; the next code change must replace it with a polynomial-cost
  Hungarian/greedy baseline before real-world data is attempted.
