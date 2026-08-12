# Multi-person multi-view association

**Investigation complete.** I found a clear single-person assumption across the codebase and an previous `docs/swarm_iter4/multi_person_association.md` that already scoped the problem, but no implementation has been started. Because this subagent is read-only, I could not create the file; the report content is below and should be saved at the requested path.

**Intended path:** `docs/swarm_iter7/multi-person_multi-view_association.md`

---

```markdown
# Multi-Person Multi-View Association

## 1. Current state

The entire MotionFlow-MultiView pipeline is currently **single-person**:

- `motionflow_mv/data/shelf_loader.py` has a `person_id` argument and extracts one person from `annotation_3d.json` (`load_3d_annotations`, line 47; `build_shelf_dataset`, line 92).
- `motionflow_mv/data/webbridge_loader.py` exposes the same `person_id` flag for Shelf/Campus conversion (lines 203, 223).
- `motionflow_mv/ir/human_motion_ir.py` defines `person_id` in `HumanMotionIR` (line 21) and `motionflow_mv/ir/multiview_adapter.py` propagates it, but there is no logic that ever produces more than one IR per scene.
- `experiments/convert_panoptic_v1.py` and `experiments/convert_3dpw_multiview.py` both take a single `person_idx`/`person_id` and extract one actor.
- `experiments/generate_synthetic_multiview_dataset.py` generates one SMPL body per sequence.
- `motionflow_mv/fusion/ray_attention_model.py` `RayAttentionFusionModel` consumes `(B, V, J, 3)` and outputs one skeleton `(B, J, 3)`.
- `motionflow_mv/pipeline.py` `MultiViewPipeline.fuse_frame` triangulates one skeleton from `(V, J, 2)` detections.

Prior work already identified the gap: `docs/swarm_iter4/multi_person_association.md` recommended a multi-person loader, a triangulation-consistency association baseline, and a synthetic multi-person benchmark. None of these have been implemented yet.

## 2. Gap / opportunity

To move from a single-person triangulation paper to a practical multi-view pose system, the project needs:

1. **A multi-person loader** that returns detection sets `(T, M, V, J, 3)` for `M` people.
2. **A cross-view association module** that groups per-view detections into per-person tracks before fusion.
3. **A synthetic multi-person benchmark** with ground-truth identities so association accuracy can be measured independently of pose accuracy.
4. **Downstream evaluation metrics** (MOTA/IDF1-style association accuracy plus per-person MPJPE).

This is high-leverage for the paper because it turns the current geometric-fusion story into a complete multi-view system story without requiring new single-person model training.

## 3. Concrete next step

Add `experiments/associate_multi_person_synthetic.py` that:

1. Extends `generate_synthetic_multiview_dataset.py` to produce `M=2..5` SMPL bodies per sequence, each with a unique trajectory, and projects all bodies into the same `V` calibrated views.
2. Treats the projected 2D keypoints as independent per-view "detections" and adds realistic detection noise / false positives.
3. Implements a baseline association:
   - For each pair of views, compute epipolar distance between all detection pairs.
   - Build candidate `V`-tuples; triangulate each candidate with the existing `triangulate_dlt` and score by mean reprojection error.
   - Use Hungarian assignment (via `scipy.optimize.linear_sum_assignment`) to select the best `M` consistent tuples, constrained to one detection per person per view.
4. Runs the existing `RayAttentionFusionModule` on each associated person.
5. Reports:
   - **Association accuracy**: fraction of correctly linked detections across views vs. GT identity.
   - **3D MPJPE** per person compared to the single-person synthetic baseline.
   - **Runtime** of the association step.

The script should be self-contained and not modify existing core modules; it reuses `motionflow_mv/fusion/triangulation.py` and `motionflow_mv/fusion/ray_attention_module.py`.

## 4. Expected success metric

- Association accuracy ≥ 90 % on clean synthetic 2-person scenes.
- Per-person MPJPE within 10 % of the single-person `RayAttentionFusionModel` baseline when association is correct.
- Association remains ≥ 75 % with 20 % joint occlusion or 5 % 2D outliers.

## 5. Risks / blockers

- **Combinatorial cost**: naive tuple enumeration is exponential in `M` and `V`; the first version must prune with epipolar distance thresholds and confidence filtering.
- **No real multi-person labels**: the first validation must be synthetic; real Shelf/Campus multi-person annotations can be used only after the loader is extended.
- **A800-D / Docker read-only**: any large-scale training should run locally on the RTX 4090 or WSL; do not write to A800-D.
- **WebBridge data**: do not commit raw dataset downloads; keep outputs in `outputs/` or `data/` only if small and listed in `.gitignore`.
```

**Summary for the parent agent:** The repo remains single-person throughout. The most actionable, non-disruptive experiment is a self-contained `experiments/associate_multi_person_synthetic.py` that builds a multi-person synthetic scene, runs a geometric-association + Hungarian baseline, and feeds the resulting per-person detections into the existing `RayAttentionFusionModule`. This directly supports the paper's practical-systems claim without touching the current best model.