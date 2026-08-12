# Paper Outline, Figure Ideas, and Timeline toward Submission

## 1. Survey of the topic

The MotionFlow multi-view extension is now anchored by a single, well-performing technical contribution: **ray-aware attention fusion with differentiable weighted DLT triangulation** (`motionflow_mv/fusion/ray_attention_model.py`). The model consumes per-view 2D keypoints + confidences plus calibrated cameras, embeds camera centers and ray directions, predicts per-view weights via self-attention, and triangulates with a differentiable weighted DLT layer. This design directly addresses the instability of earlier `attention_v2` attempts that regressed 3D coordinates from flattened projection matrices.

Validation status:

- Synthetic 4-view data with 0.8 px noise: `ray_attention` MPJPE ≈ 0.0036 m, robust to  10% occlusion and 2% outliers.
- GVHMR multi-view projection demo: `ray_attention` 0.0021 m vs. legacy `attention` 3.68 m.
- Real-data loader (`motionflow_mv/data/shelf_loader.py`) and trainer (`experiments/train_ray_attention_real.py`) are in place for Shelf/Campus, with a DLT baseline for comparison.
- Geometric plugins (`dlt`, `temporal_refiner`, `robust_triangulation`) still provide strong baselines (≈10 px reprojection on Shelf, ≈1.5 px on Campus cross-dataset), and `ray_attention` must beat or match them on real data to justify the learned approach.

For ICRA/CVPR 2027 the paper needs a clear narrative, strong figures, and a deadline-driven execution plan. The core story is: **geometry-aware attention is a better inductive bias than direct 3D regression; weight prediction + DLT triangulation gives calibrated, interpretable, and robust multi-view fusion**.

## 2. Concrete actionable recommendations

### 2.1 Adopt a four-section paper structure

1. **Introduction & motivation**: single-view 3D human pose (GVHMR/ScoreHMR) is noisy; calibrated multi-view fusion is under-explored in the MotionFlow setting; existing attention-based fusion ignores camera geometry.
2. **Method**: ray-aware attention fusion. Define rays, embedding, attention, weighting, and differentiable weighted DLT. Contrast with direct 3D regression and with vanilla confidence-weighted DLT.
3. **Experiments**: synthetic ablations, Shelf/Campus real-data benchmarks, GVHMR multi-view projection, cross-dataset generalization.
4. **Discussion & future work**: scale-invariance, temporal consistency, extension to SMPL.

### 2.2 Design five canonical figures

1. **Architecture diagram** (`fig_arch.pdf`): inputs → ray computation → attention block → per-view weights → weighted DLT → 3D joints. Annotate tensor shapes.
2. **Ray embedding visualization** (`fig_rays.pdf`): show camera centers and rays for a sample pose, and how occluded/outlier views get down-weighted.
3. **Results bar chart** (`fig_results.pdf`): MPJPE on synthetic clean/occluded/outlier, Shelf, Campus; include DLT and vanilla-attention baselines.
4. **Ablation figure** (`fig_ablation.pdf`): flattened P vs. ray embedding; direct 3D regression vs. weighted DLT; effect of input scale and camera normalization.
5. **Timeline Gantt** (`fig_timeline.pdf`): from present to CVPR/ICRA deadline, highlighting data acquisition, real-data training, figure polish, and internal review.

### 2.3 Lock the submission venue and deadline

Target **CVPR 2027** (deadline typically mid-November 2026) as the primary venue, with **ICRA 2027** (deadline typically September 2026) as the fall-back/earlier option. Work backward:

- Internal paper draft freeze: 4 weeks before deadline.
- Real-data experiments finalized: 8 weeks before deadline.
- All figures and tables locked: 2 weeks before deadline.
- One-round internal review: 1 week before deadline.

### 2.4 Define the minimum publishable unit

The paper must demonstrate at least one real-data result where `ray_attention` outperforms the DLT baseline. If real Shelf/Campus GT is insufficient or noisy, pivot the contribution to: (a) the synthetic-to-real transfer analysis, and (b) the GVHMR multi-view projection demo, both of which already show large gaps over the legacy `attention` plugin.

### 2.5 Prepare the supplementary material early

Start a `docs/supplementary/` folder now with: per-joint MPJPE tables, camera calibration details, hyperparameters, and video frames of failure cases. Reviewers will ask for these; generating them after the deadline is expensive.

## 3. Potential risks

- **Negative real-data delta**: `ray_attention` may not beat DLT on Shelf/Campus if the dataset is too clean or too small. Mitigation: frame the work as a robustness/contribution method, not a raw-accuracy one, and report controlled synthetic ablations.
- **Data access bottleneck**: raw Shelf/Campus/H36M may not be locally available. Mitigation: use the synthetic generator and the GVHMR projection demo as primary experiments; treat real-data training as a stretch goal.
- **Camera convention drift**: mm vs. m, `t` sign, and K normalization still vary across loaders. Mitigation: enforce the plugin contract (`requires_calibration`, `input_scale`, `output_scale`) in every loader and document it in the paper.
- **Timeline slip**: figure polishing and supplementary materials always take longer than expected. Mitigation: set figure deadlines 2 weeks before submission and use the four-week draft freeze.

## 4. Fit into the paper plan

This report defines the packaging layer of the ICRA/CVPR 2027 submission. It converts the engineering progress in `motionflow_mv/fusion/ray_attention_model.py` and `experiments/train_ray_attention_real.py` into a publishable narrative. The ray-aware attention model is the central technical contribution; the paper outline, figure set, and timeline ensure that contribution is presented clearly and evaluated against the right baselines on the right datasets. The next step is to populate the five figures above with real numbers and to run `train_ray_attention_real.py` on Shelf/Campus to either confirm or adjust the contribution claim before the draft freeze.
