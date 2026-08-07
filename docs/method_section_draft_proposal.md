# Method Section Draft — Proposal & Next Steps

**Target venues:** ICRA / CVPR 2027  
**Repo:** `D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`  
**Status:** v2 method draft exists; v3 code is implemented and training; this document proposes concrete, implementable next steps to bring the method section in line with the actual codebase and ongoing experiments.

## 1. Current State

- **v2 method draft** is in `docs/paper_method_section_draft.md`. It covers: notation, intrinsic self-correction (`PrincipalPointCorrection`), visibility gating (`VisibilityGateHead`), graph-joint attention (`GraphJointAttentionV2`), Bayesian precision-weighted triangulation, adaptive Gauss-Newton refinement, residual refinement, training losses, and evaluation protocols.
- **v3 architecture** is implemented in `motionflow_mv/fusion/omniview_fusion_v3.py` and documented in `docs/omniview_fusion_v3_design.md`. It adds hierarchical multi-scale fusion, camera conditioning, and epipolar-biased cross-view attention to the v2 pipeline.
- **Training is in progress** on 4090 WSL (dense+graph v2) and A800-D (v2 and v3). See `docs/swarm_iter19_status.md`.
- **Tests and ablations** exist for v2 components (`tests/test_cross_view_graph_attention.py`, `experiments/ablate_graph_joint.py`, etc.), but the v3 ablation matrix in `docs/paper_experiments.md` is still a plan.

The method section therefore needs two things: (a) integrate the v3 components into the narrative, and (b) align the paper claims with the code that is actually running. Below are concrete, implementable next steps.

## 2. Proposed Next Steps

### 2.1 Extend `docs/paper_method_section_draft.md` with v3 subsections

The existing draft ends at v2. Add three subsections after "Graph-Joint Attention" and before "Bayesian Precision-Weighted Triangulation":

1. **Hierarchical Multi-Scale Temporal/Cross-View Fusion**
   - Reference: `motionflow_mv/fusion/omniview_fusion_v3.py`, lines 59–186 (`_HierarchicalMultiscaleFusion`).
   - Explain the `(B, T, V, J, d)` input, the per-scale temporal/joint pooling, the cross-view transformer at coarser resolution, and the residual add.
   - Use the scale schedule `s ∈ {1, 2, 4}` from the code.

2. **Camera Conditioning**
   - Reference: `motionflow_mv/fusion/omniview_fusion_v3.py`, lines 189–249 (`_CameraConditioning`).
   - Document the concatenation of flattened `K, R, t` into a 21-D vector, the MLP encoder, and the broadcast-add across joints.

3. **Epipolar-Biased Cross-View Attention**
   - Reference: `motionflow_mv/fusion/epipolar_transformer_bias.py` and `omniview_fusion_v3.py`, lines 491–500.
   - Document `compute_per_frame_epipolar_bias`, the temperature-scaled additive bias, and `build_temporal_bias_from_frames` which expands the per-frame `(V, V)` block into the full `T×V` spatio-temporal attention mask.
   - Mention that `EpipolarBiasedTransformerEncoderLayer` is a drop-in replacement for `nn.TransformerEncoderLayer` when `use_epipolar_bias=True`.

### 2.2 Make graph-joint attention view-mask aware

**Problem:** `GraphJointAttentionV2` builds a static edge list in `build_graph_joint_edge_index` (`graph_joint_attention_v2.py:38–105`). When views are dropped at inference, the cross-view edges still connect all `(v1, v2)` pairs, including dropped views. This biases variable-view results.

**Implementation:**
- Add an optional `view_mask: torch.Tensor | None = None` argument to `GraphJointAttentionV2.forward`.
- In `OmniMultiViewFusionV2._apply_graph_joint_attention`, pass a boolean mask derived from `(confidences > 1e-6).any(dim=-1)` or from the input confidence tensor.
- In `GraphJointAttentionLayer.forward`, zero out messages whose source view is masked. Because edge indices are static, the cleanest approach is to pre-compute per-edge source-view masks at construction and multiply the aggregated messages by the broadcast mask inside the layer.

**Files:**
- `motionflow_mv/fusion/graph_joint_attention_v2.py`
- `motionflow_mv/fusion/omniview_fusion_v2.py` (lines 204–226)
- `motionflow_mv/fusion/omniview_fusion_v3.py` (lines 470–471)

### 2.3 Add a configurable dense+graph hybrid branch in v2

**Problem:** `OmniMultiViewFusionV2` currently disables the dense `joint_attn` path from the ancestor. The no-graph ablation is running, but if it under-performs there is no hybrid path that combines dense joint self-attention and graph-joint attention.

**Implementation:**
- The code already reserves an optional dense branch as `self.omni_joint_attn` in `omniview_fusion_v2.py:137–151`.
- Expose a flag (e.g., `n_joint_layers`) through the trainer CLI and constructor, so the ablation matrix can include:
  - `graph_num_layers=1, n_joint_layers=0` (current default)
  - `graph_num_layers=0, n_joint_layers=1` (dense only)
  - `graph_num_layers=1, n_joint_layers=1` (hybrid)
- This lets the method section honestly compare graph-only, dense-only, and hybrid attention, rather than only graph vs. none.

**Files:**
- `motionflow_mv/fusion/omniview_fusion_v2.py`
- `experiments/train_omniview_fusion_v2_mpiinf3dhp.py`

### 2.4 Implement the v3 ablation matrix as a reproducible script suite

**Problem:** `docs/paper_experiments.md` defines an ablation matrix (runs A–H) but it is not yet executed.

**Implementation:**
- Create a small driver script (e.g., `experiments/run_v3_ablation_matrix.py`) that launches the eight runs with the exact flags from the matrix:
  - A: v2, no graph
  - B: v2, graph
  - C: v3 full
  - D: v3, multi-scale disabled
  - E: v3, camera conditioning disabled
  - F: v3, epipolar bias disabled
  - G: v2 on H36M
  - H: v3 on H36M
- The script should only submit/launch jobs (or generate shell commands) and not start training automatically on the user's main session. It must respect the existing 4090/A800 runs.
- Add a companion CSV/JSON tracking format so results can be pasted directly into `docs/results_icra_cvpr_2027.md` and the paper tables.

**Files:**
- New: `experiments/run_v3_ablation_matrix.py`
- Update: `docs/paper_experiments.md`, `docs/results_icra_cvpr_2027.md`

### 2.5 Add per-joint/per-view failure analysis to the method narrative

**Problem:** The current draft explains *what* each block does but not *why* it helps in terms of specific failure modes.

**Implementation:**
- Add a short "Failure-Mode Analysis" subsection (or a figure caption) that reports:
  - Per-joint error reduction when graph-joint attention is enabled vs. disabled (use `experiments/ablate_graph_joint.py` as the measurement script).
  - Per-view error reduction from camera conditioning and epipolar bias on the MPI-INF-3DHP 14-view rig.
  - Visibility accuracy on artificially occluded joints.
- Reference the actual evaluation code: `experiments/eval_omniview_fusion_v2_variable_views.py` and the robustness scripts in `experiments/`.

### 2.6 Add integration tests for v3 and robustness tests

**Problem:** `tests/` covers v2 components, but there is no end-to-end test with all three v3 flags enabled, nor are there tests for variable-view or robustness behavior.

**Implementation:**
- New test: `tests/test_omniview_fusion_v3_full.py` — instantiate `OmniMultiViewFusionV3` with `use_multiscale_fusion=True`, `use_camera_conditioning=True`, `use_epipolar_bias=True`, run forward/backward, and assert output shapes.
- New test: `tests/test_omniview_fusion_v3_ablations.py` — instantiate v3 with all combinations of flags and assert each runs.
- New test: `tests/test_variable_view_graph_edges.py` — feed a 4-view model a 3-view sample (via confidence masking) and assert that graph cross-view edges for the missing view are effectively ignored.

## 3. Prioritization

| Priority | Step | Why first |
|----------|------|-----------|
| 1 | 2.1 Update v3 subsections in draft | Unblocks paper writing immediately; no GPU time |
| 2 | 2.6 Add v3/variable-view tests | Locks in current behavior before more ablations |
| 3 | 2.3 Add dense+graph hybrid branch | Needed to fairly justify graph-joint attention |
| 4 | 2.2 View-mask-aware graph edges | Needed for honest variable-view claims |
| 5 | 2.4 v3 ablation matrix scripts | Needed to populate paper tables with real numbers |

## 4. Top Action Items

1. **Write v3 subsections** into `docs/paper_method_section_draft.md`: hierarchical multi-scale fusion (`_HierarchicalMultiscaleFusion`), camera conditioning (`_CameraConditioning`), and epipolar-biased attention (`EpipolarBiasedTransformerEncoderLayer`).
2. **Add view-mask-aware graph edges** in `motionflow_mv/fusion/graph_joint_attention_v2.py` so variable-view inference does not silently include dropped views.
3. **Expose and test the dense+graph hybrid branch** in `OmniMultiViewFusionV2` via the existing `omni_joint_attn` path and the trainer CLI.
4. **Create `experiments/run_v3_ablation_matrix.py`** to launch the A–H ablation runs reproducibly and track results for the paper tables.
5. **Add v3 integration and variable-view tests** in `tests/` to protect the method section claims before collecting final numbers.
