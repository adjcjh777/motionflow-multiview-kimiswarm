Research complete. I could not persist the report to disk because this codebase-exploration role lacks file-editing tools, but the full report content is below so the parent agent can create `docs/swarm_iter3/graph_neural_networks_skeleton.md` if desired.

---

# Graph Neural Networks for Skeleton-Aware Multi-View Fusion

## 1. Problem statement

Multi-view human pose estimation fuses noisy, possibly occluded 2D detections from several calibrated cameras into a single coherent 3D skeleton. The current `motionflow-multiview` stack already shows that geometric triangulation (DLT) is a surprisingly strong baseline: on Shelf frames 300–600 it reaches ~9.9 px mean reprojection error, while shallow learned attention or residual refiners only match or marginally improve it (`docs/design_v3.md`). The remaining errors are not primarily a failure of view aggregation; they are failures of *anatomical and occlusion reasoning*:

- **Noisy isolated joints** are not corrected by neighboring joints because per-joint attention treats each joint independently.
- **Occluded views** receive soft weights, but the model has no explicit bone-length or limb-symmetry prior to hallucinate plausible 3D locations.
- **Reprojection-only supervision** lets the network find 2D-consistent but anatomically impossible poses because the loss does not encode the skeleton graph.

Graph neural networks (GNNs) address exactly this gap. By representing the human body as a graph—joints as nodes and bones/symmetries as edges—a GNN can regularize fusion with anatomical structure, while cross-view correspondence edges can propagate information between cameras. This is a natural next step for a CVPR/ICRA 2027 contribution: a lightweight, geometry-aware skeleton graph that can be plugged into the existing `FusionModule` interface and trained with 3D-supervised losses.

## 2. Key related work

| Work | Venue | Relevance to our stack |
|------|-------|------------------------|
| Iskakov et al., *Learnable Triangulation of Human Pose* | ICCV 2019 | First principled learning-based triangulation; shows per-joint view weighting helps but needs 3D labels to beat DLT. |
| Wu et al., *Graph-Based 3D Multi-Person Pose Estimation Using Multi-View Images* | ICCV 2021 | Proposes task-specific GNNs for both 3D person localization and pose estimation; directly applicable to our multi-view setting. |
| Qiu et al., *Dynamic Graph Reasoning for Multi-person 3D Pose Estimation* | 2022 | Dynamic graph reasoning over views and persons; inspiration for occlusion-aware edge construction. |
| Yan et al., *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition* | AAAI 2018 | ST-GCN defines the canonical skeleton graph convolution; bone edges + temporal edges can be reused for pose refinement. |
| Liao et al., *Multiple View Geometry Transformers for 3D Human Pose Estimation (MVGFormer)* | CVPR 2024 | Hybrid geometry/appearance transformer; geometry modules are learning-free, appearance modules are learned; supports a similar plug-in design. |
| Chharia et al., *MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation* | CVPR 2025 | Replaces full cross-view attention with state-space scanning; strong generalization to new camera layouts, which is a key risk for GNN-based approaches. |

Three take-aways from the literature:

1. **Skeleton graphs beat unstructured MLPs/attention on 3D human pose.** Encoding adjacency and symmetry yields better occlusion handling and physically plausible outputs.
2. **Cross-view and cross-joint edges are complementary.** View edges fuse multi-view evidence; bone/symmetry edges impose anatomical structure.
3. **3D ground truth is usually required to outperform triangulation.** Reprojection-only training tends to converge to DLT-like solutions, as already observed in this repository.

## 3. Relation to the current codebase

The repository is already structured to host a GNN plugin:

- **`motionflow_mv/fusion/fusion_module.py`** defines the `FusionModule` ABC and `DLTFusion`. A new `GraphSkeletonFusion` can implement the same `fuse(points_2d, confidences, cameras)` contract.
- **`motionflow_mv/fusion/attention_model.py` / `attention_model_v2.py`** lift per-view `(x, y, confidence)` to `d`-dimensional features and aggregate across views. A GNN plugin can reuse the same `lift` and camera-embedding code, then add joint-joint message passing.
- **`motionflow_mv/ir/human_motion_ir.py`** stores per-view 2D observations, confidence, and camera parameters. It is the right place to also record skeleton-graph uncertainty (e.g., per-edge confidence, per-joint standard deviation).
- **`motionflow_mv/ir/multiview_adapter.py`** already fuses per-view IRs and writes `fused_joints_3d` back into the IR. The adapter can remain unchanged; only the `FusionModule` backend needs to be swapped.
- **`motionflow_mv/data/synthetic_3d_dataset.py`** generates smooth skeleton trajectories. It can be extended to expose the skeleton graph adjacency matrix and bone-length labels for supervised training.

Current weak points a GNN would fix:

- `AttentionFusionModel` has no bone or symmetry edges.
- `ResidualRefinerModel` and `TemporalRefinerModel` refine a DLT skeleton but do not propagate corrections along the skeleton graph.
- The loss functions in `experiments/train_*.py` use only reprojection or DLT pseudo-targets; no bone-length, symmetry, or 3D-supervised losses exist yet.

## 4. Concrete recommendations

### 4.1 Implement a `GraphSkeletonFusion` plugin

Create `motionflow_mv/fusion/graph_skeleton_fusion.py` plus a matching `GraphSkeletonFusionModule` registered in the `FUSION_REGISTRY`. Proposed graph:

- **Nodes:** one node per joint `j` in the selected skeleton (COCO-17 or SMPL-23).
- **Intra-view edges:**
  - Skeleton bones (parent–child).
  - Symmetric limb pairs (left–right arm/leg).
  - Self-loops.
- **Cross-view edges:** connect the same joint node across all views so multi-view evidence is fused before or jointly with skeleton propagation.

A minimal architecture:

```python
class GraphSkeletonFusion(nn.Module):
    def __init__(self, j=17, d=64, n_views=5, n_layers=3):
        self.lift = nn.Linear(3, d)
        self.cam_embed = nn.Linear(12, d)
        # GNN over (view, joint) nodes, with two edge types:
        #   - same-joint across views
        #   - skeleton bone / symmetry edges within a view
        self.gnn = HeterogeneousGraphConv(
            in_channels=d, hidden_channels=d, num_layers=n_layers
        )
        self.head = nn.Linear(d, 3)
```

Use a small GNN first (3 layers, `d=64`). PyTorch Geometric or a hand-written `MessagePassing` layer are both acceptable; the project already depends on PyTorch, so adding `torch-geometric` is low friction.

### 4.2 Loss function

Move beyond reprojection-only supervision:

- **3D position loss:** `L_pos = ||Y_hat - Y_gt||_2` (requires 3D GT).
- **Reprojection loss:** `L_reproj = sum_j w_j ||P_j Y_hat - x_j||_2` to stay 2D-consistent.
- **Bone-length loss:** `L_bone = sum_{(p,c)} | ||Y_p - Y_c||_2 - b_pc |` where `b_pc` is a dataset or subject-specific bone length.
- **Temporal smoothness:** `L_temp = ||Y_t - Y_{t-1}||_2` or a physics-informed loss.
- **Symmetry loss:** encourage left/right limb lengths to be equal.

This mirrors the finding in `docs/swarm_iter2/synthesis_phase1.md` that 3D-supervised losses are the next lever for beating DLT.

### 4.3 Datasets and training

| Dataset | Role | Notes |
|---------|------|-------|
| **Human3.6M** | Primary 3D-supervised training | Large, accurate 3D labels; requires registration. |
| **CMU Panoptic** | Cross-dataset validation | Multi-person, many views; research-only license. |
| **Shelf / Campus** | Fast reprojection sanity checks | Already loaded by `VoxelPoseShelfLoader`; too small for full training. |
| **AMASS** | Synthetic motion prior | Render SMPL sequences through virtual multi-view rigs to pre-train bone-length priors. |

Recommended schedule:

1. Pre-train the GNN on AMASS-rendered synthetic multi-view data with bone-length and 3D losses.
2. Fine-tune on Human3.6M with 3D labels.
3. Validate zero-shot on CMU Panoptic and Shelf/Campus using reprojection and 3D metrics.

### 4.4 Evaluation

Add to `experiments/eval_all_plugins_shelf.py`:

- Per-joint and per-limb PCK/MPJPE.
- Bone-length error.
- Cross-dataset generalization to unseen camera layouts.

## 5. Open questions / risks

- **3D ground-truth access.** Without real 3D labels, the GNN is unlikely to beat DLT on reprojection alone, consistent with findings in `docs/design_v3.md` and `docs/swarm_iter2/synthesis_phase1.md`. Human3.6M or Panoptic must be obtained first.
- **Camera-layout generalization.** GNNs that hard-code a fixed number of views or a fixed skeleton may fail on new rigs. MV-SSM-style state-space scanning or learned camera embeddings should be compared.
- **Multi-person scenes.** Current `multiview_adapter.py` assumes single-person per-view IRs. A skeleton GNN for multi-person needs person-association edges, which is an open problem.
- **Computational cost.** Graph convolutions over `(V × J)` nodes can be heavier than per-joint attention. Start with sparse adjacency and lightweight GCN/GAT before exploring heterogeneous graph transformers.
- **Venue fit.** For CVPR/ICRA 2027, the novelty must be more than “add a GNN.” A strong paper would combine the skeleton graph with explicit SMPL shape/pose parameters, uncertainty-aware message passing, or a calibration-free variant.

## 6. Bottom line

Graph neural networks are the natural next fusion primitive for `motionflow-multiview` because they explicitly model the skeleton structure that DLT and per-joint attention ignore. The codebase already has the plugin interface, IR, and calibration code needed to integrate a GNN. The critical path is not the GNN architecture itself, but **obtaining 3D-supervised training data** (Human3.6M/AMASS) and designing losses that combine 3D position, reprojection, bone-length, and temporal smoothness. Once those are in place, a lightweight skeleton GNN can replace or augment the current `attention`/`residual_refiner` plugins and push the project toward CVPR/ICRA-level numbers.