# MotionFlow-MultiView Ablation Study Design (ICRA / CVPR 2027)

**Scope:** full ablation matrix for `OmniMultiViewFusionV5` (and its v10--v16
variants) on MPI-INF-3DHP, Human3.6M, and WebBridge.  
**Last updated:** 2026-08-07.  
**Target branch:** `swarm/ablation_design`.  
**Tracking issue:** #76.

---

## 1. Goals and Scientific Questions

The ablation study must answer six questions for the ICRA/CVPR 2027 paper:

1. **Which architectural components of v5 contribute to clean accuracy?**
2. **Which components are responsible for robustness under occlusion, calibration noise, and variable views?**
3. **What is the marginal value of the v10--v16 incremental improvements?**
4. **How sensitive is the model to capacity (d, layers, residual hidden size)?**
5. **Does multi-dataset training generalize across datasets?**
6. **At what view count does the system become usable for robotics (MPJPE threshold < 20 mm)?**

All ablations share the same training seed, data split, and base hyperparameters
unless explicitly varied.  The anchor is the full v13 + v12 configuration
(temporal + adaptive multiscale + robust DLT + IRLS, currently running as
v10--v16 baselines on A800 GPUs 4--7).

---

## 2. Model Component Ablation Matrix

Run every row with `d=128`, `residual_hidden=256`, `n_st_layers=3`,
`graph_num_layers=1`, `n_joint_layers=1`, `n_heads=4`, 20 epochs, 2000 train
samples, batch size 16, and the WebBridge mixed loader.  Start from the v13
config and toggle a single block per row.

| Row | Short name | Toggled flag vs. v13 | Scientific question |
|-----|------------|----------------------|---------------------|
| R00 | **Raw DLT** | `use_full_precision_dlt=false`, all learned modules disabled, triangulate with uniform confidence | Geometric lower bound |
| R01 | v10 no-outlier | v10 full config, `outlier_view_prob=0.0` | Best v10 without synthetic outliers |
| R02 | v10 aleatoric-outlier | v10 full config, `outlier_view_prob=0.3` | Value of outlier-view augmentation |
| R03 | v11 IRLS | `use_irls_reweight=true` (adds to v10) | Value of Cauchy IRLS reweighting |
| R04 | v12 adaptive multiscale | `use_adaptive_multiscale_fusion=true` (adds to v11) | Value of adaptive scale-selective fusion |
| R05 | v13 temporal | `temporal_loss_weight=0.03`, `clip_len=27` (adds to v12) | Value of temporal consistency losses |
| R06 | v15 kinematic chain | `bone_loss_weight>0`, `joint_limit_weight>0` (adds to v13) | Value of skeleton kinematic auxiliary losses |
| R07 | v16 occlusion/noise | `occlusion_augment_*` enabled (adds to v13) | Value of joint-level occlusion augmentation |
| R08 | Full v13 + camera embedding | `use_camera_view_embedding=true`, `use_set_view_aggregator=true` | Does camera-conditioned set aggregation help full views? |
| R09 | Full v13 + perceiver | `use_perceiver_aggregator=true` instead of ISAB | ISAB vs. Perceiver view aggregation |
| R10 | No principal-point correction | `correct_principal_point=false` | How much accuracy comes from intrinsic self-calibration? |
| R11 | No epipolar bias | `use_epipolar_bias=false` | Value of epipolar-biased ST transformer |
| R12 | No camera conditioning | `use_camera_conditioning=false` | Value of camera extrinsics as features |
| R13 | No graph-joint attention | `graph_num_layers=0` | Value of skeleton-graph joint attention |
| R14 | No context visibility | `use_context_visibility=false` | Value of learned per-view visibility gating |
| R15 | No skeleton residual | `use_skeleton_residual=false` | Value of skeleton-graph residual refiner |
| R16 | No rotation correction | `use_rotation_correction=false` | Value of rotation-correction head |
| R17 | No entropy regularization | `use_entropy_regularization=false` | Value of triangulation-weight entropy loss |
| R18 | No domain embedding | `use_domain_embedding=false` | Value of dataset-specific embedding for mixed H36M/MPI training |
| R19 | No variable-view training | `use_variable_view_training=false` | Does variable-view curriculum hurt clean full-view accuracy? |

**Control rule:** only one column in each row should differ from the row it is
compared against.  When a component is removed, all other flags stay at the v13
value to isolate marginal contributions.

---

## 3. Hyperparameter Ablation Matrix

Fix the full v13 config and vary one hyperparameter per sub-matrix.

### 3.1 Feature / model capacity

| Parameter | Values to test | Default | Notes |
|-----------|----------------|---------|-------|
| `d` | 32, 64, 128, 256 | 128 | Feature dimension of the ST transformer |
| `residual_hidden` | 32, 64, 128, 256, 512 | 256 | Residual MLP hidden size |
| `n_st_layers` | 1, 2, 3, 4 | 3 | Spatiotemporal transformer layers |
| `graph_num_layers` | 0, 1, 2, 3 | 1 | Skeleton-graph joint-attention layers |
| `n_heads` | 2, 4, 8 | 4 | Attention heads (d must be divisible) |

### 3.2 Training / loss

| Parameter | Values to test | Default | Notes |
|-----------|----------------|---------|-------|
| `clip_len` | 9, 15, 21, 27, 37 | 27 | Temporal window; only applicable to temporal variants |
| `temporal_loss_weight` | 0.0, 0.01, 0.03, 0.1, 0.3 | 0.03 | Weight of velocity + acceleration consistency |
| `reproj_loss_weight` | 0.0, 0.05, 0.1, 0.2 | 0.1 | 2D reprojection loss weight |
| `aleatoric_reproj_loss_weight` | 0.0, 0.05, 0.1, 0.2 | 0.1 | Aleatoric reprojection term |
| `pa_loss_weight` | 0.0, 0.25, 0.5, 1.0 | 0.5 | Procrustes-aligned loss weight |
| `bone_loss_weight` | 0.0, 0.02, 0.05, 0.1 | 0.05 | Bone-length loss (v15) |
| `joint_limit_weight` | 0.0, 0.01, 0.02, 0.05 | 0.02 | Joint-limit loss (v15) |
| `attention_entropy_weight` | 0.0, 0.005, 0.01, 0.02 | 0.01 | Entropy regularization weight |
| `monotonic_loss_weight` | 0.0, 0.05, 0.1, 0.2 | 0.1 | Multi-view ranking loss weight |
| `lr` x schedule | {1e-4, 3e-4, 1e-3} x {cosine, plateau} | 1e-3 cosine | Learning rate and schedule |

### 3.3 Data augmentation

| Parameter | Values to test | Default | Notes |
|-----------|----------------|---------|-------|
| `outlier_view_prob` | 0.0, 0.1, 0.3, 0.5 | 0.3 | Probability of corrupting a whole view |
| `outlier_view_noise_std` | 0, 5, 10, 15, 25 | 15 | Pixel std of outlier Gaussian noise |
| `occlusion_augment_prob` | 0.0, 0.25, 0.5, 0.75 | 0.5 | v16 joint-level occlusion probability |
| `occlusion_joint_rate` | 0.0, 0.04, 0.08, 0.16 | 0.08 | Per-joint occlusion probability |
| `noise_std` | 0.0, 1.0, 2.0, 5.0 | 1.0 | Base 2D observation noise |
| `view_dropout_rate` | 0.0, 0.1, 0.2, 0.3 | 0.0 | Whole-view dropout at training |
| `variable_view_min_views` | 2, 3, 4 | 2 | Minimum views in variable-view training |
| `variable_view_max_views` | 4, 8, 14 | 14 | Maximum views in variable-view training |

---

## 4. Robustness Ablation Matrix

Run the full v13 checkpoint under the following perturbation protocols.
Record MPJPE/PA-MPJPE at each level.  These can be run on the validation split
without retraining.

| Dimension | Levels | Metric / protocol |
|-----------|--------|-------------------|
| **Variable views** | k = 2, 3, 4, 6, 8, 10, 14 | `eval_variable_views_batched.py`; best-k and random-k subsets |
| **Rotation noise** | σ = 0.0°, 0.3°, 0.5°, 1.0°, 2.0° | Perturb R before triangulation; report clean vs. perturbed |
| **Translation noise** | σ = 0, 0.01 m, 0.05 m, 0.1 m | Perturb t |
| **Focal-length noise** | 0%, 1%, 3%, 5%, 10% | Perturb K[0,0] and K[1,1] |
| **Principal-point noise** | 0, 5, 10, 20, 50 px | Perturb K[0,2] and K[1,2] |
| **2D Gaussian noise** | σ = 0, 1, 2, 5, 10 px | Add to input keypoints |
| **Sparse outliers** | 0%, 5%, 10%, 20% of views | Large 2-D offset / noise injection |
| **Joint occlusion** | 0%, 20%, 50% of joints | v16-style per-joint masking |
| **Cross-dataset** | MPI → H36M, MPI → AIST, H36M → MPI | Train on one dataset, evaluate on another |

For each condition report:

- `MPJPE_clean` / `PA_MPJPE_clean`
- `MPJPE_perturbed` / `PA_MPJPE_perturbed`
- Relative degradation Δ%
- Failure rate (samples with MPJPE > 100 mm)

---

## 5. Data and Domain Ablations

| Row | Setup | Goal |
|-----|-------|------|
| D01 | MPI-INF-3DHP only | Baseline trained only on MPI |
| D02 | H36M only | Baseline trained only on H36M (4-view rig) |
| D03 | H36M + MPI mixed | Current v10--v16 default |
| D04 | D03 + AIST++ | Add motion-rich dance data |
| D05 | D03 with `use_domain_embedding=true` | Explicit domain token vs. implicit mixing |
| D06 | 25% of MPI data | Data-efficiency baseline |
| D07 | 50% of MPI data | Data-efficiency baseline |
| D08 | 100% of MPI data | Full data baseline |

Metrics: MPJPE/PA-MPJPE on each domain's validation set, plus per-domain
PCK@50/100/150 and AUC.

---

## 6. Metrics and Reporting

Every ablation must report the same standardized set of metrics so results can
be placed in a single paper table.

### 6.1 Required metrics

| Metric | Source | Why |
|--------|--------|-----|
| MPJPE (mm) | `motionflow_mv/eval/metrics.py` | Primary accuracy |
| PA-MPJPE (mm) | `motionflow_mv/eval/metrics.py` | Alignment-insensitive accuracy |
| PCK@50/100/150 mm | `motionflow_mv/eval/metrics.py` | Standard 3DHP metric |
| AUC | `motionflow_mv/eval/metrics.py` | Standard 3DHP metric |
| Per-joint MPJPE | `motionflow_mv/eval/metrics.py` | Identify limb/trunk trade-offs |
| Mean reprojection error (px) | Forward pass | 2D geometric consistency |
| Per-view weight entropy | Forward pass | Attention diversity |
| Visible-view support count | Forward pass | Robustness under dropout |
| Parameter count | PyTorch | Model complexity |
| Training wall time (h) | Logger | Cost |
| Inference time per clip (ms) | Timer | Real-time feasibility |
| GPU memory peak (GB) | `nvidia-smi` / PyTorch profiler | Capacity constraints |

### 6.2 Master results table

Create a single CSV at `outputs/ablation_study_iter20_master.csv` with columns:

```text
run_id, name, branch, commit, seed, d, residual_hidden, n_st_layers, graph_num_layers,
use_full_precision_dlt, use_robust_dlt_reweight, use_irls_reweight,
use_adaptive_multiscale_fusion, temporal_loss_weight, clip_len,
use_context_visibility, use_skeleton_residual, use_rotation_correction,
use_entropy_regularization, use_camera_view_embedding, use_set_view_aggregator,
use_perceiver_aggregator, use_domain_embedding, use_variable_view_training,
outlier_view_prob, occlusion_augment_prob, mpjpe, pa_mpjpe, pck50, pck100,
pck150, auc, reproj_px, params_M, wall_hours, gpu_mem_gb
```

This CSV feeds `docs/tables/ablation_component_table.md` and the plotting script
`experiments/ablation_csv_plotting.py`.

---

## 7. Training and Evaluation Protocol

### 7.1 Common training command

```bash
# Example: R02 v10 aleatoric outlier on GPU 0 (already running as v10_aleatoric_outlier on GPU 4)
tmux new-session -d -s abl_r02 \
  "source .venv/bin/activate && \
   CUDA_VISIBLE_DEVICES=0 \
   python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
     --use_mixed_loader \
     --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
     --use_full_precision_dlt --use_robust_dlt_reweight --use_domain_embedding \
     --d 128 --residual_hidden 256 --n_st_layers 3 \
     --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
     --epochs 20 --batch_size 16 --train_samples 2000 --val_stride 10 \
     --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 \
     --max_grad_norm 1.0 --ema_decay 0.999 \
     --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
     --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
     --use_entropy_regularization true --attention_entropy_weight 0.01 \
     --use_camera_view_embedding --use_set_view_aggregator \
     --use_variable_view_training \
     --variable_view_min_views 2 --variable_view_max_views 14 \
     --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 \
     --variable_view_permute \
     --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
     --reproj_loss_weight 0.1 --reproj_warmup_epochs 3 \
     --aleatoric_reproj_loss_weight 0.1 \
     --outlier_view_prob 0.3 --outlier_view_max_views 1 \
     --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
     --output outputs/ablation_r02_v10_aleatoric.pth \
     > outputs/ablation_r02_v10_aleatoric.log 2>&1"
```

### 7.2 Smoke-first rule

Before launching any full ablation, run the same command with `--smoke --epochs 1`
and a tiny batch on CPU or a single GPU for 10 minutes to confirm:

- No import or shape errors.
- Forward/backward pass completes.
- Checkpoint saving works.
- Log parsing regexes match.

### 7.3 Evaluation command

```bash
python experiments/eval_omniview_fusion_v5_mpiinf3dhp.py \
    --checkpoint outputs/ablation_r02_v10_aleatoric.pth \
    --dataset mpiinf3dhp_val \
    --output_json outputs/ablation_r02_v10_aleatoric_eval.json \
    --run_robustness_matrix
```

For H36M and cross-dataset transfer, use the corresponding eval scripts in
`experiments/`.

---

## 8. Resource Plan

Assuming each full run takes ~24 h on one A800 GPU and there are 8 free GPUs
(GPUs 0--3 plus 6 when available; GPUs 4--7 currently occupied), the priority
queue is:

| Phase | Runs | GPU-days | Goal |
|-------|------|----------|------|
| P1: Component ablations R00--R07, R10--R14 | 12 | ~12 | Core component contributions |
| P2: v13 + view-embedding variants R08--R09 | 2 | ~2 | Variable-view aggregation |
| P3: Hyperparameter sweeps | 8 | ~8 | Capacity / loss sensitivity |
| P4: Robustness matrix | 1 checkpoint x many conditions | 1 | Paper robustness figure |
| P5: Data/domain ablations D01--D08 | 8 | ~8 | Cross-dataset and data efficiency |
| **Total (minimum)** | **~30 runs** | **~30 GPU-days** | Full ablation set |

To reduce cost, run all hyperparameter ablations in P3 at half epochs (10) on a
representative subset (e.g., MPI S2/Seq1 only), then reproduce the top-3
configurations for the full 20 epochs.

---

## 9. Expected Findings (Hypotheses)

| Component | Expected impact | How to verify |
|-----------|---------------|---------------|
| Full-precision DLT + robust reweight | Moderate clean gain, large robustness gain under outliers | Compare R00 vs. R01/R02 |
| IRLS reweighting | Reduces tail errors, small clean gain | R02 vs. R03 |
| Adaptive multiscale fusion | Small clean gain, larger gain for variable views | R03 vs. R04 |
| Temporal consistency | Largest temporal continuity improvement; small clean MPJPE gain | R04 vs. R05 |
| Kinematic-chain aux. losses | Better limb joints, lower PA-MPJPE | R05 vs. R06 |
| Occlusion/noise augmentation | Best robustness under occlusion and 2D noise | R05 vs. R07 |
| Camera view embedding + set aggregator | Improves variable-view generalization, may slightly hurt fixed-view full views | R05 vs. R08 |
| Principal-point correction | Large gain under `pp_px > 10` perturbation | R05 robustness matrix |
| Epipolar bias | Modest gain for multi-view consistency | R05 vs. R12 |

---

## 10. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| 30+ ablations exceed available A800 time | High | Smoke first; prioritize P1; use half-epoch screening for P3 |
| v13 checkpoint not yet available | Medium | Use v12 as anchor; update table when v13/v15/v16 finish |
| Component interactions confound single-factor ablation | Medium | Add leave-one-out and add-one-in ladders around the top model |
| Variable-view training destabilizes small models | Medium | Lower `variable_view_max_views_start` to 4 and ramp gradually |
| Mixed-dataset loader biases toward larger dataset | Low | Balance sampler or report per-domain metrics |
| NaN in full-precision DLT under perturbed cameras | Medium | Re-run with `--use_full_precision_dlt=false` fallback and flag |

---

## 11. Deliverables

1. `docs/ablation_study_design.md` (this file) -- design and protocol.
2. `scripts/run_ablation_matrix_iter20.sh` -- launcher for all ablations.
3. `experiments/ablation_csv_plotting.py` -- plots from the master CSV.
4. `docs/tables/ablation_component_table.md` -- component ablation results.
5. `docs/tables/ablation_hyperparam_table.md` -- hyperparameter sweep results.
6. `docs/tables/ablation_robustness_table.md` -- robustness matrix results.
7. `outputs/ablation_study_iter20_master.csv` -- raw master results.

---

## 12. Summary

The ablation study is organized as a **component ladder** (R00--R19), a
**hyperparameter grid**, a **robustness matrix**, and a **data/domain matrix**.
All experiments run on the same `OmniMultiViewFusionV5` codebase with
per-toggle flags, the same WebBridge mixed loader, and a common metric set.
The minimum plan is ~30 GPU-days on A800; the output is a set of paper-ready
tables that quantify the contribution of every v10--v16 innovation and the
sensitivity of the full v13 model to its design choices.
