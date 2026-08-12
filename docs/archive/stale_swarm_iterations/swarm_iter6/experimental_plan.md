# Swarm Iter 6: Publishable Benchmark Experimental Plan

**Date:** 2026-08-04  
**Goal:** Define the exact experiments, datasets, baselines, and metrics needed to move the current best residual refinement model from a promising local result (~13.84 mm on MPI-INF-3DHP S2 Seq1) to a publishable ICRA/CVPR 2027 benchmark.

## Current State

- **Best model:** `RayAttentionFusionModelTemporalResidual` in `motionflow_mv/fusion/ray_attention_temporal_residual_model.py`
- **Best checkpoint:** `outputs/ray_attention_temporal_residual_v2.pth` (~13.84 mm MPJPE on MPI-INF-3DHP val S2 Seq1)
- **Training script:** `experiments/train_ray_attention_temporal_residual_mpiinf3dhp.py`
- **Key finding from iter5:** A lightweight residual MLP head on top of the temporal ray-attention DLT triangulator yielded a large relative improvement (25.2 mm → 13.84 mm) on the MPI-INF-3DHP cross-subject split.

## Data Inventory (verified)

| Dataset | Files | Shape `points_2d` | Views | Joints | Notes |
|---------|-------|-------------------|-------|--------|-------|
| MPI-INF-3DHP | `data/webbridge/mpi_inf_3dhp/s_*_seq_*_v14_multiview_m.npz` | `(T, 14, 28, 2)` | 14 | 28 | Main benchmark; subjects 1/2 train, subject 2/3 val/test |
| H36M | `data/h36m_hf/s_01_acts_*_multiview_m.npz` | `(T, 4, 17, 2)` | 4 | 17 | Train on S01, cross-subject val on S05/S09/S11 |
| Shelf | `data/shelf_campus/Shelf_Seq1/pseudogt_m.npz` | `(3200, 5, 17, 2)` | 5 | 17 | Cross-scene zero-shot evaluation |
| Campus | `data/shelf_campus/Campus_Seq1/pseudogt_m.npz` | `(1423, 3, 17, 2)` | 3 | 17 | Cross-scene zero-shot evaluation |

**Cross-dataset challenge:** Joints per skeleton (17 vs 28) and camera counts (3–14) differ, so a single shared-head model cannot be naively trained on all datasets. Options: (a) train/evaluate separate models per dataset; (b) add dataset-specific output heads. This plan starts with (a), which is sufficient for a strong benchmark table.

## Deliverables Added in This Round

| File | Purpose |
|------|---------|
| `experiments/train_ray_attention_temporal_residual_v3_mpiinf3dhp.py` | Extended trainer with aux losses, schedulers, weight decay, grad clipping, JSON logging |
| `experiments/eval_ray_attention_temporal_residual_v1.py` | Full eval script for the residual model (MPJPE, PA-MPJPE, PCK, AUC) |
| `experiments/run_publishable_benchmark_plan.py` | Orchestrator that prints and can run the entire plan in smoke/fast mode |
| `docs/swarm_iter6/experimental_plan.md` | This report |

## Proposed Experiments (Ranked by Expected Impact)

### Priority 1 — Scale the Residual Model on MPI-INF-3DHP

| ID | Experiment | Datasets | Compute | Metric Goal |
|----|-----------|----------|---------|-------------|
| `residual_mpi_d128_h256_clip13` | d=128, residual_hidden=256, clip_len=13 | MPI S1→S2 Seq1 | ~30 min smoke / ~8 h full | Beat 13.84 mm MPJPE |
| `residual_mpi_d128_h256_clip27` | Longer temporal context (clip_len=27) | MPI S1→S2 Seq1 | high | Improve PCK/AUC and smoothness |

- **Why:** The current best uses d=64, residual_hidden=128. Doubling capacity is the cheapest way to push accuracy.
- **Baseline:** Compare against the non-residual temporal model and the current `ray_attention_temporal_residual_v2.pth`.
- **Metrics:** MPJPE, PA-MPJPE, PCK@50/100/150 mm, AUC, per-joint breakdown.

### Priority 2 — Human3.6M Cross-Subject Benchmark

| ID | Experiment | Datasets | Compute | Metric Goal |
|----|-----------|----------|---------|-------------|
| `residual_h36m_s01_train_s05_val` | Train on S01 all actions, val on S05 | H36M S01→S05 | medium | Establish H36M number for paper table |

- **Why:** H36M is the de-facto 3D pose benchmark. A strong number here is essential for publication.
- **Note:** H36M uses 17 joints and 4 views. The model infers `j` and `n_views` automatically, so the same script works.

### Priority 3 — Auxiliary Losses and Robustness

| ID | Experiment | Datasets | Compute | Metric Goal |
|----|-----------|----------|---------|-------------|
| `residual_mpi_aux_bone_velocity` | Add bone-length + velocity losses | MPI S1→S2 | medium | Improve temporal consistency, modest MPJPE gain |

- **Why:** MPJPE alone does not reward biomechanically plausible poses. Bone-length regularisation and temporal smoothness improve visual quality and PCK tails.
- **Implementation:** `--aux_weight 0.001 --velocity_weight 0.001` in the v3 trainer.

### Priority 4 — Cross-Scene Zero-Shot on Shelf / Campus

| ID | Experiment | Datasets | Compute | Metric Goal |
|----|-----------|----------|---------|-------------|
| `residual_mpi_zs_shelf` | Zero-shot eval of MPI-trained model on Shelf | Shelf | low | Measure generalisation |
| `residual_mpi_zs_campus` | Zero-shot eval of MPI-trained model on Campus | Campus | low | Measure generalisation |

- **Why:** Demonstrates the model is not over-fitted to MPI camera layouts.

### Priority 5 — Baseline Comparison

| ID | Experiment | Datasets | Compute | Metric Goal |
|----|-----------|----------|---------|-------------|
| `small_residual_baseline_mpi` | Small residual model (d=64, h=128) on MPI | MPI S1→S2 | low | Show residual head gain over tiny baseline |

- **Why:** A baseline table must include at least plain DLT and the small residual model. The plain-DLT-only baseline can be added by evaluating the non-residual `RayAttentionFusionModelTemporal`.

## Metrics to Report

| Metric | Unit | Source |
|--------|------|--------|
| MPJPE | mm | `motionflow_mv.eval.metrics.mpjpe_batch` |
| PA-MPJPE | mm | `motionflow_mv.eval.metrics.pa_mpjpe` |
| PCK@50/100/150 mm | 0–1 | `motionflow_mv.eval.metrics.pck_batch` |
| PCK AUC (0–150 mm) | 0–1 | `motionflow_mv.eval.metrics.pck_auc` |
| Per-joint errors | mm | `per_joint_mpjpe`, `pa_mpjpe_per_joint` |
| Inference time / model params | ms / count | `experiments/benchmark_inference_v3.py` (adapt for residual model) |

## Exact Smoke-Test Commands (<=10 epochs, local RTX 4090)

```bash
# Print the full plan
conda run -n mf python experiments/run_publishable_benchmark_plan.py --smoke

# Run the smoke subset (1 epoch, 1000 clips, batch_size=2)
conda run -n mf python experiments/run_publishable_benchmark_plan.py --run --smoke
```

The orchestrator runs only one heavy job at a time, avoiding GPU contention.

## Smoke-Test Results (this round)

The new v3 trainer and evaluator were validated on a 200-frame MPI smoke subset and a synthetic 4-view/17-joint stub.

| Test | Script | Result |
|------|--------|--------|
| Synthetic stub train | `train_ray_attention_temporal_residual_v3_mpiinf3dhp.py` | Completed 1 epoch; val MPJPE 101.07 mm (random data) |
| Synthetic stub eval | `eval_ray_attention_temporal_residual_v1.py` | MPJPE 161.13 mm, PA-MPJPE 154.72 mm, PCK@150mm 0.4645 |
| MPI smoke (200 fr) train | `train_ray_attention_temporal_residual_v3_mpiinf3dhp.py` | Completed 1 epoch; val MPJPE 14.62 mm |
| MPI smoke (200 fr) eval | `eval_ray_attention_temporal_residual_v1.py` | MPJPE 23.54 mm, PA-MPJPE 21.37 mm, PCK@150mm 1.0000 |
| Orchestrator smoke run | `run_publishable_benchmark_plan.py` | 1 epoch d=128/h=256 clip13: val MPJPE 9.60 mm; eval MPJPE 15.04 mm, PA-MPJPE 14.24 mm |

The smoke numbers are not benchmarks — they simply confirm that the new scripts load real MPI-INF-3DHP data, run an epoch, save a checkpoint, and evaluate it end-to-end. The full benchmark runs remain to be executed.

## Full-Run Commands (to be launched after smoke validation)

```bash
# Priority 1: scaled residual model on MPI
conda run -n mf python experiments/train_ray_attention_temporal_residual_v3_mpiinf3dhp.py \
    --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
            data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 128 --residual_hidden 256 --epochs 30 --batch_size 2 \
    --train_samples 8000 --scheduler cosine \
    --log outputs/swarm_iter6_residual_mpi_d128_h256_clip13_log.json

# Evaluation
conda run -n mf python experiments/eval_ray_attention_temporal_residual_v1.py \
    --checkpoint outputs/swarm_iter6_residual_mpi_d128_h256_clip13.pth \
    --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --clip_len 13 --d 128 --residual_hidden 256 \
    --out outputs/swarm_iter6_residual_mpi_d128_h256_clip13_eval.json
```

## Compute Budget (local RTX 4090 24 GB)

- **Smoke tests:** ≤30 min total, 1 epoch each, batch_size=2.
- **Priority 1 full run:** ~6–8 h (30 epochs, MPI).
- **Priority 2 full run:** ~4–6 h (H36M S01).
- **Priority 3–5:** Low/medium; mostly evaluations or short fine-tunes.

**Safety rule:** never run two GPU training jobs concurrently; evaluations can run sequentially after training.

## Blockers and Risks

1. **Memory with d=128 / clip_len=27:** The temporal transformer memory scales as O(T·V·J · d). clip_len=27 on MPI (14 views, 28 joints) may require batch_size=1 on the RTX 4090.
2. **Cross-dataset joint mismatch:** 17-joint vs 28-joint skeletons prevent a single shared-head model. Future work can add dataset-specific joint regression heads.
3. **H36M eval files:** `s_05_acts_02_multiview.npz`, `s_09_acts_02_multiview.npz`, `s_11_acts_02_multiview.npz` are present but not the `_m` (meter-normalised) versions; the script loads `camera_K/R/t` from these files, which should still work.
4. **NumPy BLAS instability:** The env has a broken NumPy BLAS backend; evaluation uses PyTorch/NumPy carefully, but large SVD/Procrustes calls in PA-MPJPE could still be fragile.

## Next Steps (post-swarm)

1. Run the full Priority 1 experiment and record MPJPE/PCK/AUC.
2. If Priority 1 beats 13.84 mm, run Priority 2 (H36M) and Priority 4 (Shelf/Campus) evaluations.
3. Integrate the non-residual `RayAttentionFusionModelTemporal` as a proper DLT-only baseline in the orchestrator.
4. Add a model-size vs accuracy table for the paper.
5. Run inference benchmarking on the final checkpoint to report latency/throughput.
