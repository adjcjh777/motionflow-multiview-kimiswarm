# v49: Scalable Architecture for A800

**Status:** Proposal / ready for design review  
**Labels:** `experiment`, `P1-next`, `infra`  
**Tracking issue:** #167 (proposed)  
**Depends on:** #160 (v46-SVG), #162 (v47-temporal), #164 (v48-domain)  

---

## 1. Problem statement

The v46–v48 stack layers three new heads on top of the v25/v45 geometry-fusion backbone:

- **v46** adds a per-view reliability MLP and view-dropout augmentation.  
- **v47** adds a 2-layer temporal transformer over `(time, joint)` tokens.  
- **v48** adds a domain-conditional FiLM/GRL adapter and dynamic domain-weighted loss.

Full A800 runs already target `d=128`, `batch_size=16`, `clip_len=13` (`launch_v33_a800_queue.py`).
As these heads are stacked, activation memory and per-step runtime grow multiplicatively:

- The v47 temporal attention scales as `O((T·J)^2)` in memory.
- The v48 domain adapter doubles the feature footprint for domain-conditional offsets.
- The v25 geometry-fusion attention block already holds large `(B, T, V, J, C)` activations.

Without explicit scaling interventions, the v48 full stack will either OOM on A800 or require
aggressive batch-size reductions that slow convergence.  Worse, the A800 queue currently runs
one GPU per experiment; we lack knobs to trade memory for speed or to fit larger runs on the
same hardware.

**v49 therefore introduces optional memory/compute scaling mechanisms for the A800-bound
variant of the v46–v48 stack.**  It is not a new model component; it is a *scaling layer* that
makes the existing stack trainable at full capacity on A800.

---

## 2. Proposed approach and fit with v46–v48

Keep every v46–v48 module unchanged in terms of semantics; add optional memory and speed
optimisations that can be enabled only for A800 full runs.

```text
multi-view video
    |
    ▼
v25/v45 geometry fusion backbone
    |
    ├── v46 Sparse-View Generalization (reliability MLP)
    │
    ├── v47 Temporal Aggregation (2-layer transformer over T·J tokens)
    │       └─ optional gradient checkpointing / local-window attention
    │
    ├── v48 Domain Adapter (FiLM/GRL, domain-conditional offsets)
    │       └─ optional activation checkpointing / fused domain embedding
    │
    └── v49 Scalable A800 layer (memory/speed knobs, adaptive compute)
            ├─ gradient checkpointing on heavy blocks
            ├─ memory-efficient / flash attention backend
            ├─ torch.compile on static sub-graphs
            └─ adaptive depth gated by v37/v46 reliability
```

The guiding principles are:

1. **Default-off.** Every new knob is a flag; existing v46/v47/v48 runs are bit-exact.
2. **No new parameters.** v49 does not add learnable weights; it only changes how existing
   weights are executed.
3. **A800-only by default.** The smoke config tests correctness on RTX 4090; the full config
   targets A800.
4. **Self-evolution aware.** Use the v37 self-critique / v46 reliability score to decide where
   full v47/v48 compute is needed, skipping expensive paths on high-confidence frames.

---

## 3. Concrete code-level changes

### 3.1 New files

| File | Purpose |
|------|---------|
| `motionflow_mv/training/scalable_utils_v49.py` | Helpers for gradient-checkpoint wrappers, attention backend dispatch, and `torch.compile` guards. |
| `configs/benchmark_v49_scalable_a800_smoke.yaml` | Local smoke config (small batch, short clip). |
| `configs/benchmark_v49_scalable_a800_full.yaml` | A800 full config (`d=128`, `batch_size=16`, `clip_len=13`). |
| `scripts/run_v49_scalable_a800_smoke_local_4090.sh` | Local 4090 smoke script. |

### 3.2 Modified files

| File | Change |
|------|--------|
| `motionflow_mv/fusion/omniview_fusion_v5.py` | Add v49 flags; wrap v25 attention, v47 transformer, and v48 adapter with checkpointing/compile helpers when flags are on. |
| `motionflow_mv/fusion/temporal_aggregation_v47.py` | Accept optional `attention_backend` and `checkpoint_every_n_layers`; implement local-window attention more efficiently. |
| `motionflow_mv/fusion/domain_adapter_v48.py` | Accept optional activation-checkpointing flag; fuse domain embedding with FiLM offsets in one kernel when possible. |
| `motionflow_mv/training/trainer_v2.py` | Respect `amp_enabled` with v49 compile; log peak memory per step when `--profile_memory` is set. |
| `experiments/train_omniview_fusion_v5_webbridge_multi.py` | Expose v49 CLI flags; profile peak memory and throughput. |
| `scripts/launch_v33_a800_queue.py` | Add a v48+scalable-A800 queue entry. |

### 3.3 New training / model flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `use_v49_scalable_a800` | bool | `False` | Master switch. |
| `v49_use_gradient_checkpointing` | bool | `True` | Checkpoint activations inside v25/v47/v48 heavy blocks. |
| `v49_checkpoint_every_n_layers` | int | `1` | Checkpoint every N transformer/attention layers (`0` = disable). |
| `v49_attention_backend` | str | `"default"` | `"default"`, `"flash"`, or `"memory_efficient"`. Falls back to `"default"` if backend unavailable. |
| `v49_use_torch_compile` | bool | `False` | `torch.compile` the static parts of the model (eval mode only until stable). |
| `v49_compile_mode` | str | `"reduce-overhead"` | `torch.compile` mode (`reduce-overhead`, `max-autotune`, `default`). |
| `v49_use_adaptive_compute` | bool | `False` | Skip v47/v48 heads on high-confidence frames (gated by v46 reliability). |
| `v49_adaptive_compute_threshold` | float | `0.85` | Reliability score above which the expensive heads are skipped. |
| `v49_memory_efficient_v47_window` | int | `None` | If set, force v47 to use a fixed local window (trades accuracy for memory). |
| `v49_profile_memory` | bool | `False` | Log `torch.cuda.max_memory_allocated()` per train/val step. |

### 3.4 Minimal interface sketch

```python
# motionflow_mv/training/scalable_utils_v49.py

import torch
from torch.utils.checkpoint import checkpoint
from typing import Callable, Optional


def checkpoint_layer(layer: torch.nn.Module, *args, **kwargs):
    """Forward `layer` with gradient checkpointing if enabled globally."""
    if getattr(torch, "_v49_checkpoint_enabled", False):
        return checkpoint(layer, *args, use_reentrant=False, **kwargs)
    return layer(*args, **kwargs)


def set_attention_backend(name: str):
    """Return a backend-compatible attention callable or None if unsupported."""
    if name == "flash":
        try:
            from torch.nn.functional import scaled_dot_product_attention as sdpa
            # Flash attention is selected automatically by PyTorch when inputs are in fp16/bf16.
            return sdpa
        except Exception:
            return None
    if name == "memory_efficient":
        try:
            from torch.backends.cuda import enable_math_sdp, enable_mem_efficient_sdp
            enable_mem_efficient_sdp(True)
            from torch.nn.functional import scaled_dot_product_attention as sdpa
            return sdpa
        except Exception:
            return None
    return None
```

The v47 temporal transformer would wrap each encoder layer with:

```python
for i, layer in enumerate(self.layers):
    if i % v49_checkpoint_every_n_layers == 0:
        x = checkpoint_layer(layer, x)
    else:
        x = layer(x)
```

---

## 4. Risks / failure modes

| Risk | Mitigation |
|------|------------|
| Gradient checkpointing slows the forward pass by 20–30 %. | Only enable on A800 full runs; keep 4090 smoke without it by default. |
| `torch.compile` fails with dynamic view masks or variable `T`. | Compile only the static sub-graphs; guard with `torch._dynamo.reset()` on first failure. |
| Flash attention is not installed / dtype mismatch. | Auto-fallback to `"default"`; do not fail the run. |
| Adaptive compute skips too many frames, hurting sparse-view accuracy. | Gate only the v47/v48 heads, never v25/v46; keep a small residual always computed. |
| v49 flags interact badly with AMP. | Test smoke with `--amp` before any A800 run. |
| Memory profiling adds overhead. | Make `--v49_profile_memory` optional and default-off. |

---

## 5. Success metrics and recommended experiments

### 5.1 Success criteria

1. With `use_v49_scalable_a800=true`, the v48 full stack (`d=128`, `batch=16`, `clip_len=13`)
   fits in a single A800 GPU (≤ 80 GiB) without OOM.
2. Peak training memory is reduced by ≥ 20 % compared to the same run with v49 disabled.
3. Throughput (samples/sec) is within 15 % of the non-scaled baseline.
4. `val_MPJPE@full` is within 0.5 mm of the non-scaled v48 baseline.
5. No NaN/OOM across the first epoch on A800.

### 5.2 Smoke experiment (RTX 4090)

| Field | Value |
|-------|-------|
| Config | `configs/benchmark_v49_scalable_a800_smoke.yaml` |
| Hardware | Local RTX 4090 |
| Stack | v25 + v45 + v46 + v47 + v48 |
| Knobs | `v49_use_gradient_checkpointing=true`, `v49_attention_backend=memory_efficient`, `v49_profile_memory=true` |
| Goal | `val_MPJPE` finite, no NaN/OOM, memory log produced, `< 80 mm` |
| Duration | ~1–2 hours |

### 5.3 Full experiment (A800-D)

| Field | Value |
|-------|-------|
| Config | `configs/benchmark_v49_scalable_a800_full.yaml` |
| Hardware | A800-D single GPU |
| Stack | v25 + v45 + v46 + v47 + v48 |
| Knobs | `v49_use_gradient_checkpointing=true`, `v49_attention_backend=flash`, `v49_use_adaptive_compute=true` |
| Goal | Fit `d=128`, `batch=16`, `clip_len=13`; complete ≥ 1 epoch; `val_MPJPE@full` ≤ 20 mm |
| Baseline | Same v48 config without v49 flags; expect OOM or much larger memory footprint. |

### 5.4 Launch via A800 queue

Add to `scripts/launch_v33_a800_queue.py`:

```python
(
    "v48_domain_generalization_scalable_a800",
    "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
    "--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
    "--use_v45_adaptive_geometry_fusion --v45_adaptive_weight_type per_view_joint --v45_adaptive_weight_hidden 32 --v45_adaptive_weight_n_layers 1 "
    "--use_v46_sparse_view_generalization --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 --v46_svg_hidden 64 --v46_svg_use_curriculum "
    "--use_v47_temporal_aggregation --v47_temporal_d_model 64 --v47_temporal_n_heads 4 --v47_temporal_num_layers 2 --v47_temporal_loss_weight 0.01 --v47_use_view_count_conditioning "
    "--use_v48_domain_generalization --v48_dg_hidden 64 --v48_dg_grl_lambda 0.1 --v48_dg_use_domain_film --v48_dg_use_ddwl --v48_dg_ddwl_temperature 2.0 --v48_dg_ddwl_warmup_epochs 1 "
    "--use_v49_scalable_a800 --v49_use_gradient_checkpointing --v49_checkpoint_every_n_layers 1 --v49_attention_backend flash --v49_use_adaptive_compute --v49_profile_memory "
    "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 200 --epochs 5 --early_stopping_patience 2 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
    "omniview_fusion_v48_domain_generalization_scalable_a800",
),
```

---

## 6. Self-evolution feedback loop

v49 reuses the v37 self-critique / v46 reliability signal to drive **adaptive compute depth**:

1. After the v46 reliability head, compute a frame-level confidence score
   `c_t = mean_v,j(r_v,j,t)` where `r` is the v46 reliability weight.
2. If `c_t > v49_adaptive_compute_threshold`, skip the v47 temporal transformer and the v48
   domain adapter for that frame; output the v46 triangulated pose directly.
3. Low-confidence frames (few views, occlusions, noisy detections) still run the full v47/v48
   refinement, so robustness is preserved where it matters.
4. During training, the skipped-path frames receive gradients only through v25/v46, while the
   full path frames back-propagate through v47/v48.  This is no different from ordinary masking.

This closes a *compute-aware* self-evolution loop: the model learns which views/frames are
reliable (v37/v39) and spends its A800 budget where it is most needed, rather than applying the
full v47/v48 stack uniformly.

---

## 7. Paper story fit

v49 supports the paper claim: *Our multi-view pose pipeline is not only accurate and robust,
it is also trainable at full capacity on modern high-memory GPUs.*  By enabling the v46–v48
stack to run at `d=128` and batch size 16 on A800, v49 makes the full paper pipeline
reproducible at scale and prevents memory from becoming the bottleneck that forces
simplifying assumptions.

---

## 8. Next steps

1. Wait for v47-temporal (#162) and v48-domain (#164) smoke results.
2. Implement `motionflow_mv/training/scalable_utils_v49.py` and the v49 flags in
   `OmniMultiViewFusionV5`.
3. Run the local 4090 smoke and verify no regression versus v48 without v49 flags.
4. Update `scripts/launch_v33_a800_queue.py` with the v48 scalable-A800 entry.
5. Start the A800 full run when a GPU frees; monitor memory logs and throughput.
6. Document the final memory/throughput numbers in `AGENTS.md`.
