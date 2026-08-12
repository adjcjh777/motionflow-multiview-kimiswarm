# Runtime &amp; efficiency benchmark on RTX 4090

Done. I examined the existing benchmark scripts, the current best temporal-residual model, and the paper draft, then wrote the report.

**Report:** `docs/swarm_iter7/runtime_efficiency_benchmark_rtx4090.md`

**Summary of findings**
- The repo already has `experiments/benchmark_residual_temporal.py` and an saved result at `outputs/benchmark_residual_temporal_4090.md`, reporting 78.3 ms/clip at B=1 and up to 195 clips/s at B=16 for the 243 k-parameter `RayAttentionFusionModelTemporalResidual`.
- The paper draft (`docs/paper_draft_icra_cvpr_2027.md:106–114`) already includes this table, so the current state is documented.
- No optimization work has been done yet: no `torch.compile`, no per-component `torch.profiler` breakdown, no real-data timing on the final checkpoint, and no memory-footprint measurement.
- The proposed next step is to add a new profiling script that loads the final checkpoint, benchmarks real WebBridge clips (with synthetic fallback), records per-op latency, tries `torch.compile`, and reports GPU memory — without modifying core fusion code.