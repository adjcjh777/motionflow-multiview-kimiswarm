# Dependency Note: Temporal Failure Analysis

For `experiments/analyze_failures_temporal_mpiinf3dhp.py`, the following packages were added to the `mf` conda environment on the local RTX 4090:

* `matplotlib` 3.10.9
* `seaborn` 0.13.2 (pulled in via pandas)

Install command used:

```bash
conda run -n mf pip install matplotlib seaborn
```

No global installs were made; everything is isolated to the `mf` environment.
