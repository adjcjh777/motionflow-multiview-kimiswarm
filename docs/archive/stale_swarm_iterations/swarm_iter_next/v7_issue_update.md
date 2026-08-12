# v7 Issue/PR Update — 2026-08-07

> This is the content intended for GitHub issue #85 / PR #86.  
> Blocker: no `GITHUB_TOKEN` or authenticated `gh` is available in this session, so the issue/PR has not been posted automatically.

## 1. v6 Status Update (for issue #84)

| Run | GPU | Status | Notes |
|-----|-----|--------|-------|
| v6_mpi_isab | 5 | **restarted** | NaN not observed; running 30 epochs, 10k samples/epoch |
| v6_mpi_perceiver | 6 | **restarted** | Running |
| v6_h36m_isab | 4 | **restarted with fix** | Crashed with NaN in `AttentionEntropyLoss`; fixed by zeroing ST transformer outputs for masked views |

- Root cause: in variable-view mode, masked query tokens in the ST transformer received softmax over fully-blocked positions, producing NaN that leaked to triangulation weights.
- Fix: in `motionflow_mv/fusion/omniview_fusion_v5.py`, after the ST transformer we now reshape the features, zero out masked views with `torch.where`, and then continue.

## 2. v7 Progress (for issue #85)

- **20-agent swarm synthesis**: committed to `docs/swarm_iter_next/v7_20agent_synthesis.md`.
- **Full-precision DLT triangulation**: implemented in `motionflow_mv/fusion/triangulation.py` and wired into `OmniMultiViewFusionV5`. Smoke test passed (96.41 mm on 1-epoch mixed loader, d=32).
- **Mixed-loader smoke**: passed with the `dataset_id` fix.
- **v7 launcher**: `scripts/tmux_v7_mixed_precision.sh` ready to run mixed H36M+MPI with full-precision DLT.
- **Monitor**: `scripts/a800_session_monitor.sh` updated to track v6 sessions and v4_adaptive; v7 is commented out until a GPU is free.

## 3. GPU Status on A800-D

```
0-3: occupied by VLLM (read-only for our work)
4:   v6_h36m_isab (restarted)
5:   v6_mpi_isab (restarted)
6:   v6_mpi_perceiver (restarted)
7:   v4_varview_adaptive_mpi (user run, kept alive)
```

v7 mixed-dataset training cannot start until GPU 4/5/6 finish v6 or GPU 7 frees up after v4_adaptive completes.

## 4. Next Actions

1. Wait for v6 first-epoch results (estimate ~2 h per epoch).
2. If v6 H36M MPI clean MPJPE is stable, start `v7_mixed_precision` on the first free GPU.
3. Post actual results to #85 and open a PR for v7 once a full v7 run converges.
