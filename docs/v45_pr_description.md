# v45 Planning Docs PR Description

## Title
Add v45 planning documents and decision criteria

## Summary
This PR introduces the planning documentation for the **v45** architecture iteration of MotionFlow-MultiView. v45 follows the v44 decision process and codifies the next round of experiments based on the outcomes of the A800 runs queued in issue #154.

Specifically, it adds:

- `docs/v45_plan.md`: high-level plan for the v45 architecture, including the four candidate branches inherited from v44 (v25-based simplification, v42/v43 complex-stack regularization, adaptive per-node residual, and capacity/data scaling) and the conditions under which each is selected.
- `docs/v45_ablation_checklist.md`: concrete ablations and engineering follow-ups for whichever branch is chosen (SWA/EMA, stochastic depth, variable-view training, outlier augmentation, learned depth triangulation, etc.).
- `docs/results_snapshot_2026_08_09.md` is referenced but left unchanged; v45 decisions should be validated against the A800 results recorded there.

The v45 plan is intentionally **conditional**: no GPU runs are launched by this PR. It documents the decision tree and next actions so the team can act as soon as the pending A800 experiments finish.

## Checklist

- [ ] Read and confirm `docs/v44_decision_plan.md` and `docs/v43_decision_criteria.md`.
- [ ] Verify the A800 runs in issue #154 have produced epoch-1 validation MPJPE values.
- [ ] Choose the v45 branch according to the decision rules in `docs/v45_plan.md`.
- [ ] Update `docs/v45_ablation_checklist.md` with the actual branch selected and remove irrelevant ablations.
- [ ] Run a smoke test on the local RTX 4090 before any full A800 run.
- [ ] Add or update the A800 queue entry for the chosen v45 variant.
- [ ] Update `AGENTS.md` / status tables if the v45 branch changes the active GPU queue.
- [ ] Link this PR to issue #154 and any newly created v45 tracking issues.
