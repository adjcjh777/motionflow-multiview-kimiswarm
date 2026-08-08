## Description

Briefly describe the purpose of this pull request and the experiment/model change it introduces.

- **Related experiment issue:** <!-- Link to the corresponding experiment issue, e.g. Closes #123 -->
- **Type of change:** <!-- Bug fix, new feature, ablation, config change, etc. -->
- **Affected components:** <!-- e.g. model, loss, data loader, evaluation -->

## Changes Made

- List the key changes in this PR.
- Include any new config files or command-line options.
- Note any dependencies that were added or updated.

## Validation

- [ ] Training/test command used:
- [ ] Results summary (metrics, runtime, GPU):
- [ ] Comparison against baseline:
- [ ] No regression in existing tests / benchmarks
- [ ] GitHub smoke tests pass locally (`bash scripts/run_github_smoke_tests.sh`)

## Experiment Context

If this PR supports an experiment, include the relevant context here:

- **Experiment ID:**
- **Branch/commit:**
- **A800 job status (if applicable):**
- **Key findings:**

## v25 Geometry Fusion Round (if applicable)

If this PR touches `MultiViewGeometryFusionV25` or its integration, also confirm:

- [ ] `pytest tests/test_multiview_geometry_fusion_v25.py -q` passes
- [ ] `v25_use_geometry_bundle_adjustment` starts as identity / no-op and is bounded
- [ ] `v25_geom_loss_weight` is documented and defaults to `0.1`
- [ ] No v21-style camera regression observed in smoke run
- [ ] Related v25 issue is linked: <!-- e.g. Closes #<v25-issue> -->

## Checklist

- [ ] Code follows the project style
- [ ] Config files are committed and documented
- [ ] Tests pass locally (`pytest` or relevant test script)
- [ ] Experiment issue is referenced and updated
- [ ] CHANGELOG / docs updated if needed
