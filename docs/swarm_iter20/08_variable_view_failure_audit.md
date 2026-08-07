# Swarm Iteration 20 — Variable-View Failure Audit

## Observed failure

The H36M dense+graph table records `14.99 mm` with four views, `1619.90 mm`
with three, and `1990.56 mm` with two. This is a real output of the historical
pipeline, but it does not isolate a geometric loss of observability or a single
model component.

## Protocol defects

1. The integrated v2/v3 evaluators built each clip as `(T,V,J,3)` and passed it
   directly to the wrapper. The model interpreted that four-dimensional tensor
   as `(B,V,J,3)`, inserted a temporal dimension of one, and therefore treated
   frames as independent batch elements. The corrected path inserts the batch
   dimension explicitly and removes it from the prediction afterward. The
   dedicated variable-view evaluator already used the correct five-dimensional
   path, but its checkpoint inference omitted `n_joint_layers` and reduced any
   graph stack to a Boolean. It now counts both module stacks from state-dict
   keys so a dense checkpoint is not silently evaluated by a no-dense builder.
2. Training `view_dropout_rate` only sets confidence to zero. The true pixel
   coordinates remain available to ray embedding, view/graph/ST attention, and
   the residual path. It is therefore confidence dropout with observation
   leakage, not absent-view training.
3. The inference wrapper zeros `(x,y,confidence)`. This creates a different,
   unseen token: a phantom `(0,0)` ray plus a real camera embedding and fixed
   view positional embedding. No attention key mask, graph edge mask, or masked
   view pooling removes it.
4. Omni v2/v3 multiply learned weights by confidence and visibility, then clamp
   the result to at least `1e-4`. An inactive view is consequently reintroduced
   into DLT and Gauss-Newton with a small positive weight. This violates exact
   exclusion, but it is not large enough by itself to explain the two-metre
   cliff in a well-conditioned synthetic rig.

## CPU geometry diagnostic

`experiments/prototypes/swarm_iter20/variable_view_dlt_mask_probe.py` uses a
four-camera rig, 1000 random 3-D points, every two/three-view subset, and 1 px
Gaussian image noise. It compares exact subset DLT with padded zero observations,
the triangulator's `1e-6` epsilon, and the caller's `1e-4` weight floor.

| Active views | Exact subset mean | Padded + weight floor mean | Increment |
|---:|---:|---:|---:|
| 2 | 6.0590 mm | 6.0635 mm | +0.0045 mm |
| 3 | 4.3986 mm | 4.3994 mm | +0.0009 mm |

The exact zero-padded solve matches the active-subset solve to floating-point
precision. This diagnostic does not reproduce the H36M rig, learned weights,
or network residuals, so it cannot prove the checkpoint's root cause. It does
show that ordinary two-view geometry and the current minimum DLT weight are not
a sufficient explanation for an error near 2000 mm.

## Current mechanism decision

The leading code-supported mechanism is the train/eval masking mismatch
combined with unmasked attention, graph propagation, and view pooling. The DLT
weight floor is a secondary contract bug. The historical curve is classified
as `PROTOCOL_CONFOUNDED`, not as evidence that two or three calibrated cameras
are intrinsically inadequate and not as a clean graph-vs-no-graph result.

The integrated evaluator shape bug is fixed now. Changing the trained model's
attention topology or claiming a recovered MPJPE requires a same-checkpoint
stage-ablation rerun, which is outside this CPU-only iteration. No GPU run was
started.

The caller floor and DLT epsilon must eventually be changed together so zero
weights remain zero in both DLT and Gauss-Newton. That global numerical change
is deferred here: this checkout cannot run the existing Torch CPU/CUDA
rank-deficiency and backward tests, and changing only one of the two sites
would preserve the same broken contract.
