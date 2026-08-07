# Swarm Iteration 20 — Camera-Centric Scale Geometry

## Deterministic defect

The camera-centric variant describes `scale_v` as a per-view ray-depth scale,
but the integrated model previously computed `mean(scale_v) * X_world` after
triangulation. That operation:

- scales about the arbitrary world origin rather than each camera center;
- discards which view predicted which scale;
- is not translation equivariant: after a world-origin shift `q`, its output
  changes by `mean(scale) * q` instead of `q`.

The scale is not returned, logged, or directly supervised by the available
trainer. This iter15 variant is commented out of the registry, has no dedicated
train/eval script, and has no repository checkpoint or reported result.

## Minimal geometric contract

For triangulated joint `X_j`, corrected camera center `C_v`, observed unit ray
`d_vj`, view scale `s_v`, and triangulation weight `w_vj`, the repaired
post-DLT residual is

```
lambda_vj = dot(d_vj, X_j - C_v)
alpha_vj = w_vj / sum_v(w_vj)
X'_j = X_j + sum_v alpha_vj (s_v - 1) lambda_vj d_vj
```

This applies only the depth component along each observed ray, preserves the
per-view scale assignment, and exactly reduces to `X_j` when all scales equal
one. Conditional on fixed scales and weights, the operator is equivariant to a
change of world origin.

## CPU gate

`experiments/prototypes/swarm_iter20/camera_centric_scale_geometry_probe.py`
compares the historical and repaired operators. The focused Torch test freezes
three contracts: identity reduction, translation equivariance, and different
outputs when same-mean per-view scales are swapped between distinct cameras.

For a world shift `(10,-4,2)`, the historical operator adds an error
`(-0.5,0.2,-0.1)` while the repaired operator's error is numerical zero. Two
same-mean scale assignments swapped between the cameras produce outputs
`1.169995` units apart instead of collapsing to the same global scale.

This helper is a minimal post-DLT ray-depth residual fusion. With noisy rays,
the scaled per-view depths need not intersect at one common 3-D point. The full
network is also not structurally translation equivariant because other feature
and residual paths consume absolute camera/world values. Therefore this repair
does not establish full-model physical consistency or an accuracy improvement.

Any first camera-centric checkpoint should be trained from the anchor after
this change. No GPU experiment was run and no MPJPE claim is made.
