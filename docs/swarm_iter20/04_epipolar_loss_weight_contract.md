# Swarm Iteration 20 — Epipolar Loss-Weight Contract

**Date:** 2026-08-07
**Scope:** static code-path audit and fix; no training and no GPU.

## Finding

The Bayesian Tri trainer exposed `--epipolar_loss_weight`, but the three model
constructors did not receive that value. Their forward methods used the model
default `0.05`, then the trainer multiplied the returned loss by the CLI value
again.

The old effective coefficient was therefore

`lambda_effective = 0.05 * lambda_cli`

not `lambda_cli` and not generally `lambda_cli^2`. For example, a script that
specified `0.02` trained with an effective coefficient of `0.001`.

## Fix

For `bayesian_tri`, `bayesian_tri_v2`, and `bayesian_tri_v2_visibility`:

1. Pass `args.epipolar_loss_weight` into the model constructor.
2. Add the returned `epi_loss` directly in the trainer because model forward
   already applies that coefficient once.

The new contract is

`lambda_effective = lambda_cli`.

Consequently, re-running an old launch script unchanged does not reproduce its
old objective: CLI `0.02` changes from effective `0.001` to `0.02`, and CLI
`0.05` changes from `0.0025` to `0.05` (both 20-fold increases).

## Evidence boundary

Historical checkpoints and metrics remain observations from their old recipes.
Any discussion of epipolar-loss strength must use the old effective coefficient,
not the command-line value printed in the launch script. This fix is not a new
training result and does not rehabilitate the previously incorrect epipolar
source/destination geometry.

The four Bayesian trainer variants now also return raw triangulated 3D before
their final epipolar/combined auxiliary scalar when `--reproj_raw_weight` is
enabled. The trainer selects that explicit tuple position instead of treating
the final scalar as a 3D pose.
