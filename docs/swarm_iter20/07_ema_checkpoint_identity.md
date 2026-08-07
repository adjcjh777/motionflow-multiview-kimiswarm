# Swarm Iteration 20 — EMA Checkpoint Identity

## Finding

`TrainerV2.evaluate()` temporarily applies EMA parameters whenever EMA and
`ema_eval` are enabled. Best-checkpoint selection therefore used EMA validation
loss. The old `save_checkpoint()` ran after those parameters had been restored
and wrote raw parameters to `model`; every Omni evaluator, test-set inference
script, and warm-start loader then read only that raw `model` entry.

The selected validation metric and the externally evaluated checkpoint thus
referred to different parameter identities. The best epoch was also selected
by validation `loss`, while the three Omni training scripts printed its MPJPE
as `Best val MPJPE`; that value was not necessarily the minimum-MPJPE epoch.

## Minimal contract

- `model` always means raw training parameters and remains aligned with the
  optimizer for resume.
- `eval_model` is the complete state used for checkpoint-time evaluation: EMA
  parameters plus the model buffers when EMA evaluation is enabled, otherwise
  the raw state.
- `eval_weights` records `ema` or `raw`.
- best checkpoints record `monitor=loss`, `monitor_mode=min`, and the selected
  value. The running best loss is included in the training state.
- the epoch scheduler is stepped before a best checkpoint is written, so its
  scheduler state matches the next epoch.

One shared `checkpoint_eval_state_dict()` helper now supplies evaluation and
warm-start weights. Omni v2/v3 clean evaluation, variable-view evaluation,
camera-perturbation evaluation, v2 test-set inference, and the three Omni
warm-start paths use it.

## Existing checkpoint compatibility

Old composite TrainerV2 checkpoints already contain both raw `model` and
`ema.shadow`. For the known Omni paths, the helper copies the full raw state and
overlays the EMA parameter entries, exactly reproducing the parameter identity
used by `evaluate()` while retaining model buffers. Plain state dicts remain
plain state dicts. Resume continues to load raw `model` and the saved optimizer,
scheduler, AMP, and EMA state.

No EMA bias correction is added. The shadow starts from the initial model
parameters rather than zero, so dividing it by `1 - decay**step` would
incorrectly amplify early weights.

## Historical evidence boundary

Historical TrainerV2 validation logs with EMA enabled describe EMA parameters;
historical external Omni evaluations used raw parameters. Both are observations,
but their difference cannot be attributed to EMA without re-evaluating both
identities under the same data and metric protocol. Runs with EMA disabled are
not affected by this mismatch.

In particular, the documented H36M `20.91 mm` trainer value and the external
`15.03 mm` clean validation value were produced through different weight-loading
paths and must not be presented as measurements of one identical checkpoint
state. The old composite file can be re-evaluated with its reconstructed EMA
state if the artifact remains available.

For resume compatibility, loading an older checkpoint without EMA initializes
the new shadow from the loaded raw parameters, and an absent `best_metric` is
recovered from validation-loss history. A run with best saving disabled does
not advance the saved-best threshold.

## Validation boundary

This iteration adds one focused CPU checkpoint-identity test and performs
Python syntax/static checks. The local checkout has no Torch environment, so
the Torch test is added but not claimed as locally executed. No GPU experiment
was run.
