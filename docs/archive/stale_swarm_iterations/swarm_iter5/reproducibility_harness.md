# Reproducibility harness (swarm iter 5)

## Task
Add config-driven training (YAML), deterministic seeding, and optional W&B logging for
`RayAttentionFusionModelV3`.

## Deliverables

* `experiments/train_ray_attention_reproducible.py`
  * Loads all hyperparameters from a YAML config file passed via `--config`.
  * Sets `random`, `numpy`, and `torch` seeds deterministically; disables
    `cudnn.benchmark` and enables `cudnn.deterministic`.
  * Optionally logs to Weights & Biases when `wandb.enabled: true`.
  * Reuses the existing `CameraDataset`, `collate_fn`, and `augment_batch` from
    `experiments/train_ray_attention_v3_h36m.py` to stay compatible with the
    H36M NPZ layout.

* `configs/train_ray_attention_reproducible.yaml`
  * Example config with dataset path, training hyperparameters, augmentation,
    W&B switches, and output paths.

* `docs/swarm_iter5/reproducibility_harness.md` (this file)

## Design decisions

* Keep the trainer minimal and local to one new file; no changes to the model,
  data loaders, or other training scripts.
* PyYAML is declared via an explicit import guard so users get a clear error if
  it is missing.
* W&B is fully optional. When enabled, the script verifies `wandb` is installed
  before calling `wandb.init`.
* Config is flat and human-readable; defaults are embedded so the YAML can be
  short.

## Verification

A smoke test with `epochs=2` on the local WSL 4090 confirms the script parses
config, seeds RNGs, and runs the training/validation loop. No full 50-epoch run
was launched per the task constraint.

## Next steps

* Add a similar reproducibility wrapper for `train_ray_attention_real.py` if
  Shelf/Campus training becomes the bottleneck.
* Consider logging LR schedule and gradient norms once an optimizer scheduler is
  added.
