# v31 Hierarchical Deeper Spatial Ablation

**Task identifier:** `design_v31_hierarchical_deeper_spatial`  
**Depends on:** v30 (`docs/proposals/v30_design.md`, `motionflow_mv/fusion/hierarchical_multiview_v30.py`)  
**Status:** Design / candidate direction  

---

## 1. Problem

v29 proved that a hierarchical multi-scale view encoder can learn, but v29a overfits after the first epoch on small data. v30 hardened that encoder with dataset-aware part groups, gated cross-scale residuals, stochastic depth, and dropout. Those changes slow overfitting, yet the spatial hierarchy remains shallow: the part-scale block is only a single transformer layer (`v30_n_part_layers=1`).

The part scale is the most information-dense level of the hierarchy. It must aggregate joints into anatomical parts, reason across views, and then re-distribute part-level signals back to every joint. A single layer may under-fit the part-level relationships, leaving accuracy on the table. At the same time, any capacity increase must be accompanied by stronger regularization, because v29/v30 already overfit quickly on small smoke subsets. The broken TTE module must stay off, and the v29 physical loss needs a warmup to avoid destabilising early training.

This ablation therefore asks: *can we make the v30 spatial hierarchy meaningfully deeper while keeping it stable?*

## 2. Proposed method

Keep the v30 encoder architecture, but deepen the **part-scale** branch and add regularisation:

| Hyper-parameter | v30 smoke default | v31 deeper spatial |
|-----------------|-------------------|--------------------|
| `v30_n_part_layers` | 1 | **3** |
| `v30_stochastic_depth_prob` | 0.1 | **0.15** |
| `v30_dropout` | 0.1 | **0.2** |
| `v29_physical_loss_warmup_epochs` | 0 | **2** |

All other components remain the same: v25 geometry fusion, v18 deformable cross-view attention, set-view aggregator, variable-view training, and outlier-view augmentation. TTE is **not** used. The physical-space temporal loss is enabled with a -epoch linear warmup so floor/bone/com-jitter terms only ramp up once training is stable.

The part-scale block receives three stacked `TransformerEncoderLayer`s instead of one. Stochastic depth is applied per block, and the higher dropout rate targets the attention/MLP paths. The gated residual and zeroed output projections keep the module close to identity at init, so the deeper path is effectively a residual refinement rather than a new un-trained branch.

## 3. Expected impact

On the local 4090 smoke (3 epochs, 200 train samples), the deeper hierarchy should extract finer part-level cross-view cues. We expect a small first-epoch improvement or parity, but the main gain is a slower overfit curve: if v30 smoke jumps from ~28 mm at epoch 1 to >45 mm by epoch 2, v31 should keep epoch 2 closer to ~35 mm thanks to stochastic depth and dropout.

If the smoke justifies the capacity, a full A800 run could close part of the remaining gap toward the v25 small baseline (~18 mm) while maintaining better few-view robustness. The physical-loss warmup should keep the floor/bone priors from dominating before the geometry head has stabilised.

## 4. Main risk and mitigation

**Risk: faster overfitting.** Three part-scale layers add parameters and give the model more ways to memorise the small smoke subset. We mitigate this with higher stochastic depth (drop whole blocks during training), higher dropout, and physical-loss warmup.

**Risk: memory / OOM on the RTX 4090.** Deeper attention on the part scale increases activation memory. The smoke script keeps `d=64`, `batch_size=4`, and `clip_len=9`; if OOM appears, the first knob to turn is `v30_n_part_layers` back to 2.

**Fallback:** If the smoke shows overfitting after epoch 1, reduce `v30_n_part_layers` to 2 and keep the extra regularisation; if it shows no gain, the spatial-depth hypothesis is rejected and we look elsewhere (e.g. learned part groups or cross-scale attention redesign) for v31.

## 5. Launch script

- `scripts/launch_v31_hierarchical_deeper_spatial_local4090.sh`
