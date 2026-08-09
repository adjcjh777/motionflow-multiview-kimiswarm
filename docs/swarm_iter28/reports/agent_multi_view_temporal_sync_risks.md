# Agent Risk Report: v54 Multi-View Temporal Sync

## 1. Long-sequence memory and runtime blow-up

**Risk.** The MVTS attention is over `(B, J, T, d)` tokens. For full-sequence training the codebase often uses `T = 243` or more; materializing the full `(T, T)` attention matrix for every joint can sharply increase memory and slow training compared with the frame-wise v52/v53 blocks.

**Mitigation.** Default `v54_mvts_window` to a small causal window (e.g., `7` or `13` frames) so attention is `O(T \cdot window)` instead of `O(T^2)`. Provide a `null` option only for short-clip smoke tests. Benchmark memory on the RTX 4090 smoke before allowing `window=null` in full A800 runs.

## 2. Over-smoothing / loss of high-frequency motion

**Risk.** The temporal smoothness loss can suppress legitimate fast motion (e.g., sprinting, ball strikes), causing MPJPE to rise on high-dynamics sequences.

**Mitigation.** Weight the smoothness term by the UWT reliability `\bar{w}_{t,j}` so low-confidence frames are regularized more strongly while high-confidence frames retain their original detail. Gate the residual with `sigmoid(-6.0)` at init so the module grows into the correction gradually. Add a smoke ablation that disables the temporal loss and compares `MPJPE` on the H36M high-activity subset.

## 3. Broken identity-at-init when stacked on v53

**Risk.** If the residual gate is not initialized correctly, or if the attention layer projects the pose before the residual branch, a v54-enabled model loaded from a v53 checkpoint can shift MPJPE by more than `0.1 mm` and break reproducibility.

**Mitigation.** Zero-initialize the final residual MLP layer and use `residual_gate_init=-6.0` (sigmoid ≈ 0.0025). Add a unit test in `tests/test_v54_identity_at_init.py` that loads a v53 checkpoint into a v54-enabled model and asserts `|MPJPE_v54 - MPJPE_v53| < 0.1 mm`.

## 4. Dependency on v52 UWT weight quality

**Risk.** MVTS treats v52 UWT weights as the temporal synchronization signal. If v52 produces over-confident or noisy weights, the module may propagate errors across time instead of correcting them.

**Mitigation.** Add a fallback path: when `use_v54_multi_view_temporal_sync=true` but `use_v52_uncertainty_weighted_triangulation=false`, use uniform weights `1/V` and a learned per-joint reliability token. Monitor the entropy of the sync attention weights during training; if entropy collapses, add a small entropy bonus to keep temporal borrowing diverse.

## 5. Integration complexity with downstream residual MLP / v51 ensemble

**Risk.** `OmniMultiViewFusionV5` already chains v25 → v52 → v53 before the final residual MLP and optional v51 domain-agnostic ensemble. Inserting v54 changes the shape and gradient path to every downstream block, and a bug in loss gating or tensor reshaping can silently zero the v54 contribution.

**Mitigation.** Keep the v54 output shape identical to its input `(B*T, J, 3)` and place it immediately after the v53 block, before the residual MLP. Follow the existing pattern used for v52/v53: store the auxiliary loss in `self._v54_mvts_loss`, gate it with `v54_mvts_warmup_epochs`, and add it to `epi_loss`. Add a smoke test that trains one step with `v54_mvts_loss_weight=0` and another with `>0`, checking that the loss changes.
