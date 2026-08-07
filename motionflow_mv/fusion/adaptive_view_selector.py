"""Adaptive view selector for variable-view multi-view fusion.

``AdaptiveViewSelector`` predicts per-view, per-joint binary selection
masks.  During training it samples with Gumbel-softmax straight-through
top-``k`` selection; during inference it uses deterministic hard top-``k``.
A differentiable budget loss encourages the average number of selected views
to match a target budget.

The module is fully optional.  When ``use_selector=False`` the forward pass
returns an all-ones mask and a zero budget loss, so downstream triangulation
can bypass it without branch logic elsewhere.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveViewSelector(nn.Module):
    """Gumbel-softmax adaptive view selector.

    Parameters
    ----------
    d:
        Input feature dimension.
    n_views:
        Number of views in the fixed-view rig.
    n_joints:
        Number of body joints (used only for shape validation / diagnostics).
    target_k:
        Target number of active views per joint.  May be an integer in
        ``[1, n_views]`` or a float ratio in ``(0, 1]``.
    temperature:
        Temperature for the Gumbel-softmax relaxation.  Lower values make the
        selection sharper.
    budget_weight:
        Weight of the budget loss that penalises deviations of the mean selected
        view count from ``target_k``.
    hard_training:
        If True, the training forward pass returns a straight-through hard top-k
        mask.  If False, returns the soft softmax probabilities.
    use_selector:
        If False, the module always returns an all-ones mask and zero loss.
    """

    def __init__(
        self,
        d: int = 64,
        n_views: int = 4,
        n_joints: int = 17,
        target_k: Union[int, float] = 2,
        temperature: float = 0.5,
        budget_weight: float = 0.01,
        hard_training: bool = True,
        use_selector: bool = True,
    ):
        super().__init__()
        if d <= 0:
            raise ValueError("d must be positive")
        if n_views < 1:
            raise ValueError("n_views must be at least 1")

        self.d = d
        self.n_views = n_views
        self.n_joints = n_joints
        self.temperature = temperature
        self.budget_weight = budget_weight
        self.hard_training = hard_training
        self.use_selector = use_selector

        self._target_k_float = float(target_k)
        if 0.0 < target_k <= 1.0:
            self.target_k = max(1, min(n_views, round(target_k * n_views)))
        else:
            self.target_k = int(max(1, min(n_views, round(target_k))))

        self.logit_proj = nn.Linear(d, 1)

    def extra_repr(self) -> str:  # noqa: D401
        return (
            f"n_views={self.n_views}, n_joints={self.n_joints}, "
            f"target_k={self.target_k}, temperature={self.temperature}, "
            f"budget_weight={self.budget_weight}, use_selector={self.use_selector}"
        )

    def _gumbel_noise(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Return standard Gumbel(0,1) noise."""
        uniform = torch.rand(shape, device=device)
        # Clamp away from 0 and 1 so both logs stay finite, then compute
        # -log(-log(uniform)) in the correct order.
        eps = 1e-8
        uniform = uniform.clamp(min=eps, max=1.0 - eps)
        return -((-uniform.log()).log())

    def _topk_mask(
        self,
        logits: torch.Tensor,
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a top-k selection mask.

        Args
        ----
        logits:
            Per-view logits of shape ``(N, V, J)``.
        training:
            Whether to add Gumbel noise and use the softmax relaxation.

        Returns
        -------
        mask:
            Selected mask of shape ``(N, V, J)``.  During training this is the
            straight-through hard mask when ``hard_training`` is enabled.
        probs:
            Softmax probabilities of shape ``(N, V, J)``.
        hard_mask:
            Non-differentiable hard top-k mask of shape ``(N, V, J)``.
        """
        N, V, J = logits.shape
        if training:
            gumbel = self._gumbel_noise(logits.shape, logits.device)
            scaled = (logits + gumbel) / self.temperature
        else:
            scaled = logits / self.temperature

        probs = F.softmax(scaled, dim=1)

        # Hard top-k: select the k highest probability views per (N, J).
        topk = min(self.target_k, V)
        _, top_indices = torch.topk(probs, topk, dim=1)  # (N, k, J)
        hard_mask = torch.zeros_like(probs)
        hard_mask.scatter_(1, top_indices, 1.0)

        if training and self.hard_training:
            # Straight-through estimator: forward hard, backward through softmax.
            mask = hard_mask + probs - probs.detach()
        else:
            # Inference always uses a deterministic hard top-k mask.
            mask = hard_mask

        return mask, probs, hard_mask

    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args
        ----
        features:
            Per-view per-joint features of shape ``(N, V, J, d)``.

        Returns
        -------
        mask:
            Selection mask of shape ``(N, V, J)``.  In training this is a
            differentiable (straight-through) mask; in inference it is a hard
            one-hot top-k mask.
        budget_loss:
            Scalar budget loss.  Zero when ``use_selector=False`` or during
            inference.
        """
        if not self.use_selector:
            return torch.ones(
                features.shape[0],
                features.shape[1],
                features.shape[2],
                device=features.device,
                dtype=features.dtype,
            ), torch.tensor(0.0, device=features.device, dtype=features.dtype)

        if features.dim() != 4:
            raise ValueError("features must have shape (N, V, J, d)")

        N, V, J, d = features.shape
        if V != self.n_views or J != self.n_joints:
            raise ValueError(
                f"expected (V={self.n_views}, J={self.n_joints}), got ({V}, {J})"
            )

        logits = self.logit_proj(features).squeeze(-1)  # (N, V, J)
        mask, probs, _ = self._topk_mask(logits, training=self.training)

        # Budget loss on the *soft* probabilities so it has gradients.
        if self.training:
            selected_count = probs.sum(dim=1)  # (N, J)
            budget_loss = (
                (selected_count.mean() - self.target_k) ** 2
                * self.budget_weight
            )
        else:
            budget_loss = torch.tensor(
                0.0, device=features.device, dtype=features.dtype
            )

        return mask, budget_loss


if __name__ == "__main__":
    # CPU smoke test: V=4, J=17, target_k=2.
    V, J, d = 4, 17, 64
    selector = AdaptiveViewSelector(
        d=d,
        n_views=V,
        n_joints=J,
        target_k=2,
        temperature=0.5,
        budget_weight=0.1,
    )

    # Training path with Gumbel-softmax straight-through top-k.
    selector.train()
    x = torch.rand(8, V, J, d)
    mask, budget_loss = selector(x)
    assert mask.shape == (8, V, J), mask.shape
    assert torch.allclose(mask.sum(dim=1), torch.full((8, J), 2.0), atol=1e-5)
    assert budget_loss.item() >= 0.0
    print(f"training mask per-joint view count: {mask[0].sum(dim=0).unique()}")

    # Inference path: deterministic hard top-k.
    selector.eval()
    with torch.no_grad():
        mask_inf, budget_loss_inf = selector(x)
    assert mask_inf.shape == (8, V, J)
    assert torch.allclose(
        mask_inf.sum(dim=1), torch.full((8, J), 2.0), atol=1e-5
    )
    assert set(mask_inf.unique().tolist()).issubset({0.0, 1.0})
    print("inference hard top-k mask passed")

    # Gradient sanity check.
    selector.train()
    x_grad = torch.rand(4, V, J, d, requires_grad=True)
    mask, budget_loss = selector(x_grad)
    loss = mask.mean() + budget_loss
    loss.backward()
    assert any(p.grad is not None for p in selector.parameters())
    print("adaptive view selector CPU smoke test passed")
