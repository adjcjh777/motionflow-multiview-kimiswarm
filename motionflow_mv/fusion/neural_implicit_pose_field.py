"""Neural implicit 3-D pose field for multi-view skeleton refinement.

The module provides:

* ``SinusoidalPositionalEncoding`` – sinusoidal positional encoding for 3-D
  coordinates.
* ``NeuralImplicitPoseField`` – a small MLP that maps a 3-D position, a
  joint-id embedding, and a per-joint feature to a scalar field value.
* ``NeuralImplicitPoseFieldRefiner`` – a drop-in replacement for the dense
  ``residual_mlp`` in the anchor model. It receives the same ``(N, J, d+3)``
  concatenation of pooled spatio-temporal features and the raw triangulated 3-D
  pose, and refines the pose by stepping along the field gradient toward the
  zero level-set.

The refiner is intentionally lightweight: a few hidden layers, a small
positional encoding, and a single Newton-style refinement step by default.  This
keeps the change minimally invasive and easy to validate on a short smoke run.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for 3-D coordinates.

    Parameters
    ----------
    in_dim:
        Coordinate dimension (default 3).
    num_freq:
        Number of frequency octaves.  For octave ``i`` the encoding uses
        ``sin(π·2^i·x)`` and ``cos(π·2^i·x)``.
    """

    def __init__(self, in_dim: int = 3, num_freq: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.num_freq = num_freq
        # Coordinate + ``num_freq`` sin/cos pairs per coordinate.
        self.out_dim = in_dim * (2 * num_freq + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positional encoding of ``x``.

        Args:
            x: ``(..., in_dim)`` coordinate tensor.

        Returns:
            ``(..., out_dim)`` encoded tensor.
        """
        features = [x]
        for i in range(self.num_freq):
            freq = (2 ** i) * torch.pi
            features.append(torch.sin(freq * x))
            features.append(torch.cos(freq * x))
        return torch.cat(features, dim=-1)


class NeuralImplicitPoseField(nn.Module):
    """Joint-conditioned neural implicit field.

    The network predicts a scalar field value ``f(pos, feat, joint_id)``.
    When trained, the zero level-set ``f = 0`` should coincide with the true
    3-D joint location.

    Parameters
    ----------
    j:
        Number of joints.
    feat_dim:
        Dimension of the per-joint conditioning feature.
    hidden_dim:
        Hidden width of the MLP.
    num_layers:
        Total number of linear layers (including the final output layer).
    embed_dim:
        Dimension of the learnable joint-id embedding.
    pe_freq:
        Number of positional-encoding octaves passed to
        ``SinusoidalPositionalEncoding``.
    """

    def __init__(
        self,
        j: int = 17,
        feat_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        embed_dim: int = 16,
        pe_freq: int = 6,
    ):
        super().__init__()
        self.j = j
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        self.pos_enc = SinusoidalPositionalEncoding(in_dim=3, num_freq=pe_freq)
        self.joint_embed = nn.Embedding(j, embed_dim)

        in_dim = self.pos_enc.out_dim + embed_dim + feat_dim
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.field_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        pos: torch.Tensor,
        feat: torch.Tensor,
        joint_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Query the implicit field.

        Args:
            pos: ``(N, J, 3)`` 3-D coordinates.
            feat: ``(N, J, feat_dim)`` per-joint conditioning features.
            joint_ids: Optional ``(N, J)`` integer joint ids.  If ``None``, ids
                ``0 … J-1`` are assumed for every sample.

        Returns:
            ``(N, J)`` scalar field values.
        """
        N, J, _ = pos.shape
        if joint_ids is None:
            joint_ids = torch.arange(J, device=pos.device, dtype=torch.long)
            joint_ids = joint_ids.unsqueeze(0).expand(N, -1)

        pe = self.pos_enc(pos)  # (N, J, pe_dim)
        j_emb = self.joint_embed(joint_ids)  # (N, J, embed_dim)
        x = torch.cat([pe, j_emb, feat], dim=-1)  # (N, J, in_dim)
        h = self.mlp(x)  # (N, J, hidden_dim)
        return self.field_head(h).squeeze(-1)  # (N, J)


class NeuralImplicitPoseFieldRefiner(nn.Module):
    """Implicit-field residual refiner.

    Replaces the dense ``residual_mlp`` in the anchor model.  It receives the
    concatenation ``[feat, raw_3d]`` and returns a residual correction
    ``refined_3d - raw_3d``.

    Parameters
    ----------
    j:
        Number of joints.
    feat_dim:
        Dimension of the per-joint feature (``d`` in the anchor).
    hidden_dim, num_layers, pe_freq:
        Passed to ``NeuralImplicitPoseField``.
    n_iters:
        Number of Newton-style field-refinement steps.
    step_size:
        Step-size multiplier for each Newton step.
    return_field:
        If ``True``, ``forward`` also returns the final field values for
        auxiliary losses.
    capture_field:
        If ``True``, retain the final values for a containing model while
        keeping the tensor-only return contract.
    """

    def __init__(
        self,
        j: int = 17,
        feat_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        n_iters: int = 1,
        step_size: float = 0.5,
        pe_freq: int = 6,
        return_field: bool = False,
        capture_field: bool = False,
    ):
        super().__init__()
        self.field = NeuralImplicitPoseField(
            j=j,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pe_freq=pe_freq,
        )
        self.n_iters = n_iters
        self.step_size = step_size
        self.return_field = return_field
        self.capture_field = capture_field
        self.last_field_values = None

    def forward(self, residual_input: torch.Tensor) -> torch.Tensor:
        """Refine the raw triangulated pose toward the implicit field zero-set.

        Args:
            residual_input: ``(N, J, d + 3)`` concatenation of per-joint feature
                ``(N, J, d)`` and raw 3-D position ``(N, J, 3)``.

        Returns:
            ``(N, J, 3)`` residual correction.  If ``return_field=True``, returns
            a tuple ``(delta, field_values)``.
        """
        feat = residual_input[..., :-3]
        pos = residual_input[..., -3:]
        pos0 = pos
        outer_grad_enabled = torch.is_grad_enabled()

        field_values = None
        for _ in range(self.n_iters):
            with torch.enable_grad():
                if not outer_grad_enabled:
                    pos = pos.detach()
                pos = pos.requires_grad_(True)
                f = self.field(pos, feat)  # (N, J)
                grad = torch.autograd.grad(
                    f.sum(),
                    pos,
                    create_graph=outer_grad_enabled,
                    retain_graph=outer_grad_enabled,
                )[0]  # (N, J, 3)

            # Detach the normal direction so backward is first-order w.r.t. the
            # field network; the update still depends on the field value itself.
            grad = grad.detach()
            grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (N, J, 1)
            unit_grad = grad / grad_norm
            # Newton step: move along the unit normal by f / ||grad||.
            pos = pos - self.step_size * (f.unsqueeze(-1) / grad_norm) * unit_grad
            field_values = f if outer_grad_enabled else f.detach()

        delta = pos - pos0
        if not outer_grad_enabled:
            delta = delta.detach()
            if field_values is not None:
                field_values = field_values.detach()
        if self.capture_field:
            self.last_field_values = field_values
        if self.return_field:
            return delta, field_values
        return delta


def _smoke_test():
    """CPU sanity check: forward + backward through the refiner."""
    torch.manual_seed(0)
    N, J, d = 2, 17, 32
    field_refiner = NeuralImplicitPoseFieldRefiner(
        j=J, feat_dim=d, hidden_dim=64, num_layers=2, n_iters=1, step_size=0.5
    )
    optimizer = torch.optim.Adam(field_refiner.parameters(), lr=1e-3)

    feat = torch.randn(N, J, d)
    pos = torch.randn(N, J, 3)
    target = torch.randn(N, J, 3)
    residual_input = torch.cat([feat, pos], dim=-1)

    delta = field_refiner(residual_input)
    pred = pos + delta
    loss = (pred - target).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"NeuralImplicitPoseFieldRefiner smoke test passed (loss={loss.item():.4f})")


if __name__ == "__main__":
    _smoke_test()
