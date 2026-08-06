"""Lightweight per-view/per-joint dynamic view-selection gate.

The gate consumes the post-attention feature tokens and predicts a soft scalar
``g_vj`` in ``[0, 1]`` for every view and joint.  The gate is then multiplied with
the triangulation weights before weighted DLT, allowing the model to down-weight
or ignore noisy/occluded views on a per-joint basis.
"""

import torch
import torch.nn as nn


class DynamicViewSelectionGate(nn.Module):
    """Predict a per-view/per-joint soft gate for multi-view triangulation.

    Parameters
    ----------
    d:
        Feature dimension of the input tokens.
    n_views:
        Number of camera views (kept for API symmetry; not used internally).
    """

    def __init__(self, d: int, n_views: int = None):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, feat):
        """Forward pass.

        Args
        ----
        feat:
            Post-attention feature tensor of shape ``(B*T, V, J, d)``.

        Returns
        -------
        gate_weights:
            Soft gate ``(B*T, V, J)`` in ``[0, 1]``.
        gate_logits:
            Pre-sigmoid logits ``(B*T, V, J)`` for loss computation.
        """
        logits = self.gate_mlp(feat).squeeze(-1)  # (B*T, V, J)
        gate = torch.sigmoid(logits)
        return gate, logits
