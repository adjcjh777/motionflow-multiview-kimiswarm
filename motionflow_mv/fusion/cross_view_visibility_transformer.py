"""Geometry-aware cross-view visibility transformer.

Predicts a per-view/per-joint soft visibility mask from spatio-temporal
features.  Unlike a single-view MLP, this module attends across views and
joints, allowing it to exploit geometric redundancy in a calibrated multi-view
rig and to make consistent occlusion decisions.
"""

import torch
import torch.nn as nn


class CrossViewVisibilityTransformer(nn.Module):
    """Cross-view visibility transformer head.

    Parameters
    ----------
    d:
        Feature dimension coming from the spatio-temporal encoder.
    n_heads:
        Number of attention heads inside the visibility transformer.
    n_layers:
        Number of transformer encoder layers.
    n_views:
        Number of camera views.
    j:
        Number of joints.
    dropout:
        Dropout probability for the transformer layers.
    """

    def __init__(
        self,
        d: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_views: int = 4,
        j: int = 17,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.j = j

        # Embed confidence as an additional channel so the transformer can
        # distinguish between low-confidence detections and true occlusions.
        self.feat_proj = nn.Linear(d + 1, d)

        # Learnable positional embeddings for view and joint identity.
        self.view_pos_embed = nn.Parameter(torch.randn(n_views, d) * 0.02)
        self.joint_pos_embed = nn.Parameter(torch.randn(j, d) * 0.02)

        # Small transformer over (view, joint) tokens.
        self.transformer = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=d * 2,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        # Final visibility logit.
        self.logit_head = nn.Linear(d, 1)

    def forward(
        self,
        feat: torch.Tensor,
        confidences: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-view/per-joint visibility logits.

        Parameters
        ----------
        feat:
            Spatio-temporal features of shape (N, V, J, d).
        confidences:
            Detector confidences of shape (N, V, J).

        Returns
        -------
        logits:
            Visibility logits of shape (N, V, J).
        """
        N, V, J, d = feat.shape
        assert d == self.d
        assert V == self.n_views
        assert J == self.j

        # Concatenate confidence as an extra feature.
        x = torch.cat([feat, confidences.unsqueeze(-1)], dim=-1)  # (N, V, J, d+1)
        x = self.feat_proj(x)  # (N, V, J, d)

        # Add view and joint positional embeddings.
        view_emb = self.view_pos_embed[:V].view(1, V, 1, d)
        joint_emb = self.joint_pos_embed[:J].view(1, 1, J, d)
        x = x + view_emb + joint_emb

        # Flatten (view, joint) into a single token sequence.
        x = x.reshape(N, V * J, d)
        for layer in self.transformer:
            x = layer(x)

        # Reshape and predict logits.
        x = x.reshape(N, V, J, d)
        logits = self.logit_head(x).squeeze(-1)  # (N, V, J)
        return logits


if __name__ == "__main__":
    # Lightweight shape/gradient sanity check.
    B, T, V, J, d = 2, 3, 4, 17, 64
    head = CrossViewVisibilityTransformer(d=d, n_views=V, j=J)
    feat = torch.randn(B * T, V, J, d)
    confidences = torch.rand(B * T, V, J)
    logits = head(feat, confidences)
    assert logits.shape == (B * T, V, J)

    loss = logits.sum()
    loss.backward()
    assert any(p.grad is not None for p in head.parameters())
    print("CrossViewVisibilityTransformer sanity check passed")
