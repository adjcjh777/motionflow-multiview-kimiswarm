"""Camera-conditioned view embedding.

Replaces fixed learned positional embeddings over camera indices with a
permutation-equivariant MLP that encodes calibrated camera intrinsics and
extrinsics.  This makes the model agnostic to view ordering and the number of
views, which is a prerequisite for variable-view inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CameraConditionedViewEmbedding(nn.Module):
    """Encode each view from its intrinsic and extrinsic parameters.

    Parameters
    ----------
    d:
        Output feature dimension for each view.
    camera_hidden:
        Hidden dimension of the two-layer MLP.
    """

    def __init__(self, d: int, camera_hidden: int = 32):
        super().__init__()
        self.d = d
        self.camera_hidden = camera_hidden
        self.mlp = nn.Sequential(
            nn.Linear(9 + 9 + 3, camera_hidden),
            nn.ReLU(),
            nn.Linear(camera_hidden, d),
        )

    def forward(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Encode camera parameters into per-view features.

        Args
        ----
        K:
            Intrinsics, ``(N, V, 3, 3)``.
        R:
            Rotation matrices, ``(N, V, 3, 3)``.
        t:
            Translation vectors, ``(N, V, 3)``.

        Returns
        -------
        Per-view embeddings of shape ``(N, V, d)``.
        """
        if K.dim() not in (3, 4):
            raise ValueError("K must be (V, 3, 3) or (N, V, 3, 3)")
        if K.dim() == 3:
            K = K.unsqueeze(0)
            R = R.unsqueeze(0)
            t = t.unsqueeze(0)

        N, V, _, _ = K.shape
        camera_feat = torch.cat(
            [
                K.reshape(N, V, -1),
                R.reshape(N, V, -1),
                t.reshape(N, V, -1),
            ],
            dim=-1,
        )  # (N, V, 21)
        return self.mlp(camera_feat)


if __name__ == "__main__":
    B, V, d = 2, 4, 64
    K = torch.randn(B, V, 3, 3)
    R = torch.randn(B, V, 3, 3)
    t = torch.randn(B, V, 3)

    embed = CameraConditionedViewEmbedding(d=d, camera_hidden=32)
    out = embed(K, R, t)
    assert out.shape == (B, V, d)

    # Permutation invariance of the *embedding values* (not ordering)
    perm = torch.randperm(V)
    out_perm = embed(K[:, perm], R[:, perm], t[:, perm])
    assert out_perm.shape == (B, V, d)
    # The embeddings should be permuted consistently with the input.
    assert torch.allclose(out[:, perm], out_perm)

    # Gradient sanity.
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in embed.parameters())
    print("CameraConditionedViewEmbedding CPU smoke test passed")
