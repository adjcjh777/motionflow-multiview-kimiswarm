"""Hierarchical attention with entropy regularisation for interpretable view selection.

Extends ``RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint``
with a per-joint attention-entropy penalty on the final per-view triangulation
weights.  Low entropy encourages the model to concentrate mass on a small subset
of views, which is easier to interpret and can improve robustness to noisy or
occluded cameras.
"""

import torch

from .ray_attention_hierarchical_view_temporal_joint_residual_principal_point_model import (
    RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint,
)


class RayAttentionFusionModelHierarchicalAttentionEntropyReg(
    RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint,
):
    """Hierarchical attention model with an interpretability-driven entropy loss.

    Parameters
    ----------
    attention_entropy_weight:
        Weight of the entropy penalty on the normalised per-view weights.
        ``0.0`` disables the penalty (default behaviour is unchanged).
    entropy_temperature:
        Softmax temperature applied before computing the entropy.  Lower values
        make the entropy penalty more aggressive.
    See ``RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint``
    for the remaining arguments.
    """

    def __init__(
        self,
        attention_entropy_weight: float = 0.01,
        entropy_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention_entropy_weight = attention_entropy_weight
        self.entropy_temperature = entropy_temperature

    def _entropy_regularization(self, weights: torch.Tensor) -> torch.Tensor:
        """Return the mean per-joint entropy of the per-view weight distribution.

        Args
        ----
        weights:
            ``(B, T, V, J)`` or ``(B*T, V, J)`` non-negative per-view weights.

        Returns
        -------
        Scalar entropy value (higher when the weight distribution is uniform).
        """
        # Normalise across views to obtain a probability distribution per joint.
        p = weights / (weights.sum(dim=-3, keepdim=True) + 1e-8)
        # Entropy per joint.
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=-3)  # (B, T, J) or (B*T, J)
        return entropy.mean()

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        # Re-use the base model forward logic to obtain predictions and weights.
        # We then compute the entropy penalty from the returned per-view weights.
        out = super().forward(x, cameras=cameras, K=K, R=R, t=t)
        pred_3d, weights = out[0], out[1]

        # Compute entropy on the returned per-view weights.  These weights are
        # already non-negative because they are sigmoid(confidences * visibility).
        entropy_loss = self._entropy_regularization(weights)
        entropy_loss = self.attention_entropy_weight * entropy_loss

        # The entropy loss is returned last so existing training scripts can
        # simply add it to the total loss.
        return (*out, entropy_loss)


if __name__ == "__main__":
    import numpy as np

    from ..calibration.camera import Camera

    def _make_cameras(n_views: int = 4):
        cameras = []
        for i in range(n_views):
            theta = 2 * np.pi * i / n_views
            c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
            forward = -c / np.linalg.norm(c)
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            R = np.stack([right, up, -forward], axis=0)
            t = -R @ c
            K = np.eye(3)
            K[0, 0] = K[1, 1] = 800.0
            K[0, 2] = 320.0
            K[1, 2] = 240.0
            cameras.append(Camera(K=K, R=R, t=t))
        return cameras

    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelHierarchicalAttentionEntropyReg(
        j=J, d=64, n_views=V, attention_entropy_weight=0.01
    )
    pred, weights, pp_delta, entropy_loss = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)
    assert entropy_loss.shape == ()
    loss = pred.mean() + entropy_loss
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("Hierarchical attention entropy-reg model smoke test passed")
