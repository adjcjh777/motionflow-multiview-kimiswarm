"""Differentiable Procrustes alignment loss for 3D poses.

The loss Procrustes-aligns the prediction to the ground truth per frame (rigid
similarity: translation + rotation + uniform scale) and then computes the MSE.
This directly optimises the metrics used in HPE benchmarks (PA-MPJPE).
"""

import torch
import torch.nn.functional as F


def procrustes_align(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Align ``pred`` to ``target`` with a rigid similarity transform.

    Args:
        pred: ``(B, J, 3)`` predicted 3D poses.
        target: ``(B, J, 3)`` ground-truth 3D poses.
        eps: small constant for numerical stability.

    Returns:
        ``(B, J, 3)`` aligned predictions.
    """
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)

    pred_c = pred - pred_mean
    target_c = target - target_mean

    # Optimal uniform scale: argmin_s ||s*p - t||^2 = (p·t) / (p·p)
    pred_norm = (pred_c ** 2).sum(dim=(1, 2), keepdim=True)
    scale = (pred_c * target_c).sum(dim=(1, 2), keepdim=True) / (pred_norm + eps)
    pred_c = pred_c * scale.unsqueeze(-1)

    # Cross-covariance for the optimal rotation.
    H = torch.bmm(pred_c.transpose(1, 2), target_c)  # (B, 3, 3)
    U, _, Vt = torch.linalg.svd(H)

    # Rotation: R = V U^T, with reflection correction.
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
    det = torch.det(R)
    # For reflections, flip the last singular value.
    if (det < 0).any():
        Vt_neg = Vt.clone()
        Vt_neg[:, -1, :] = Vt_neg[:, -1, :] * -1.0
        R_neg = torch.bmm(Vt_neg.transpose(1, 2), U.transpose(1, 2))
        R = torch.where((det < 0).view(-1, 1, 1), R_neg, R)

    aligned = torch.bmm(pred_c, R) + target_mean
    return aligned


def procrustes_mse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """MSE between the target and the Procrustes-aligned prediction.

    Args:
        pred: ``(B, J, 3)`` predictions.
        target: ``(B, J, 3)`` ground truth.
        eps: stability constant.

    Returns:
        Scalar loss.
    """
    pred_aligned = procrustes_align(pred, target, eps=eps)
    return F.mse_loss(pred_aligned, target)


if __name__ == "__main__":
    B, J = 4, 17
    pred = torch.randn(B, J, 3, requires_grad=True)
    target = torch.randn(B, J, 3)
    loss = procrustes_mse_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    print("Procrustes MSE loss smoke test passed", loss.item())
