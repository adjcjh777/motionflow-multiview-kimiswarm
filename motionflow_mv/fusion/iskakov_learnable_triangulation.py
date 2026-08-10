"""Learnable Triangulation baseline (Iskakov et al., ICCV 2019).

Faithful standalone re-implementation of the weight-prediction branch of:

    Iskakov, D., Kasneci, E., 'Learnable Triangulation of Human Pose', ICCV 2019.

The method keeps the classic DLT triangulation closed-form but learns small
per-view, per-joint weights with a shared MLP over 2D-detection features, so
the network learns to down-weight unreliable views/joints.  Gradients flow
through the differentiable weighted least-squares solve
(``motionflow_mv.fusion.triangulation.triangulate_dlt_batched_lstsq``).

Design choices (documented deviations from the paper):

* The paper extracts deep per-view features from a backbone; this protocol
  uses off-the-shelf 2D detections, so the per-view per-joint feature vector
  is built from raw detection statistics:
  ``[u_norm, v_norm, confidence, mean_u_norm, mean_v_norm, mean_conf,
     ray_dist_m]`` (7 features).  ``ray_dist_m`` is the distance from the
  camera centre to the 3D point on the viewing ray at the training set's mean
  skeleton height -- a calibration-aware scale cue (Shelf and Campus have very
  different focal lengths / pixel scales).
* ``cross_view=True`` (main variant, matching the paper's design where the
  confidence volume is produced jointly over all views): per-view features
  are concatenated with their cross-view mean and passed through a shared MLP.
  ``cross_view=False`` yields the simpler per-view-only variant.
* The MLP's final layer is zero-initialised, so at initialisation
  ``logit = 0 -> sigmoid = 0.5`` and every view contributes equally; the
  frozen confidence-weighted DLT is reported by the trainer as the reference
  the learned weights must beat.
* Weights are ``sigmoid`` in (0, 1); the weighted DLT routine applies
  ``sqrt(w)`` row scaling internally.

No temporal information is used, matching the frame-wise baseline protocol of
``docs/results_true_gt_shelf_campus.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq

# Feature layout indices (for documentation / debugging).
FEATURE_NAMES = (
    "u_norm",
    "v_norm",
    "confidence",
    "mean_u_norm",
    "mean_v_norm",
    "mean_conf",
    "ray_dist_m",
)
NUM_FEATURES = len(FEATURE_NAMES)


def build_features(
    points_2d: torch.Tensor,
    confidences: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    default_ray_depth: float = 3.0,
) -> torch.Tensor:
    """Build per-view per-joint features.

    Args:
        points_2d: (N, V, J, 2) 2D keypoints in pixels.
        confidences: (N, V, J) non-negative detection confidences.
        K: (V, 3, 3) camera intrinsics.
        R: (V, 3, 3) world-to-camera rotations.
        t: (V, 3) world-to-camera translations (camera model: P = K [R | t]).
        default_ray_depth: metres; fallback ray distance when no finite
            camera-centre height is available.

    Returns:
        (N, V, J, NUM_FEATURES) feature tensor in the input dtype.
    """
    if points_2d.dim() != 4:
        raise ValueError(f"points_2d must be (N, V, J, 2), got {tuple(points_2d.shape)}")
    N, V, J, _ = points_2d.shape
    dtype = points_2d.dtype
    device = points_2d.device

    fx = K[:, 0, 0].to(device=device, dtype=dtype).view(1, V, 1)
    fy = K[:, 1, 1].to(device=device, dtype=dtype).view(1, V, 1)
    cx = K[:, 0, 2].to(device=device, dtype=dtype).view(1, V, 1)
    cy = K[:, 1, 2].to(device=device, dtype=dtype).view(1, V, 1)

    # Normalised image coordinates (~[-1, 1] inside the frame).
    u_norm = (points_2d[..., 0] - cx) / fx.clamp(min=1e-6)
    v_norm = (points_2d[..., 1] - cy) / fy.clamp(min=1e-6)

    mean_u = u_norm.mean(dim=1, keepdim=True).expand(-1, V, -1)
    mean_v = v_norm.mean(dim=1, keepdim=True).expand(-1, V, -1)
    mean_c = confidences.mean(dim=1, keepdim=True).expand(-1, V, -1)

    # Camera centre in world coordinates: C = -R^T t.
    C = -torch.einsum("vij,vj->vi", R.to(device=device, dtype=dtype).transpose(-2, -1), t.to(device=device, dtype=dtype))  # (V, 3)
    up = -C[:, 2]  # world z of the camera centre (z-up scenes)
    # Finite only for cameras below/above the world origin plane.
    scale = torch.where(torch.isfinite(up) & (up.abs() > 1e-3), up, torch.zeros_like(up))
    ray_dist = torch.full((1, V, 1), float(default_ray_depth), device=device, dtype=dtype)
    nonzero = scale.abs() > 1e-3
    if bool(nonzero.any()):
        ray_dist = ray_dist.clone()
        ray_dist[:, nonzero, :] = scale[nonzero].view(1, -1, 1).abs().to(dtype)
    ray_dist = ray_dist.expand(N, -1, J)

    feats = torch.stack(
        [u_norm, v_norm, confidences.to(dtype), mean_u, mean_v, mean_c, ray_dist],
        dim=-1,
    )
    return feats


class IskakovLearnableTriangulation(nn.Module):
    """Learnable per-view weights + weighted DLT triangulation.

    Args:
        hidden_dim: width of the shared MLP (default 32; ~2.4k params total).
        cross_view: if True (default, main variant), each view's feature
            vector is concatenated with the cross-view mean of all features
            before the shared MLP.
        default_ray_depth: fallback ray-distance feature in metres.
    """

    def __init__(self, hidden_dim: int = 32, cross_view: bool = True, default_ray_depth: float = 3.0):
        super().__init__()
        self.cross_view = cross_view
        self.default_ray_depth = default_ray_depth
        in_dim = NUM_FEATURES * (2 if cross_view else 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-init the final layer: at init logit == 0, so all weights are
        # equal (0.5) and the model reduces to unweighted DLT.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def predict_weights(self, features: torch.Tensor) -> torch.Tensor:
        """Map features to per-view per-joint weights in (0, 1).

        Args:
            features: (N, V, J, NUM_FEATURES).

        Returns:
            (N, V, J) weights.
        """
        N, V, J, F = features.shape
        if self.cross_view:
            cross = features.mean(dim=1, keepdim=True).expand(-1, V, -1, -1)
            mlp_in = torch.cat([features, cross], dim=-1)  # (N, V, J, 2F)
        else:
            mlp_in = features
        logit = self.mlp(mlp_in).squeeze(-1)  # (N, V, J)
        return torch.sigmoid(logit)

    def forward(
        self,
        points_2d: torch.Tensor,
        confidences: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        return_weights: bool = False,
    ):
        """Triangulate 3D joints from per-view detections.

        Args:
            points_2d: (N, V, J, 2) pixels.
            confidences: (N, V, J).
            K: (V, 3, 3), R: (V, 3, 3), t: (V, 3) shared cameras.
            return_weights: also return the predicted (N, V, J) weights.

        Returns:
            X: (N, J, 3) triangulated points in the same units as the
            cameras (metres for this protocol); optionally (X, weights).
        """
        P = build_projection_matrices(K, R, t)
        feats = build_features(points_2d, confidences, K, R, t, self.default_ray_depth)
        weights = self.predict_weights(feats)
        X = triangulate_dlt_batched_lstsq(points_2d, P, weights=weights)
        if return_weights:
            return X, weights
        return X


def build_projection_matrices(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(V, 3, 4) projection matrices P = K [R | t] (world-to-camera R, t)."""
    Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (V, 3, 4)
    return K @ Rt
