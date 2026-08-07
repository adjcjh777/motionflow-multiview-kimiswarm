"""OmniMultiViewFusion prototype (swarm iter18 / P17).

This module is intentionally a skeleton. It wires the existing building blocks
into a single forward signature so that the issue draft in
``docs/swarm_iter18/P17_github_issue_draft.md`` has a concrete code reference.
Run this file directly for a CPU smoke test.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]


class OmniMultiViewFusion(nn.Module):
    """Unified multi-view fusion backbone for swarm iter18 / P17.

    Parameters
    ----------
    d_model:
        Backbone feature dimension.
    residual_hidden:
        Hidden size of the residual refinement head.
    num_joints:
        Number of joints (default 28 for MPI-INF-3DHP).
    max_views:
        Maximum number of camera views.
    max_offset:
        Bound for the principal-point correction head.
    """

    def __init__(
        self,
        d_model: int = 64,
        residual_hidden: int = 128,
        num_joints: int = 28,
        max_views: int = 14,
        max_offset: float = 20.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.residual_hidden = residual_hidden
        self.num_joints = num_joints
        self.max_views = max_views

        # Existing components are imported lazily so the prototype is testable
        # even when not all production modules are perfectly aligned.
        try:
            import sys

            sys.path.insert(0, str(REPO_ROOT))
            from motionflow_mv.fusion.principal_point_correction import (
                PrincipalPointCorrection,
            )

            self.pp_correction = PrincipalPointCorrection(
                d=d_model, hidden=d_model, max_offset=max_offset
            )
        except Exception as exc:  # pragma: no cover - smoke only
            raise ImportError(
                "PrincipalPointCorrection not available; "
                "run from repo root with motionflow_mv on PYTHONPATH."
            ) from exc

        # Project the raw 2D + confidence (3 channels) up to d_model.
        self.feat_proj = nn.Linear(3, d_model)

        # Stub heads for the new directions. These will be replaced by real
        # implementations as the issue progresses.
        self.visibility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self.weight_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Placeholder residual refinement MLP.
        self.residual_mlp = nn.Sequential(
            nn.Linear(3 + d_model, residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, 3),
        )

    def forward(
        self,
        keypoints_2d: torch.Tensor,
        confidence: torch.Tensor,
        cameras: dict,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning a dict with 3D pose and auxiliary outputs.

        Parameters
        ----------
        keypoints_2d:
            Tensor of shape ``(B, T, V, J, 2)``.
        confidence:
            Tensor of shape ``(B, T, V, J)``.
        cameras:
            Dictionary with at least ``K`` of shape ``(B, V, 3, 3)``.

        Returns
        -------
        Dict with keys ``pose_3d``, ``visibility``, ``weights``, ``log_var``.
        """
        B, T, V, J, _ = keypoints_2d.shape
        assert J == self.num_joints, f"Expected {self.num_joints} joints, got {J}"
        assert V <= self.max_views, f"Expected <= {self.max_views} views, got {V}"

        K = cameras["K"]

        # 1. Principal-point correction (placeholder feature used).
        raw_feat = torch.cat([keypoints_2d, confidence[..., None]], dim=-1)  # (B, T, V, J, 3)
        pp_x = raw_feat.mean(dim=1)  # (B, V, J, 3)
        K_corrected, _ = self.pp_correction(K, x=pp_x)

        # 2. Project raw feature and pool over time for the stub heads.
        proj_feat = self.feat_proj(raw_feat)  # (B, T, V, J, d_model)
        vis_feat = proj_feat.mean(dim=1)  # (B, V, J, d_model)
        visibility = self.visibility_head(vis_feat).squeeze(-1)  # (B, V, J)
        weights = self.weight_head(vis_feat).squeeze(-1)  # (B, V, J)
        log_var = self.uncertainty_head(vis_feat).squeeze(-1)  # (B, V, J)

        # 3. Triangulation stub: simple weighted average of lifted rays.
        #    The full model replaces this with Bayesian DLT + Gauss-Newton.
        ones = torch.ones_like(keypoints_2d[..., :1])
        rays = torch.cat([keypoints_2d, ones], dim=-1)  # (B, T, V, J, 3)
        w = weights[:, None, :, :] * visibility[:, None, :, :] * confidence  # (B, T, V, J)
        w = w / (w.sum(dim=2, keepdim=True) + 1e-8)  # normalize over views
        pose_3d = (rays * w[..., None]).sum(dim=2)  # (B, T, J, 3)

        # 4. Residual refinement stub.
        global_feat = proj_feat.mean(dim=(1, 2, 3))  # (B, d_model)
        delta = self.residual_mlp(
            torch.cat([pose_3d, global_feat[:, None, None, :].expand(B, T, J, -1)], dim=-1)
        )
        pose_3d = pose_3d + delta

        # K is returned for inspection.
        return {
            "pose_3d": pose_3d,
            "visibility": visibility,
            "weights": weights,
            "log_var": log_var,
            "K_corrected": K_corrected,
        }


def _cpu_smoke() -> None:
    """Instantiate the model and run one forward pass on CPU."""
    model = OmniMultiViewFusion(d_model=64, residual_hidden=128)

    B, T, V, J = 2, 9, 4, 28
    keypoints_2d = torch.randn(B, T, V, J, 2)
    confidence = torch.rand(B, T, V, J)
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)

    out = model(keypoints_2d, confidence, {"K": K})

    assert out["pose_3d"].shape == (B, T, J, 3)
    assert out["visibility"].shape == (B, V, J)
    assert out["weights"].shape == (B, V, J)
    assert out["log_var"].shape == (B, V, J)
    assert out["K_corrected"].shape == (B, V, 3, 3)

    print("[OK] OmniMultiViewFusion CPU smoke test passed.")
    print(f"     pose_3d shape: {out['pose_3d'].shape}")


if __name__ == "__main__":
    _cpu_smoke()
