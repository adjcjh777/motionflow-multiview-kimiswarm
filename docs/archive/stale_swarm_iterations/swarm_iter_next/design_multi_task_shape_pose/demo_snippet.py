"""
Minimal prototype: SMPL shape/pose head plugged into the pooled feature of the
existing ray-attention fusion model.  This snippet is intentionally lightweight
and runs a synthetic forward/backward check without training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Optional: smplx may not be installed in every environment.
try:
    import smplx
    HAS_SMPLX = True
except Exception:  # pragma: no cover
    HAS_SMPLX = False


class SMPLShapePoseHead(nn.Module):
    """Predict shared clip-level betas and per-frame SMPL pose/transl.

    Parameters
    ----------
    d : int
        Dimension of the pooled per-frame feature.
    smpl_model_path : str
        Path to ``SMPL_NEUTRAL.pkl``.
    """

    def __init__(self, d: int, smpl_model_path: str):
        super().__init__()
        if not HAS_SMPLX:
            raise ImportError("smplx is required for the SMPL shape/pose head.")

        self.d = d
        self.betas_head = nn.Linear(d, 10, bias=True)
        self.body_pose_head = nn.Linear(d, 69, bias=True)
        self.global_orient_head = nn.Linear(d, 3, bias=True)
        # Initialize transl near zero; the pelvis offset can be added externally.
        self.transl_head = nn.Linear(d, 3, bias=True)

        # Cache the SMPL layer; batch_size is set per forward call.
        self._smpl_path = smpl_model_path
        self.smpl_model: smplx.SMPL | None = None

    def _load_smpl(self, batch_size: int, device: torch.device) -> smplx.SMPL:
        if self.smpl_model is None or self.smpl_model.batch_size != batch_size:
            self.smpl_model = smplx.SMPL(self._smpl_path, batch_size=batch_size).to(device)
        return self.smpl_model

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Args:
            feat: (B*T, d) pooled feature.

        Returns:
            dict with keys ``betas, body_pose, global_orient, transl,
            smpl_joints, pred_joints_17``.
        """
        N = feat.shape[0]
        device = feat.device

        # Shared clip-level shape: average over the temporal batch.
        betas = self.betas_head(feat).mean(dim=0, keepdim=True)  # (1, 10)
        body_pose = self.body_pose_head(feat)                     # (N, 69)
        global_orient = self.global_orient_head(feat)             # (N, 3)
        transl = self.transl_head(feat)                           # (N, 3)

        smpl_model = self._load_smpl(N, device)
        output = smpl_model(
            betas=betas.expand(N, -1),
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
        )

        # First 17 joints correspond to the H36M-style skeleton used by the repo.
        pred_joints_17 = output.joints[:, :17, :]  # (N, 17, 3)

        return {
            "betas": betas,
            "body_pose": body_pose,
            "global_orient": global_orient,
            "transl": transl,
            "smpl_joints": output.joints,
            "pred_joints_17": pred_joints_17,
        }


def main():
    if not HAS_SMPLX:
        print("smplx not available; install it to run this demo.")
        return

    d = 64
    N = 4  # batch x temporal samples
    feat = torch.randn(N, d, requires_grad=True)

    head = SMPLShapePoseHead(d, smpl_model_path="data/smpl/SMPL_NEUTRAL.pkl")
    out = head(feat)

    assert out["betas"].shape == (1, 10)
    assert out["body_pose"].shape == (N, 69)
    assert out["global_orient"].shape == (N, 3)
    assert out["transl"].shape == (N, 3)
    assert out["pred_joints_17"].shape == (N, 17, 3)

    loss = out["pred_joints_17"].mean()
    loss.backward()
    assert feat.grad is not None
    print("SMPL shape/pose head prototype passed shape and gradient checks.")


if __name__ == "__main__":
    main()
