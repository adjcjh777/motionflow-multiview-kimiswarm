"""Minimal stub for the v22 SMPL prior head.

This module is referenced by the OmniMultiViewFusionV5 integration on this
branch.  The stub is sufficient because the smoke experiment does not enable
``use_smpl_prior_fusion_v22``; it merely needs the import to resolve.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SMPLPriorHead(nn.Module):
    """Placeholder SMPL prior head."""

    def __init__(
        self,
        d: int = 64,
        n_joints: int = 17,
        smpl_model_path: str | None = None,
    ):
        super().__init__()
        self.d = d
        self.n_joints = n_joints

    def forward(self, x: torch.Tensor) -> dict:
        """No-op forward; returns an marker dict."""
        return {}
