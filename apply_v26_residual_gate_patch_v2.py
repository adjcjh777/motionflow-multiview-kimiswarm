"""Apply v26 residual gate patch and commit to the current branch."""
import subprocess
import sys


def replace_in_file(path: str, old: str, new: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old not in content:
        print(f"[WARN] Pattern not found in {path}", file=sys.stderr)
        return False
    content = content.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Patched {path}")
    return True


def main() -> None:
    tgf_path = "motionflow_mv/fusion/temporal_geometry_fusion_v26.py"
    omni_path = "motionflow_mv/fusion/omniview_fusion_v5.py"
    train_path = "experiments/train_omniview_fusion_v5_webbridge_multi.py"
    test_path = "tests/test_temporal_geometry_fusion_v26.py"

    # temporal_geometry_fusion_v26.py
    ok = True
    ok &= replace_in_file(
        tgf_path,
        "    temporal_window:\n"
        "        Size of the temporal window; must be odd. Default ``3`` corresponds to\n"
        "        offsets ``[-1, 0, +1]``.\n"
        "    dropout:\n"
        "        Dropout probability on the output projection.",
        "    temporal_window:\n"
        "        Size of the temporal window; must be odd. Default ``3`` corresponds to\n"
        "        offsets ``[-1, 0, +1]``.\n"
        "    dropout:\n"
        "        Dropout probability on the output projection.\n"
        "    residual_gate_init:\n"
        "        Initial value of the learnable residual gate. ``0.0`` makes the temporal\n"
        "        attention a no-op at initialization; a small positive value lets it\n"
        "        contribute immediately.",
    )
    ok &= replace_in_file(
        tgf_path,
        "    def __init__(\n"
        "        self,\n"
        "        d: int,\n"
        "        n_heads: int,\n"
        "        n_views: int,\n"
        "        temporal_window: int = 3,\n"
        "        dropout: float = 0.1,\n"
        "    ):",
        "    def __init__(\n"
        "        self,\n"
        "        d: int,\n"
        "        n_heads: int,\n"
        "        n_views: int,\n"
        "        temporal_window: int = 3,\n"
        "        dropout: float = 0.1,\n"
        "        residual_gate_init: float = 0.0,\n"
        "    ):",
    )
    ok &= replace_in_file(
        tgf_path,
        "        self.dropout = nn.Dropout(dropout)\n\n"
        "        # Identity at init: zero the output projection so the residual vanishes.\n"
        "        nn.init.zeros_(self.out_proj.weight)\n"
        "        if self.out_proj.bias is not None:\n"
        "            nn.init.zeros_(self.out_proj.bias)",
        "        self.dropout = nn.Dropout(dropout)\n\n"
        "        # Identity at init: zero the output projection so the residual vanishes.\n"
        "        nn.init.zeros_(self.out_proj.weight)\n"
        "        if self.out_proj.bias is not None:\n"
        "            nn.init.zeros_(self.out_proj.bias)\n\n"
        "        # Learnable residual gate. Starting at ``residual_gate_init`` lets the\n"
        "        # temporal path contribute gradually; ``0.0`` gives a strict warm-start\n"
        "        # from the per-frame v25 behaviour.\n"
        "        self.residual_gate = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float32))",
    )
    ok &= replace_in_file(
        tgf_path,
        "        out = self.out_proj(out)\n"
        "        out = out.view(B, J, T, V, d).permute(0, 2, 3, 1, 4)\n"
        "        return self.dropout(out)",
        "        out = self.out_proj(out)\n"
        "        out = out.view(B, J, T, V, d).permute(0, 2, 3, 1, 4)\n"
        "        # Gated residual: the temporal path is scaled by a learnable scalar so it\n"
        "        # can be warmed up from zero (or any chosen initial value).\n"
        "        return self.residual_gate * self.dropout(out)",
    )
    ok &= replace_in_file(
        tgf_path,
        "    temporal_loss_weight:\n"
        "        Weight of the temporal velocity-smoothness loss.\n"
        "    dropout:\n"
        "        Dropout rate.",
        "    temporal_loss_weight:\n"
        "        Weight of the temporal velocity-smoothness loss.\n"
        "    temporal_attention_residual_gate_init:\n"
        "        Initial value of the residual gate on each temporal attention layer.\n"
        "        ``0.0`` warm-starts from the per-frame v25 path.\n"
        "    dropout:\n"
        "        Dropout rate.",
    )
    ok &= replace_in_file(
        tgf_path,
        "        temporal_loss_weight: float = 0.1,\n"
        "        dropout: float = 0.1,\n"
        "        use_uncertainty_depth_proposals_v27: bool = False,",
        "        temporal_loss_weight: float = 0.1,\n"
        "        dropout: float = 0.1,\n"
        "        temporal_attention_residual_gate_init: float = 0.0,\n"
        "        use_uncertainty_depth_proposals_v27: bool = False,",
    )
    ok &= replace_in_file(
        tgf_path,
        "        self.temporal_loss_weight = temporal_loss_weight\n"
        "        self.use_uncertainty_depth_proposals_v27 = use_uncertainty_depth_proposals_v27",
        "        self.temporal_loss_weight = temporal_loss_weight\n"
        "        self.temporal_attention_residual_gate_init = temporal_attention_residual_gate_init\n"
        "        self.use_uncertainty_depth_proposals_v27 = use_uncertainty_depth_proposals_v27",
    )
    ok &= replace_in_file(
        tgf_path,
        "                    TemporalGeometryAttention(d, n_heads, n_views, temporal_window, dropout)\n"
        "                    for _ in range(n_temporal_layers)\n"
        "                ]",
        "                    TemporalGeometryAttention(\n"
        "                        d,\n"
        "                        n_heads,\n"
        "                        n_views,\n"
        "                        temporal_window,\n"
        "                        dropout,\n"
        "                        residual_gate_init=temporal_attention_residual_gate_init,\n"
        "                    )\n"
        "                    for _ in range(n_temporal_layers)\n"
        "                ]",
    )

    # omniview_fusion_v5.py
    ok &= replace_in_file(
        omni_path,
        "        use_temporal_geometry_fusion_v26: bool = False,\n"
        "        v26_temporal_window: int = 3,\n"
        "        use_uncertainty_depth_proposals_v27: bool = False,",
        "        use_temporal_geometry_fusion_v26: bool = False,\n"
        "        v26_temporal_window: int = 3,\n"
        "        v26_temporal_attention_residual_gate_init: float = 0.0,\n"
        "        use_uncertainty_depth_proposals_v27: bool = False,",
    )
    ok &= replace_in_file(
        omni_path,
        "            self.multiview_geometry_fusion_v25 = TemporalGeometryFusionV26(\n"
        "                d=self.d,\n"
        "                n_heads=self.n_heads,\n"
        "                n_views=n_views,\n"
        "                temporal_window=v26_temporal_window,\n"
        "                use_geometry_attention=v25_use_geometry_attention,\n"
        "                use_learned_depth_triangulation=v25_use_learned_depth_triangulation,\n"
        "                use_uncertainty_depth_proposals_v27=use_uncertainty_depth_proposals_v27,\n"
        "                v27_uncertainty_loss_weight=v27_uncertainty_loss_weight,\n"
        "                v27_udp_n_mixtures=v27_udp_n_mixtures,\n"
        "            )",
        "            self.multiview_geometry_fusion_v25 = TemporalGeometryFusionV26(\n"
        "                d=self.d,\n"
        "                n_heads=self.n_heads,\n"
        "                n_views=n_views,\n"
        "                temporal_window=v26_temporal_window,\n"
        "                temporal_attention_residual_gate_init=v26_temporal_attention_residual_gate_init,\n"
        "                use_geometry_attention=v25_use_geometry_attention,\n"
        "                use_learned_depth_triangulation=v25_use_learned_depth_triangulation,\n"
        "                use_uncertainty_depth_proposals_v27=use_uncertainty_depth_proposals_v27,\n"
        "                v27_uncertainty_loss_weight=v27_uncertainty_loss_weight,\n"
        "                v27_udp_n_mixtures=v27_udp_n_mixtures,\n"
        "            )",
    )

    # train script
    ok &= replace_in_file(
        train_path,
        "        \"use_temporal_geometry_fusion_v26\": getattr(args, \"use_temporal_geometry_fusion_v26\", False),\n"
        "        \"v26_temporal_window\": getattr(args, \"v26_temporal_window\", 3),\n"
        "        \"use_uncertainty_depth_proposals_v27\": getattr(args, \"use_uncertainty_depth_proposals_v27\", False),",
        "        \"use_temporal_geometry_fusion_v26\": getattr(args, \"use_temporal_geometry_fusion_v26\", False),\n"
        "        \"v26_temporal_window\": getattr(args, \"v26_temporal_window\", 3),\n"
        "        \"v26_temporal_attention_residual_gate_init\": getattr(args, \"v26_temporal_attention_residual_gate_init\", 0.0),\n"
        "        \"use_uncertainty_depth_proposals_v27\": getattr(args, \"use_uncertainty_depth_proposals_v27\", False),",
    )
    ok &= replace_in_file(
        train_path,
        "    parser.add_argument(\"--use_temporal_geometry_fusion_v26\", action=\"store_true\", default=False, help=\"Use v26 temporal geometry fusion instead of v25\")\n"
        "    parser.add_argument(\"--v26_temporal_window\", type=int, default=3, help=\"Temporal window size for v26 (must be odd)\")\n"
        "    parser.add_argument(\"--use_uncertainty_depth_proposals_v27\", action=\"store_true\", default=False, help=\"Use v27 uncertainty-aware depth-proposal triangulation head in v25/v26\")",
        "    parser.add_argument(\"--use_temporal_geometry_fusion_v26\", action=\"store_true\", default=False, help=\"Use v26 temporal geometry fusion instead of v25\")\n"
        "    parser.add_argument(\"--v26_temporal_window\", type=int, default=3, help=\"Temporal window size for v26 (must be odd)\")\n"
        "    parser.add_argument(\"--v26_temporal_attention_residual_gate_init\", type=float, default=0.0, help=\"Initial value of residual gate on v26 temporal attention (0.0 = warm-start from v25)\")\n"
        "    parser.add_argument(\"--use_uncertainty_depth_proposals_v27\", action=\"store_true\", default=False, help=\"Use v27 uncertainty-aware depth-proposal triangulation head in v25/v26\")",
    )

    # test file: write complete new version with the added tests.
    test_source = '''"""Unit tests for motionflow_mv/fusion/temporal_geometry_fusion_v26.py.

Covers the public contract of ``TemporalGeometryFusionV26``:
- forward shape ``(B, T, V, J, 3) -> (B, T, J, 3)``
- identity-at-init behaviour (falls back to v25)
- gradient flow through inputs
- view masking
- temporal boundary handling (T < temporal_window)
- toggle coverage
"""

import numpy as np
import pytest
import torch

from motionflow_mv.fusion.multiview_geometry_fusion_v25 import triangulate_initial
from motionflow_mv.fusion.temporal_geometry_fusion_v26 import (
    TemporalGeometryAttention,
    TemporalGeometryFusionV26,
)


def _make_cameras(n_views: int = 4) -> tuple:
    """Build a simple circular camera rig."""
    Ks, Rs, ts = [], [], []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
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
        Ks.append(K)
        Rs.append(R)
        ts.append(t)
    return (
        torch.from_numpy(np.stack(Ks)).float(),
        torch.from_numpy(np.stack(Rs)).float(),
        torch.from_numpy(np.stack(ts)).float(),
    )


def _project_points(
    joints_3d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Project world points into all views; returns (F, V, J, 2)."""
    t = t[:, None, None, :]
    X_cam = torch.einsum("vab,fjb->vfja", R, joints_3d) + t
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])
    points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
    return points_2d.permute(1, 0, 2, 3)


def _make_batch(
    B: int = 2,
    T: int = 5,
    V: int = 4,
    J: int = 17,
):
    """Return synthetic (points_2d, K, R, t, pred_3d_init, view_mask)."""
    K, R, t = _make_cameras(V)
    torch.manual_seed(42)
    joints_3d = torch.randn(T, J, 3) * 0.3
    points_2d = _project_points(joints_3d, K, R, t)

    K = K.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
    R = R.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
    t = t.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
    points_2d = points_2d.unsqueeze(0).expand(B, -1, -1, -1, -1)
    confidence = torch.ones(B, T, V, J)
    points_2d = torch.cat([points_2d, confidence[..., None]], dim=-1)

    pred_3d_init = joints_3d.unsqueeze(0).expand(B, -1, -1, -1)
    view_mask = torch.ones(B, T, V).bool()
    return points_2d, K, R, t, pred_3d_init, view_mask


# ---------------------------------------------------------------------------
# Core shape / forward tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("J", [17, 28])
def test_forward_shape(J):
    module = TemporalGeometryFusionV26(d=64, n_heads=2, n_views=4, temporal_window=3)
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch(B=2, T=5, V=4, J=J)
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert out.shape == (2, 5, J, 3)


def test_temporal_attention_forward_shape():
    B, T, V, J, d = 2, 5, 4, 17, 64
    tokens = torch.randn(B, T, V, J, d)
    epipolar_dist = torch.rand(B, T, V, V, J)
    ray_logit = torch.randn(B, T, V, V, J)
    attn = TemporalGeometryAttention(d=d, n_heads=2, n_views=V, temporal_window=3)
    out = attn(tokens, epipolar_dist, ray_logit)
    assert out.shape == (B, T, V, J, d)


# ---------------------------------------------------------------------------
# Identity at init
# ---------------------------------------------------------------------------

def test_identity_at_init():
    module = TemporalGeometryFusionV26(
        d=64,
        n_heads=2,
        n_views=4,
        use_geometry_attention=False,
        use_temporal_geometry_attention=False,
        use_learned_depth_triangulation=False,
    )
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert torch.allclose(out, pred_3d_init, atol=1e-5)


def test_identity_at_init_with_temporal_attention_enabled():
    """With residual gate initialised to 0, temporal attention is a no-op at init."""
    module = TemporalGeometryFusionV26(
        d=64,
        n_heads=2,
        n_views=4,
        use_geometry_attention=False,
        use_temporal_geometry_attention=True,
        use_learned_depth_triangulation=False,
    )
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert torch.allclose(out, pred_3d_init, atol=1e-5)
    # Verify the gate was initialised to zero.
    for layer in module.temporal_attn_layers:
        assert layer.residual_gate.item() == pytest.approx(0.0, abs=1e-6)


def test_identity_at_init_without_pred():
    module = TemporalGeometryFusionV26(
        d=64,
        n_heads=2,
        n_views=4,
        use_geometry_attention=False,
        use_temporal_geometry_attention=False,
        use_learned_depth_triangulation=False,
    )
    points_2d, K, R, t, _, _ = _make_batch()
    out, _ = module(points_2d, K, R, t)
    expected = triangulate_initial(points_2d[..., :2], K, R, t)
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    module = TemporalGeometryFusionV26(d=64, n_heads=2, n_views=4)
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()
    points_2d.requires_grad_(True)
    K.requires_grad_(True)
    R.requires_grad_(True)
    t.requires_grad_(True)
    pred_3d_init.requires_grad_(True)

    out, geom_loss = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    loss = out.mean() + geom_loss
    loss.backward()

    assert points_2d.grad is not None
    assert K.grad is not None
    assert R.grad is not None
    assert t.grad is not None
    assert pred_3d_init.grad is not None


def test_residual_gate_learnable():
    """The residual gate should be learnable and scale the temporal attention."""
    module = TemporalGeometryFusionV26(
        d=64,
        n_heads=2,
        n_views=4,
        use_geometry_attention=False,
        use_temporal_geometry_attention=True,
        use_learned_depth_triangulation=False,
    )
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()
    # At init the gate is zero, so the output equals the input estimate.
    out_zero, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert torch.allclose(out_zero, pred_3d_init, atol=1e-5)

    # Setting the gate to a non-zero value should change the output.
    for layer in module.temporal_attn_layers:
        nn.init.constant_(layer.residual_gate, 1.0)
    out_open, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert not torch.allclose(out_open, pred_3d_init, atol=1e-5)

    # The gate should receive gradients.
    loss = out_open.mean()
    loss.backward()
    for layer in module.temporal_attn_layers:
        assert layer.residual_gate.grad is not None


# ---------------------------------------------------------------------------
# View masking
# ---------------------------------------------------------------------------

def test_view_mask_ignores_dropped_view():
    module = TemporalGeometryFusionV26(d=64, n_heads=2, n_views=4)
    points_2d, K, R, t, _, _ = _make_batch()
    view_mask = torch.ones(2, 5, 4).bool()
    view_mask[:, :, -1] = False

    out, _ = module(points_2d, K, R, t, view_mask=view_mask)
    assert out.shape == (2, 5, 17, 3)
    assert out.isfinite().all()


# ---------------------------------------------------------------------------
# Temporal boundary handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [1, 2, 3, 7])
def test_temporal_window_larger_than_clip(T):
    """Clip length shorter than temporal window should not crash."""
    module = TemporalGeometryFusionV26(d=64, n_heads=2, n_views=4, temporal_window=5)
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch(B=2, T=T, V=4)
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert out.shape == (2, T, 17, 3)


# ---------------------------------------------------------------------------
# Toggle coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_geom_attn", [True, False])
@pytest.mark.parametrize("use_temporal", [True, False])
@pytest.mark.parametrize("use_learned_depth", [True, False])
def test_toggles_forward(use_geom_attn: bool, use_temporal: bool, use_learned_depth: bool):
    module = TemporalGeometryFusionV26(
        d=64,
        n_heads=2,
        n_views=4,
        use_geometry_attention=use_geom_attn,
        use_temporal_geometry_attention=use_temporal,
        use_learned_depth_triangulation=use_learned_depth,
    )
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert out.shape == (2, 5, 17, 3)


def test_invalid_head_dimension_raises():
    with pytest.raises(ValueError):
        TemporalGeometryAttention(d=64, n_heads=3, n_views=4)


def test_invalid_temporal_window_raises():
    with pytest.raises(ValueError):
        TemporalGeometryAttention(d=64, n_heads=4, n_views=4, temporal_window=4)


if __name__ == "__main__":
    test_forward_shape(17)
    test_forward_shape(28)
    test_temporal_attention_forward_shape()
    test_identity_at_init()
    test_identity_at_init_with_temporal_attention_enabled()
    test_identity_at_init_without_pred()
    test_gradient_flow()
    test_residual_gate_learnable()
    test_view_mask_ignores_dropped_view()
    test_temporal_window_larger_than_clip(1)
    test_temporal_window_larger_than_clip(2)
    test_temporal_window_larger_than_clip(3)
    test_temporal_window_larger_than_clip(7)
    for ga in [True, False]:
        for ut in [True, False]:
            for ud in [True, False]:
                test_toggles_forward(ga, ut, ud)
    test_invalid_head_dimension_raises()
    test_invalid_temporal_window_raises()
    print("All TemporalGeometryFusionV26 unit tests passed")
'''
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_source)
    print(f"[OK] Wrote {test_path}")

    if not ok:
        print("[FAIL] Some patches did not apply.", file=sys.stderr)
        sys.exit(1)

    # Stage and commit only the v26 files plus the new smoke script.
    subprocess.run(["git", "add", "-f", tgf_path, omni_path, train_path, test_path, "scripts/run_v26_temporal_geometry_fusion_smoke_warmstart.sh"], check=True)
    subprocess.run(
        ["git", "commit", "-m", "v26: add warm-startable residual gate to temporal geometry attention\n\n"
         "- Adds a learnable residual gate to TemporalGeometryAttention, initialised at 0.0\n"
         "  so v26 starts as the per-frame v25 path and gradually learns to use temporal context.\n"
         "- Wires the gate through TemporalGeometryFusionV26, OmniMultiViewFusionV5, and the\n"
         "  training script via --v26_temporal_attention_residual_gate_init.\n"
         "- Adds unit tests for identity-at-init with temporal attention enabled and for the\n"
         "  gate being learnable.\n"
         "- Includes a warm-start smoke script for local RTX 4090 validation."],
        check=True,
    )
    print("[OK] Committed v26 residual gate patch.")


if __name__ == "__main__":
    main()
