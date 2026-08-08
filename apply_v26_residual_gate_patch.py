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

    # test file
    ok &= replace_in_file(
        test_path,
        "def test_identity_at_init():\n"
        "    module = TemporalGeometryFusionV26(\n"
        "        d=64,\n"
        "        n_heads=2,\n"
        "        n_views=4,\n"
        "        use_geometry_attention=False,\n"
        "        use_temporal_geometry_attention=False,\n"
        "        use_learned_depth_triangulation=False,\n"
        "    )\n"
        "    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()\n"
        "    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)\n"
        "    assert torch.allclose(out, pred_3d_init, atol=1e-5)\n"
        "\n"
        "\ndef test_identity_at_init_without_pred():",
        "def test_identity_at_init():\n"
        "    module = TemporalGeometryFusionV26(\n"
        "        d=64,\n"
        "        n_heads=2,\n"
        "        n_views=4,\n"
        "        use_geometry_attention=False,\n"
        "        use_temporal_geometry_attention=False,\n"
        "        use_learned_depth_triangulation=False,\n"
        "    )\n"
        "    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()\n"
        "    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)\n"
        "    assert torch.allclose(out, pred_3d_init, atol=1e-5)\n"
        "\n"
        "\ndef test_identity_at_init_with_temporal_attention_enabled():\n"
        "    \"\"\"With residual gate initialised to 0, temporal attention is a no-op at init.\"\"\"\n"
        "    module = TemporalGeometryFusionV26(\n"
        "        d=64,\n"
        "        n_heads=2,\n"
        "        n_views=4,\n"
        "        use_geometry_attention=False,\n"
        "        use_temporal_geometry_attention=True,\n"
        "        use_learned_depth_triangulation=False,\n"
        "    )\n"
        "    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()\n"
        "    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)\n"
        "    assert torch.allclose(out, pred_3d_init, atol=1e-5)\n"
        "    # Verify the gate was initialised to zero.\n"
        "    for layer in module.temporal_attn_layers:\n"
        "        assert layer.residual_gate.item() == pytest.approx(0.0, abs=1e-6)\n"
        "\n"
        "\ndef test_identity_at_init_without_pred():",
    )
    ok &= replace_in_file(
        test_path,
        "    assert K.grad is not None\n"
        "    assert R.grad is not None\n"
        "    assert t.grad is not None\n"
        "    assert pred_3d_init.grad is not None\n"
        "\n"
        "# ---------------------------------------------------------------------------\n"
        "# View masking\n"
        "# ---------------------------------------------------------------------------",
        "    assert K.grad is not None\n"
        "    assert R.grad is not None\n"
        "    assert t.grad is not None\n"
        "    assert pred_3d_init.grad is not None\n"
        "\n"
        "\ndef test_residual_gate_learnable():\n"
        "    \"\"\"The residual gate should be learnable and scale the temporal attention.\"\"\"\n"
        "    module = TemporalGeometryFusionV26(\n"
        "        d=64,\n"
        "        n_heads=2,\n"
        "        n_views=4,\n"
        "        use_geometry_attention=False,\n"
        "        use_temporal_geometry_attention=True,\n"
        "        use_learned_depth_triangulation=False,\n"
        "    )\n"
        "    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()\n"
        "    # At init the gate is zero, so the output equals the input estimate.\n"
        "    out_zero, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)\n"
        "    assert torch.allclose(out_zero, pred_3d_init, atol=1e-5)\n"
        "\n"
        "    # Setting the gate to a non-zero value should change the output.\n"
        "    for layer in module.temporal_attn_layers:\n"
        "        nn.init.constant_(layer.residual_gate, 1.0)\n"
        "    out_open, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)\n"
        "    assert not torch.allclose(out_open, pred_3d_init, atol=1e-5)\n"
        "\n"
        "    # The gate should receive gradients.\n"
        "    loss = out_open.mean()\n"
        "    loss.backward()\n"
        "    for layer in module.temporal_attn_layers:\n"
        "        assert layer.residual_gate.grad is not None\n"
        "\n"
        "# ---------------------------------------------------------------------------\n"
        "# View masking\n"
        "# ---------------------------------------------------------------------------",
    )
    ok &= replace_in_file(
        test_path,
        "if __name__ == \"__main__\":\n"
        "    test_forward_shape(17)\n"
        "    test_forward_shape(28)\n"
        "    test_temporal_attention_forward_shape()\n"
        "    test_identity_at_init()\n"
        "    test_identity_at_init_without_pred()\n"
        "    test_gradient_flow()\n"
        "    test_view_mask_ignores_dropped_view()\n"
        "    test_temporal_window_larger_than_clip(1)\n"
        "    test_temporal_window_larger_than_clip(2)\n"
        "    test_temporal_window_larger_than_clip(3)\n"
        "    test_temporal_window_larger_than_clip(7)\n"
        "    for ga in [True, False]:\n"
        "        for ut in [True, False]:\n"
        "            for ud in [True, False]:\n"
        "                test_toggles_forward(ga, ut, ud)\n"
        "    test_invalid_head_dimension_raises()\n"
        "    test_invalid_temporal_window_raises()\n"
        "    print(\"All TemporalGeometryFusionV26 unit tests passed\")",
        "if __name__ == \"__main__\":\n"
        "    test_forward_shape(17)\n"
        "    test_forward_shape(28)\n"
        "    test_temporal_attention_forward_shape()\n"
        "    test_identity_at_init()\n"
        "    test_identity_at_init_with_temporal_attention_enabled()\n"
        "    test_identity_at_init_without_pred()\n"
        "    test_gradient_flow()\n"
        "    test_residual_gate_learnable()\n"
        "    test_view_mask_ignores_dropped_view()\n"
        "    test_temporal_window_larger_than_clip(1)\n"
        "    test_temporal_window_larger_than_clip(2)\n"
        "    test_temporal_window_larger_than_clip(3)\n"
        "    test_temporal_window_larger_than_clip(7)\n"
        "    for ga in [True, False]:\n"
        "        for ut in [True, False]:\n"
        "            for ud in [True, False]:\n"
        "                test_toggles_forward(ga, ut, ud)\n"
        "    test_invalid_head_dimension_raises()\n"
        "    test_invalid_temporal_window_raises()\n"
        "    print(\"All TemporalGeometryFusionV26 unit tests passed\")",
    )

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
