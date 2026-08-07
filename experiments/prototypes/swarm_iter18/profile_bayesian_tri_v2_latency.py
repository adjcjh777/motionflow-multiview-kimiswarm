"""Latency profiler for the current best model (BayesianTri v2).

Profiles end-to-end and per-component inference latency for
``RayAttentionFusionModelBayesianTriV2`` on CPU/GPU with synthetic inputs.

Usage (from repo root):
    KMP_DUPLICATE_LIB_OK=TRUE python experiments/prototypes/swarm_iter18/profile_bayesian_tri_v2_latency.py

Outputs:
    outputs/swarm_iter18/bayesian_tri_v2_profile_<timestamp>.json
    outputs/swarm_iter18/bayesian_tri_v2_profile_<timestamp>.md
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
    RayAttentionFusionModelBayesianTriV2,
    _adaptive_gauss_newton,
)
from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq


class _FixedRig:
    """Single fixed camera rig built with pure torch/math."""

    def __init__(self, V: int, device: torch.device):
        self.V = V
        self.K, self.R, self.t = self._build_rig(device)

    @staticmethod
    def _rotation_y(angle: float) -> torch.Tensor:
        c = math.cos(angle)
        s = math.sin(angle)
        return torch.tensor(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32
        )

    def _build_rig(self, device: torch.device):
        V = self.V
        Ks, Rs, ts = [], [], []
        radius = 5.0
        for i in range(V):
            fx = fy = 1000.0 + (i % 3) * 50.0
            cx, cy = 640.0, 360.0
            K = torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
            )
            Ks.append(K)

            angle = 2.0 * math.pi * i / V
            Rs.append(self._rotation_y(angle))

            t_vec = torch.tensor(
                [math.cos(angle) * radius, 0.0, math.sin(angle) * radius],
                dtype=torch.float32,
            )
            t_vec = t_vec / (t_vec.norm() + 1e-8) * radius
            ts.append(t_vec)

        K = torch.stack(Ks)
        R = torch.stack(Rs)
        t = torch.stack(ts)
        return K.to(device), R.to(device), t.to(device)

    def to(self, device: torch.device):
        return self.K.to(device), self.R.to(device), self.t.to(device)


class _ProfileTimer:
    def __init__(self):
        self.times = defaultdict(float)
        self._stack = []

    def __call__(self, name):
        return _ProfileTimerContext(self, name)


class _ProfileTimerContext:
    def __init__(self, timer: _ProfileTimer, name: str):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *args):
        self.timer.times[self.name] += time.perf_counter() - self.start


class InstrumentedBayesianTriV2(RayAttentionFusionModelBayesianTriV2):
    """BayesianTriV2 forward with per-section wall-clock timers.

    The forward implementation is a mirror of the parent class; only ``time.perf_counter``
    instrumentation is added. This keeps the shared module untouched while giving us a
    breakdown of where latency is spent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = _ProfileTimer()

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from motionflow_mv.fusion.ray_attention_temporal_crossview_model import (
                _cameras_to_tensors,
            )

            K, R, t = _cameras_to_tensors(cameras, device)

        if K.dim() == 3:
            K = K.unsqueeze(0).expand(B * T, -1, -1, -1)
            R = R.unsqueeze(0).expand(B * T, -1, -1, -1)
            t = t.unsqueeze(0).expand(B * T, -1, -1)
        elif K.dim() == 4:
            K = K.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            R = R.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            t = t.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, V, 3)
        else:
            raise ValueError("K must have shape (V, 3, 3) or (B, V, 3, 3)")

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        with self.timer("principal_point_correction"):
            correction_outputs = self.principal_point_correction(
                K=K,
                x=x_flat,
                weights=confidences,
            )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        with self.timer("extract_frame_features"):
            feat = self._extract_frame_features(x_flat, K_corrected, R, t)

        with self.timer("spatio_temporal_attention"):
            feat = feat.view(B, T, V, J, self.d)
            time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
            view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
            feat = feat + time_emb + view_emb

            feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
            for layer in self.st_transformer:
                feat = layer(feat)
            feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        with self.timer("covariance_head"):
            raw_cov = self.covariance_head(feat)
            L = self._cholesky_to_covariance(raw_cov)
            precision = 1.0 / (
                L[..., 0, 0].clamp(min=1e-4) * L[..., 1, 1].clamp(min=1e-4)
            )

        with self.timer("visibility_multiplier"):
            visibility = self._visibility_multiplier(feat, confidences)

        with self.timer("weight_head"):
            feat_for_weight = feat.permute(0, 2, 1, 3)
            w_logits = self.weight_head(feat_for_weight).squeeze(-1)
            weights = torch.sigmoid(w_logits).permute(0, 2, 1)
            weights = weights * confidences * precision * visibility
            weights = weights.clamp(min=1e-4)

        with self.timer("projection_matrix"):
            Rt = torch.cat([R, t[..., None]], dim=-1)
            P = K_corrected @ Rt

        with self.timer("triangulation"):
            pred_3d_raw = triangulate_dlt_batched_lstsq(points_2d, P, weights)

        with self.timer("damping_head"):
            feat_pooled = feat.mean(dim=1)
            damping = self.damping_head(feat_pooled).squeeze(-1)
            damping = self.min_gn_damping + (
                self.max_gn_damping - self.min_gn_damping
            ) * damping

        with self.timer("gauss_newton"):
            pred_3d_gn = _adaptive_gauss_newton(
                points_2d,
                weights,
                K_corrected,
                R,
                t,
                pred_3d_raw,
                damping,
                num_iters=self.gn_iters,
            )

        with self.timer("residual_mlp"):
            residual_input = torch.cat([feat_pooled, pred_3d_gn], dim=-1)
            delta = self.residual_mlp(residual_input)
            pred_3d = pred_3d_gn + delta

        with self.timer("epipolar_loss"):
            epi_loss = self._epipolar_consistency_loss(points_2d, K_corrected, R, t, L)
            epi_loss = self.epipolar_loss_weight * epi_loss

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        L = L.view(B, T, V, J, 2, 2)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            L = L.squeeze(1)

        out = (pred_3d, weights)

        if self.return_pp_delta:
            out += (pp_delta,)
            if self.correct_focal:
                out += (focal_scale,)

        if self.return_covariance:
            out += (L,)

        if self.return_raw:
            raw_3d = pred_3d_raw.view(B, T, J, 3)
            if squeeze_output:
                raw_3d = raw_3d.squeeze(1)
            out += (raw_3d,)

        out += (epi_loss,)
        return out


def _dummy_batch(batch_size: int, T: int, V: int, J: int, device: torch.device):
    x = torch.randn(batch_size, T, V, J, 3, device=device)
    x[..., 2] = torch.sigmoid(x[..., 2])
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--j", type=int, default=17)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--residual_hidden", type=int, default=256)
    parser.add_argument("--n_st_layers", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="outputs/swarm_iter18")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = InstrumentedBayesianTriV2(
        j=args.j,
        d=args.d,
        n_views=args.n_views,
        n_heads=4,
        n_joint_layers=1,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
    )
    model = model.to(device).eval()
    param_count = sum(p.numel() for p in model.parameters())

    rig = _FixedRig(args.n_views, torch.device("cpu"))
    K, R, t = rig.to(device)

    # Single-frame input (real-time streaming scenario).
    x_single = _dummy_batch(1, 1, args.n_views, args.j, device)
    # Clip input (temporal model native unit).
    x_clip = _dummy_batch(1, args.clip_len, args.n_views, args.j, device)

    def _bench(x, name):
        # Warm-up.
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(x, K=K, R=R, t=t)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # End-to-end timing.
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.iters):
                _ = model(x, K=K, R=R, t=t)
            if device.type == "cuda":
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Per-component timing on the last iteration (timers accumulate across iterations).
        model.timer.times.clear()
        with torch.no_grad():
            _ = model(x, K=K, R=R, t=t)
            if device.type == "cuda":
                torch.cuda.synchronize()
        per_component = dict(model.timer.times)

        total = sum(per_component.values())
        per_component_pct = {k: (v / total * 100.0) if total > 0 else 0.0 for k, v in per_component.items()}

        return {
            "name": name,
            "latency_ms": elapsed / args.iters * 1000.0,
            "fps": 1.0 / (elapsed / args.iters),
            "per_component_ms": {k: v * 1000.0 for k, v in per_component.items()},
            "per_component_pct": per_component_pct,
        }

    results = []
    print(f"Profiling BayesianTriV2 on {device} ...")
    for x, name in [(x_single, "single_frame"), (x_clip, "clip")]:
        result = _bench(x, name)
        results.append(result)
        print(f"  {name}: {result['latency_ms']:.2f} ms / {result['fps']:.1f} fps")

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "model": "RayAttentionFusionModelBayesianTriV2",
        "params": param_count,
        "settings": {
            "n_views": args.n_views,
            "joints": args.j,
            "clip_len": args.clip_len,
            "d": args.d,
            "residual_hidden": args.residual_hidden,
            "n_st_layers": args.n_st_layers,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "results": results,
    }

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"bayesian_tri_v2_profile_{ts}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# BayesianTri v2 Latency Profile",
        "",
        f"**Date**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC  ",
        f"**Device**: `{device}`  ",
        f"**Model**: `RayAttentionFusionModelBayesianTriV2`  ",
        f"**Parameters**: {param_count:,}  ",
        f"**Settings**: V={args.n_views}, J={args.j}, T={args.clip_len}, d={args.d}, "
        f"residual_hidden={args.residual_hidden}, n_st_layers={args.n_st_layers}  ",
        "",
        "## End-to-end latency",
        "",
        "| Scenario | Latency (ms) | FPS |",
        "|----------|-------------:|----:|",
    ]
    for r in results:
        md_lines.append(f"| {r['name']} | {r['latency_ms']:.2f} | {r['fps']:.1f} |")

    md_lines.extend([
        "",
        "## Per-component breakdown (last iteration)",
        "",
        "| Component | ms | % |",
        "|-----------|---:|---:|",
    ])
    # Use clip scenario for component breakdown as it is the more expensive one.
    clip_result = results[1]
    for k, ms in clip_result["per_component_ms"].items():
        md_lines.append(f"| {k} | {ms:.3f} | {clip_result['per_component_pct'][k]:.1f} |")

    md_lines.extend([
        "",
        "## Notes",
        "",
        "- Timings are wall-clock using ``time.perf_counter``.",
        "- The per-component breakdown is measured on a single inference pass, while",
        "  end-to-end latency is the average over many passes (includes amortized overhead).",
        "- CPU numbers here are smoke-test baselines; target deployment is CUDA/RTX 4090.",
    ])

    md_path = out_dir / f"bayesian_tri_v2_profile_{ts}.md"
    md_path.write_text("\n".join(md_lines))

    print(f"Saved JSON: {json_path}")
    print(f"Saved report: {md_path}")


if __name__ == "__main__":
    main()
