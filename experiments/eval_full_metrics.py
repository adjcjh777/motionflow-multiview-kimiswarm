"""Generic full-metrics evaluation for the temporal-residual model family.

Supports baseline (residual), CamPE, CamPE+GraphJR, and CamPE+Adaptive variants.
Outputs MPJPE, PA-MPJPE, PCK@50/100/150, and PCK-AUC in mm.

Example
-------
    conda run -n mf python experiments/eval_full_metrics.py \
        --model campegraph \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --checkpoint outputs/ray_attention_temporal_residual_campe_graph_mpiinf3dhp.pth \
        --clip_len 13 --d 64 --n_temporal_layers 2 --graph_layers 3 --residual_hidden 128
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics, summarize_metrics
from motionflow_mv.fusion.variable_view_inference import (
    prepare_variable_view_input,
    VariableViewInferenceWrapper,
)
from motionflow_mv.fusion.ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
)
from motionflow_mv.fusion.ray_attention_temporal_residual_campe_model import (
    RayAttentionFusionModelTemporalResidualCamPE,
)
from motionflow_mv.fusion.ray_attention_temporal_residual_campe_graph_model import (
    RayAttentionFusionModelTemporalResidualCamPEGraph,
)
from motionflow_mv.fusion.ray_attention_temporal_residual_campe_adaptive_model import (
    RayAttentionFusionModelTemporalResidualCamPEAdaptive,
)
from motionflow_mv.fusion.ray_attention_temporal_residual_campe_adaptive_softgate_model import (
    RayAttentionFusionModelTemporalResidualCamPEAdaptiveSoftGate,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_model import (
    RayAttentionFusionModelTemporalCrossviewResidual,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_visibility_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_factorized_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_dynamic_gate_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointDynamicGate,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_graph_skeleton_residual_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraphSkeletonResidual,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_epipolar_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolar,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_splat_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSplat,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_kinematic_chain_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointKinematicChain,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_epipolar_bias_v2_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2,
)
from motionflow_mv.models.crossview_residual_visibility_v2 import (
    CrossviewResidualVisibilityV2,
)


MODEL_CLASSES = {
    "residual": RayAttentionFusionModelTemporalResidual,
    "campe": RayAttentionFusionModelTemporalResidualCamPE,
    "campegraph": RayAttentionFusionModelTemporalResidualCamPEGraph,
    "adaptive": RayAttentionFusionModelTemporalResidualCamPEAdaptive,
    "adaptive_softgate": RayAttentionFusionModelTemporalResidualCamPEAdaptiveSoftGate,
    "crossview_residual": RayAttentionFusionModelTemporalCrossviewResidual,
    "crossview_residual_pp": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
    "crossview_residual_pp_visibility": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility,
    "crossview_residual_pp_visibility_v2": CrossviewResidualVisibilityV2,
    "factorized_pp": RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint,
    "dynamic_gate_pp": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointDynamicGate,
    "graph_skeleton_residual_pp": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraphSkeletonResidual,
    "epipolar_pp": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolar,
    "splat_pp": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSplat,
    "kinematic_chain_pp": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointKinematicChain,
    "bayesian_tri_pp": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri,
    "epipolar_bias_v2_pp": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2,
}


class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield clips (T, V, J, 3) from a canonical .npz sequence."""

    def __init__(self, npz_path: str, clip_len: int, stride: int = 1, gt_scale: float = 1.0, camera_scale: float = 1.0):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float() * gt_scale
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float() * camera_scale
        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def build_model(args, n_views, j):
    cls = MODEL_CLASSES[args.model]
    kwargs = {
        "j": j,
        "d": args.d,
        "n_views": n_views,
    }
    if args.model in {"crossview_residual", "crossview_residual_pp", "crossview_residual_pp_visibility", "dynamic_gate_pp", "graph_skeleton_residual_pp", "epipolar_pp", "splat_pp", "kinematic_chain_pp", "bayesian_tri_pp"}:
        kwargs["n_st_layers"] = args.n_st_layers
        kwargs["residual_hidden"] = args.residual_hidden
    elif args.model == "factorized_pp":
        kwargs["n_view_layers"] = args.n_view_layers
        kwargs["n_temporal_layers"] = args.n_temporal_layers
        kwargs["residual_hidden"] = args.residual_hidden
        kwargs["principal_point_hidden"] = 64
        kwargs["principal_point_max_offset"] = 20.0
    elif args.model != "crossview_residual_pp_visibility_v2":
        kwargs["n_temporal_layers"] = args.n_temporal_layers
    if args.model in {"residual", "campe", "campegraph", "adaptive"}:
        kwargs["residual_hidden"] = args.residual_hidden
    if args.model == "campegraph":
        kwargs["graph_layers"] = args.graph_layers
        # MPI full-skeleton graph is passed via --parents and --symmetry_pairs files.
        if args.parents:
            kwargs["parents"] = _load_list(args.parents, int)
        if args.symmetry_pairs:
            kwargs["symmetry_pairs"] = _load_pairs(args.symmetry_pairs)
    if args.model == "adaptive":
        kwargs["k"] = args.k
    if args.model == "adaptive_softgate":
        kwargs["target_k"] = args.target_k
        kwargs["min_views"] = args.min_views
        kwargs["lambda_gate"] = 0.0
    if args.model == "crossview_residual_pp_visibility_v2":
        kwargs["principal_point_hidden"] = 64
        kwargs["principal_point_max_offset"] = 20.0
    return cls(**kwargs)


def _load_list(path, dtype):
    with open(path) as f:
        return [dtype(x.strip()) for x in f.read().strip().split(",") if x.strip()]


def _load_pairs(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split(",")
            out.append((int(a.strip()), int(b.strip())))
    return out


def evaluate(model, loader, device, model_name: str, source_n_views: int = None):
    model.eval()
    use_wrapper = source_n_views is not None and source_n_views != loader.dataset.K.shape[0]
    if use_wrapper:
        model = VariableViewInferenceWrapper(model)
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            if use_wrapper:
                # Use all available target views (assumed to be <= source_n_views).
                active = xb.shape[2]
                pred = model(xb, K=K, R=R, t=t, active_views=active)[0]
            else:
                out = model(xb, K=K, R=R, t=t)
                pred = out[0]
                if model_name == "adaptive_softgate":
                    pred = out[0]
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)  # (N, T, J, 3)
    gts = np.concatenate(gts, axis=0)
    preds = preds.reshape(-1, preds.shape[-2], 3)  # (N*T, J, 3)
    gts = gts.reshape(-1, gts.shape[-2], 3)
    return preds, gts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(MODEL_CLASSES), required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--n_st_layers", type=int, default=2,
                        help="Number of cross-view spatio-temporal layers for crossview_residual/crossview_residual_pp models")
    parser.add_argument("--n_view_layers", type=int, default=2, help="Number of view-level transformer layers for factorized_pp")
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--target_k", type=int, default=4)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=1, help="Stride for validation clips (higher = faster)")
    parser.add_argument("--gt_scale", type=float, default=1.0)
    parser.add_argument("--camera_scale", type=float, default=1.0)
    parser.add_argument("--parents", type=str, default=None, help="Path to comma-separated parent list for campegraph")
    parser.add_argument("--symmetry_pairs", type=str, default=None, help="Path to symmetry pairs for campegraph")
    parser.add_argument("--source_n_views", type=int, default=None, help="Fixed view count of the trained model; enables variable-view inference when target has fewer views")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    dataset = TemporalClipDataset(args.dataset, args.clip_len, stride=args.val_stride, gt_scale=args.gt_scale, camera_scale=args.camera_scale)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    source_n_views = args.source_n_views if args.source_n_views is not None else n_views
    model = build_model(args, source_n_views, j).to(device)
    missing, unexpected = model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True),
        strict=False,
    )
    if missing:
        print(f"Warning: missing keys in checkpoint: {missing[:5]}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint (ignored): {unexpected[:5]}")

    preds, gts = evaluate(model, loader, device, args.model, source_n_views=source_n_views)
    parents_arr = None
    if args.parents:
        parents_arr = np.array(_load_list(args.parents, int), dtype=np.int64)
    # Convert meters -> mm for metrics.
    report = compute_all_metrics(preds * 1000.0, gts * 1000.0, parents=parents_arr)
    print(summarize_metrics(report))
    print(f"MPJPE: {report['mpjpe']:.2f} mm")
    print(f"PA-MPJPE: {report['pa_mpjpe']:.2f} mm")
    print(f"PCK@50: {report['pck@50mm']:.4f}")
    print(f"PCK@100: {report['pck@100mm']:.4f}")
    print(f"PCK@150: {report['pck@150mm']:.4f}")
    print(f"PCK-AUC: {report['pck_auc']:.4f}")

    if args.output_json:
        # Convert numpy arrays to lists for JSON serialization.
        serializable = {}
        for k, v in report.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = float(v)
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
