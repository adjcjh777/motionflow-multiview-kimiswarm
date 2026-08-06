"""Cross-dataset generalization evaluation for ray_attention_v3.

Evaluates a trained RayAttentionFusionModelV3 checkpoint on a source dataset and a
zero-shot target dataset, optionally under 2D noise / view dropout / outliers.
MPJPE is reported for both the learned model and a DLT baseline.

Example (fast 200-frame sanity check):
    python experiments/eval_cross_dataset_generalization.py \
        --source data/h36m_hf/s_01_act_02_multiview.npz \
        --target data/h36m_hf/s_09_acts_02_multiview.npz \
        --checkpoint outputs/ray_attention_v3_h36m.pth \
        --max_frames 200

If the checkpoint is missing, the script prints a warning and evaluates the
untrained model for forward-pass sanity.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_v3_model import RayAttentionFusionModelV3
from motionflow_mv.fusion.ray_attention_model import _triangulate_weighted_dlt


def dlt_baseline(points_2d, confidences, K, R, t):
    """Triangulate a batch using confidence-weighted DLT via torch.linalg.lstsq.

    Uses the same differentiable weighted DLT implementation as the model so the
    baseline is numerically consistent and avoids a separate numpy SVD path.

    Args:
        points_2d: (T, V, J, 2)
        confidences: (T, V, J)
        K, R: (V, 3, 3)
        t: (V, 3)

    Returns:
        X: (T, J, 3)
    """
    T, V, J, _ = points_2d.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K_t = torch.from_numpy(K).float().to(device)
    R_t = torch.from_numpy(R).float().to(device)
    t_t = torch.from_numpy(t).float().to(device)
    Rt = torch.cat([R_t, t_t[:, :, None]], dim=-1)  # (V, 3, 4)
    P = K_t @ Rt  # (V, 3, 4)
    P = P.unsqueeze(0).expand(T, -1, -1, -1)
    p2d = torch.from_numpy(points_2d).float().to(device)
    # Avoid zero weights because torch.linalg.lstsq is used in homogeneous form.
    w = torch.from_numpy(confidences).float().to(device).clamp(min=1e-6)
    with torch.no_grad():
        X = _triangulate_weighted_dlt(p2d, w, P)
    return X.cpu().numpy()


def corrupt_data(points_2d, confidences, noise_std, dropout_rate, outlier_rate, outlier_scale):
    p2d = points_2d.copy().astype(np.float64)
    conf = confidences.copy().astype(np.float64)
    T, V, J, _ = p2d.shape
    if noise_std > 0:
        p2d += np.random.randn(T, V, J, 2) * noise_std
    if dropout_rate > 0:
        mask = np.random.rand(T, V, J) > dropout_rate
        conf = conf * mask
    if outlier_rate > 0:
        out_mask = np.random.rand(T, V, J) < outlier_rate
        outlier = (np.random.rand(T, V, J, 2) - 0.5) * 2 * outlier_scale
        p2d = np.where(out_mask[..., None], outlier, p2d)
    return p2d, conf


def evaluate_dataset(
    model,
    data,
    device,
    batch_size,
    noise_std,
    dropout_rate,
    outlier_rate,
    outlier_scale,
    domain_id=None,
):
    """Return MPJPE (mm) for the learned model and DLT baseline."""
    points_2d = data["points_2d"]
    confidences = data["confidences"]
    joints_3d = data["joints_3d"]
    K_np = data["camera_K"]
    R_np = data["camera_R"]
    t_np = data["camera_t"]

    p2d, conf = corrupt_data(points_2d, confidences, noise_std, dropout_rate, outlier_rate, outlier_scale)

    # DLT baseline.
    dlt_X = dlt_baseline(p2d, conf, K_np, R_np, t_np)
    dlt_mpjpe = np.linalg.norm(dlt_X - joints_3d, axis=-1).mean()

    # Model inference.
    x = torch.from_numpy(np.concatenate([p2d, conf[..., None]], axis=-1)).float()
    K = torch.from_numpy(K_np).float().unsqueeze(0).to(device)
    R = torch.from_numpy(R_np).float().unsqueeze(0).to(device)
    t = torch.from_numpy(t_np).float().unsqueeze(0).to(device)

    dataset = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    domain_logits_list = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            kwargs = {}
            if domain_id is not None:
                kwargs["domain_labels"] = torch.full((xb.size(0),), domain_id, dtype=torch.long, device=device)
            outputs = model(xb, K=K.expand(xb.size(0), -1, -1, -1), R=R.expand(xb.size(0), -1, -1, -1), t=t.expand(xb.size(0), -1, -1), **kwargs)
            pred = outputs[0]
            preds.append(pred.cpu().numpy())
            if len(outputs) > 2:
                domain_logits_list.append(outputs[2].cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    model_mpjpe = np.linalg.norm(preds - joints_3d, axis=-1).mean()

    result = {"model_mpjpe_mm": float(model_mpjpe), "dlt_mpjpe_mm": float(dlt_mpjpe)}
    if domain_logits_list:
        logits = np.concatenate(domain_logits_list, axis=0)
        result["mean_domain_logit"] = [float(x) for x in logits.mean(axis=0)]
    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset generalization for ray_attention_v3.")
    parser.add_argument("--source", type=str, default="data/h36m_hf/s_01_act_02_multiview.npz")
    parser.add_argument("--target", type=str, default="data/h36m_hf/s_09_acts_02_multiview.npz")
    parser.add_argument("--checkpoint", type=str, default="outputs/ray_attention_v3_h36m.pth")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_frames", type=int, default=0, help="If >0, limit each dataset to this many frames for a quick sanity run.")
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 2.0, 5.0])
    parser.add_argument("--dropout_rates", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--outlier_rate", type=float, default=0.0)
    parser.add_argument("--outlier_scale", type=float, default=50.0)
    parser.add_argument("--use_domain_classifier", action="store_true", help="Instantiate model with gradient-reversal domain classifier.")
    parser.add_argument("--n_domains", type=int, default=2)
    parser.add_argument("--domain_id_source", type=int, default=0)
    parser.add_argument("--domain_id_target", type=int, default=1)
    parser.add_argument("--out", type=str, default="outputs/cross_dataset_generalization.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_data = np.load(args.source)
    target_data = np.load(args.target)

    def _maybe_crop(data):
        if args.max_frames > 0 and data["points_2d"].shape[0] > args.max_frames:
            return {
                "points_2d": data["points_2d"][:args.max_frames],
                "confidences": data["confidences"][:args.max_frames],
                "joints_3d": data["joints_3d"][:args.max_frames],
                "camera_K": data["camera_K"],
                "camera_R": data["camera_R"],
                "camera_t": data["camera_t"],
            }
        return {
            "points_2d": data["points_2d"],
            "confidences": data["confidences"],
            "joints_3d": data["joints_3d"],
            "camera_K": data["camera_K"],
            "camera_R": data["camera_R"],
            "camera_t": data["camera_t"],
        }

    source_data = _maybe_crop(source_data)
    target_data = _maybe_crop(target_data)

    n_views = source_data["camera_K"].shape[0]
    n_joints = source_data["points_2d"].shape[2]

    model = RayAttentionFusionModelV3(
        j=n_joints,
        d=args.d,
        n_views=n_views,
        use_domain_classifier=args.use_domain_classifier,
        n_domains=args.n_domains,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=False)
        print(f"Loaded checkpoint {ckpt_path}")
    else:
        print(f"Warning: checkpoint {ckpt_path} not found, using random weights")

    results = {
        "source_path": args.source,
        "target_path": args.target,
        "checkpoint": args.checkpoint,
        "use_domain_classifier": args.use_domain_classifier,
        "conditions": [],
    }

    print(f"{'drop':>5} {'noise':>5} {'src_model':>10} {'src_dlt':>10} {'tgt_model':>10} {'tgt_dlt':>10}")
    print("-" * 65)

    for drop in args.dropout_rates:
        for noise in args.noise_levels:
            src_res = evaluate_dataset(
                model, source_data, device, args.batch_size,
                noise_std=noise, dropout_rate=drop,
                outlier_rate=args.outlier_rate, outlier_scale=args.outlier_scale,
                domain_id=args.domain_id_source if args.use_domain_classifier else None,
            )
            tgt_res = evaluate_dataset(
                model, target_data, device, args.batch_size,
                noise_std=noise, dropout_rate=drop,
                outlier_rate=args.outlier_rate, outlier_scale=args.outlier_scale,
                domain_id=args.domain_id_target if args.use_domain_classifier else None,
            )
            record = {
                "dropout": drop,
                "noise": noise,
                "source": src_res,
                "target": tgt_res,
            }
            results["conditions"].append(record)
            print(
                f"{drop:>5.1f} {noise:>5.2f} "
                f"{src_res['model_mpjpe_mm']:>10.2f} {src_res['dlt_mpjpe_mm']:>10.2f} "
                f"{tgt_res['model_mpjpe_mm']:>10.2f} {tgt_res['dlt_mpjpe_mm']:>10.2f}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
