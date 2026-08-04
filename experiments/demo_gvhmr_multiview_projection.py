"""Synthetic multi-view projection demo from a single-view GVHMR output.

Loads a GVHMR hmr4d_results.pt, generates N virtual calibrated cameras,
projects the SMPL 3D joints into each view, adds a little noise, and runs the
FusionModule plugins to recover the 3D pose.  Reports MPJPE w.r.t. the GVHMR
world joints.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/demo_gvhmr_multiview_projection.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import smplx
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.eval.metrics import mpjpe
from motionflow_mv.fusion import FUSION_REGISTRY
from motionflow_mv.ir.gvhmr_adapter import gvhmr_pt_to_ir


def make_cameras_on_circle(n_views: int = 4, radius: float = 4.0):
    """Return a list of n calibrated cameras looking at the origin."""
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        phi = np.pi / 3
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 900.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0

        c = radius * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def project_points(points_3d: np.ndarray, camera: Camera):
    """Project (J, 3) to (J, 2) using camera projection matrix."""
    P = camera.projection_matrix
    X_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return x


def main():
    parser = argparse.ArgumentParser(description="GVHMR multi-view projection demo.")
    parser.add_argument("--input", type=str, default="data/gvhmr_demo/hmr4d_results.pt")
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--shelf_checkpoints", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load GVHMR result and SMPL model.
    ir = gvhmr_pt_to_ir(args.input)
    smpl_model = smplx.SMPL("data/smpl/SMPL_NEUTRAL.pkl", batch_size=1)

    T = len(ir.timestamps)
    print(f"Loaded {T} frames from {args.input}")

    # Use mean betas over time to keep a single shape.
    betas = torch.from_numpy(ir.pose["betas"]).mean(dim=0, keepdim=True).float()
    body_pose = torch.from_numpy(ir.pose["body_pose"]).float()
    global_orient = torch.from_numpy(ir.pose["global_orient"]).float()
    transl = torch.from_numpy(ir.pose["transl"]).float()

    # Generate virtual cameras.
    cameras = make_cameras_on_circle(args.n_views)

    # Load plugin checkpoints.
    if args.shelf_checkpoints:
        checkpoints = {
            "attention": "outputs/attention_fusion_shelf.pth",
            "robust_triangulation": "outputs/robust_triangulation_shelf.pth",
            "residual_refiner": "outputs/residual_refiner_shelf.pth",
            "temporal_refiner": "outputs/temporal_refiner_synthetic.pth",
        }
        kwargs = {
            "attention": {"d": 64, "n_views": args.n_views},
        }
        for name, path in checkpoints.items():
            if not Path(path).exists():
                continue
            if name == "attention":
                from motionflow_mv.fusion.attention_fusion_module import AttentionFusionModule
                module = AttentionFusionModule(j=17, **kwargs[name])
                FUSION_REGISTRY._modules[name] = module
            module = FUSION_REGISTRY.get(name)
            state = torch.load(path, map_location="cpu", weights_only=True)
            module.model.load_state_dict(state)
            module.model.to(device)
            module.model.eval()
            print(f"Loaded {path} for {name}")

    per_plugin_errors = {name: [] for name in FUSION_REGISTRY.names()}
    gt_joints_all = []

    rng = np.random.default_rng(2025)
    for t in range(T):
        # SMPL forward to get 3D joints (world coordinates, meters).
        # Pad body_pose to smplx expected length (69) if it comes from GVHMR (63).
        bp = body_pose[t:t + 1]
        if bp.shape[1] < 69:
            bp_padded = torch.zeros(1, 69)
            bp_padded[:, :bp.shape[1]] = bp
        else:
            bp_padded = bp
        with torch.no_grad():
            output = smpl_model(
                betas=betas,
                body_pose=bp_padded,
                global_orient=global_orient[t:t + 1],
                transl=transl[t:t + 1],
            )
        joints_3d = output.joints[0, :17].cpu().numpy()  # (17, 3)
        gt_joints_all.append(joints_3d)

        # Project to each view and add noise.
        points_2d_list = []
        confidences_list = []
        for cam in cameras:
            x = project_points(joints_3d, cam)
            x += rng.normal(0, args.noise_std, size=x.shape)
            points_2d_list.append(x)
            confidences_list.append(rng.uniform(0.8, 1.0, size=joints_3d.shape[0]))
        points_2d = np.stack(points_2d_list, axis=0)  # (V, J, 2)
        confidences = np.stack(confidences_list, axis=0)  # (V, J)

        points_2d_px = points_2d[None]
        points_2d_norm = (points_2d / 1000.0)[None]
        confidences_batch = confidences[None]

        for name in sorted(FUSION_REGISTRY.names()):
            module = FUSION_REGISTRY.get(name)
            try:
                if name in ("attention", "attention_v2"):
                    input_2d = points_2d_norm
                    output_scale = 1000.0
                elif name == "robust_triangulation":
                    input_2d = points_2d_px
                    output_scale = 1.0
                else:
                    input_2d = points_2d_px
                    output_scale = 1.0
                pred_3d = module.fuse(input_2d, confidences_batch, cameras)
                if pred_3d.ndim == 3:
                    pred_3d = pred_3d[0]
                pred_3d_mm = pred_3d * output_scale
                # joints_3d is in meters; convert mm -> m.
                if name in ("attention", "attention_v2"):
                    pred_3d_m = pred_3d
                else:
                    pred_3d_m = pred_3d_mm / 1000.0
                err = mpjpe(pred_3d_m[None], joints_3d[None])
                per_plugin_errors[name].append(err)
            except Exception as e:
                print(f"Plugin {name} failed on frame {t}: {e}")

    print("\nMPJPE (m) vs GVHMR world joints")
    print(f"{'Plugin':<20} {'MPJPE':>10}")
    print("-" * 32)
    for name in sorted(per_plugin_errors.keys()):
        if per_plugin_errors[name]:
            errors = np.array(per_plugin_errors[name])
            print(f"{name:<20} {errors.mean():>10.4f}")
        else:
            print(f"{name:<20} {'N/A':>10}")


if __name__ == "__main__":
    main()
