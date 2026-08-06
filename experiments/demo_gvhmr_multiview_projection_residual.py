"""GVHMR multi-view projection demo with the temporal-residual ray-attention plugin.

Loads a GVHMR ``hmr4d_results.pt``, runs SMPL forward to obtain single-view
world 3D joints, and treats these as the per-view SMPL joints for a set of
virtual calibrated cameras.  Each per-view joint set is projected into its
camera, perturbed by noise, and fed into the temporal-residual fusion plugin.

The fused multi-view output is compared against the original single-view GVHMR
world reference.

Usage:
    conda run -n mf python experiments/demo_gvhmr_multiview_projection_residual.py
    conda run -n mf python experiments/demo_gvhmr_multiview_projection_residual.py --max_frames 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import smplx
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.ray_attention_temporal_residual_module import (
    RayAttentionTemporalResidualFusionModule,
)
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


def camera_projection_tensor(camera: Camera, device: torch.device):
    """Return the 3x4 projection matrix as a torch tensor."""
    Rt = torch.cat(
        [torch.from_numpy(camera.R).float(), torch.from_numpy(camera.t.reshape(3, 1)).float()],
        dim=1,
    )
    K = torch.from_numpy(camera.K).float()
    P = K @ Rt
    return P.to(device)


def project_points(points_3d: np.ndarray, P: torch.Tensor):
    """Project (J, 3) to (J, 2) using a torch projection matrix."""
    X = torch.from_numpy(points_3d).float().to(P.device)
    J = X.shape[0]
    ones = torch.ones(J, 1, device=P.device)
    X_h = torch.cat([X, ones], dim=1)  # (J, 4)
    x_h = X_h @ P.T  # (J, 3)
    x = x_h[:, :2] / x_h[:, 2:3]
    return x.cpu().numpy()


def mpjpe_m(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint position error in meters."""
    pred_t = torch.from_numpy(pred).float()
    gt_t = torch.from_numpy(gt).float()
    return float(torch.mean(torch.norm(pred_t - gt_t, dim=-1)))


def cameras_to_tensors(cameras, device):
    """Convert a list of Camera objects to (K, R, t) tensors on device."""
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    return K, R, t


def main():
    parser = argparse.ArgumentParser(description="GVHMR multi-view projection demo with temporal-residual plugin.")
    parser.add_argument("--input", type=str, default="data/gvhmr_demo/hmr4d_results.pt")
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--outlier_rate", type=float, default=0.0)
    parser.add_argument("--outlier_scale", type=float, default=100.0)
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/ray_attention_temporal_residual_h36m.pth")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ir = gvhmr_pt_to_ir(args.input)
    smpl_model = smplx.SMPL("data/smpl/SMPL_NEUTRAL.pkl", batch_size=1)

    T = len(ir.timestamps)
    if args.max_frames is not None:
        T = min(T, args.max_frames)
    print(f"Loaded {T} frames from {args.input}")

    betas = torch.from_numpy(ir.pose["betas"]).mean(dim=0, keepdim=True).float()
    body_pose = torch.from_numpy(ir.pose["body_pose"]).float()
    global_orient = torch.from_numpy(ir.pose["global_orient"]).float()
    transl = torch.from_numpy(ir.pose["transl"]).float()
    # GVHMR may emit 21 body-joint pose (63 dims). Pad to 23 body joints (69 dims)
    # so the standard SMPL model can consume it.
    if body_pose.shape[-1] == 63:
        body_pose = torch.cat([body_pose, torch.zeros(body_pose.shape[0], 6)], dim=-1)

    cameras = make_cameras_on_circle(args.n_views)
    K, R, t = cameras_to_tensors(cameras, device)
    P_list = [camera_projection_tensor(cam, device) for cam in cameras]

    # Load temporal-residual plugin.
    plugin = RayAttentionTemporalResidualFusionModule(
        j=17,
        d=args.d,
        n_views=args.n_views,
        checkpoint_path=args.checkpoint,
        input_scale=1.0,
    )
    plugin.model.to(device)
    plugin.model.eval()
    print(f"Loaded temporal-residual checkpoint: {args.checkpoint}")

    errors = []
    for frame in range(T):
        # SMPL forward for this frame.
        output = smpl_model(
            betas=betas,
            body_pose=body_pose[frame:frame + 1],
            global_orient=global_orient[frame:frame + 1],
            transl=transl[frame:frame + 1],
        )
        joints_3d = output.joints[0, :17].detach().cpu().numpy()  # (17, 3)

        # Project into each virtual camera.
        points_2d = np.stack([project_points(joints_3d, P) for P in P_list], axis=0)  # (V, J, 2)
        confidences = np.ones((args.n_views, 17), dtype=np.float64)

        # Add noise / outliers.
        if args.noise_std > 0:
            points_2d += np.random.randn(*points_2d.shape) * args.noise_std
        if args.dropout_rate > 0:
            mask = np.random.rand(args.n_views, 17) > args.dropout_rate
            confidences = confidences * mask
        if args.outlier_rate > 0:
            outlier_mask = np.random.rand(args.n_views, 17) < args.outlier_rate
            outlier = (np.random.rand(*points_2d.shape) - 0.5) * 2 * args.outlier_scale
            points_2d = np.where(outlier_mask[..., None], outlier, points_2d)

        # Fuse.
        fused = plugin.fuse(points_2d, confidences, cameras)
        errors.append(mpjpe_m(fused[-1], joints_3d))

    errors = np.array(errors)
    print(f"MPJPE vs GVHMR world reference: {errors.mean() * 1000:.4f} mm")
    print(f"Median: {np.median(errors) * 1000:.4f} mm")
    print(f"Max: {errors.max() * 1000:.4f} mm")


if __name__ == "__main__":
    main()
