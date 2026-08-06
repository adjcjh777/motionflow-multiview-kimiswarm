"""GVHMR multi-view projection demo integrating the metric ray-attention v1 plugin.

Loads a GVHMR hmr4d_results.pt, runs SMPL forward to obtain the single-view
world 3D joints, and treats these as the per-view SMPL joints for a set of
virtual calibrated cameras.  Each per-view joint set is projected into its
camera, perturbed by noise, and fed into the learned fusion plugins.

This lets us compare the fused multi-view output against the original
single-view GVHMR output (used here as the world-coordinate reference).

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/demo_gvhmr_multiview_projection.py
    /d/anaconda3/envs/jz_py310/python.exe experiments/demo_gvhmr_multiview_projection.py --max_frames 10

Note:
    The default run loads the metric-normalised ``RayAttentionFusionModule``
    (v1) checkpoint and uses ``input_scale`` to express the camera units in
    meters.  Projection and MPJPE are computed with PyTorch to work around a
    broken NumPy BLAS backend in this Windows Anaconda environment.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import smplx
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion import FUSION_REGISTRY
from motionflow_mv.fusion.ray_attention_v3_model import RayAttentionFusionModelV3
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
    """Return the 3x4 projection matrix as a torch tensor without NumPy matmul."""
    Rt = torch.cat(
        [torch.from_numpy(camera.R).float(), torch.from_numpy(camera.t.reshape(3, 1)).float()],
        dim=1,
    )
    K = torch.from_numpy(camera.K).float()
    P = K @ Rt  # torch handles this on CPU then we move to device
    return P.to(device)


def project_points(points_3d: np.ndarray, P: torch.Tensor):
    """Project (J, 3) to (J, 2) using a torch projection matrix.

    Avoids NumPy matrix-matrix multiplication, which is currently broken in the
    Windows Anaconda environment used for this demo.
    """
    X = torch.from_numpy(points_3d).float().to(P.device)
    J = X.shape[0]
    ones = torch.ones(J, 1, device=P.device)
    X_h = torch.cat([X, ones], dim=1)  # (J, 4)
    x_h = X_h @ P.T  # (J, 3)
    x = x_h[:, :2] / x_h[:, 2:3]
    return x.cpu().numpy()


def mpjpe_m(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint position error in meters, computed with torch."""
    pred_t = torch.from_numpy(pred).float()
    gt_t = torch.from_numpy(gt).float()
    return float(torch.mean(torch.norm(pred_t - gt_t, dim=-1)))


def load_ray_attention_v3(checkpoint_path: str | None, n_views: int, device: torch.device):
    """Instantiate RayAttentionFusionModelV3 and optionally load trained weights."""
    model = RayAttentionFusionModelV3(j=17, d=64, n_views=n_views).to(device)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            print(f"Loaded ray-attention v3 checkpoint: {checkpoint_path}")
        else:
            print(f"Warning: checkpoint not found, using random weights: {checkpoint_path}")
    else:
        print("No ray-attention v3 checkpoint provided; using random weights.")
    model.eval()
    return model


def cameras_to_tensors(cameras, device):
    """Convert a list of Camera objects to (K, R, t) tensors on device."""
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    return K, R, t


def main():
    parser = argparse.ArgumentParser(description="GVHMR multi-view projection demo.")
    parser.add_argument("--input", type=str, default="data/gvhmr_demo/hmr4d_results.pt")
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--shelf_checkpoints", action="store_true")
    parser.add_argument("--ray_v3_checkpoint", type=str, default=None,
                        help="Path to a RayAttentionFusionModelV3 checkpoint. If omitted, random weights are used.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Process only the first N frames for a quick test.")
    parser.add_argument("--ray_v1_checkpoint", type=str,
                        default="outputs/ray_attention_v1_s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.pth",
                        help="Path to the metric-normalised RayAttentionFusionModel v1 checkpoint.")
    parser.add_argument("--ray_v1_input_scale", type=float, default=1.0,
                        help="Scale factor to convert input camera units to meters (e.g. 1000 for mm, 100 for cm, 1 for m).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load GVHMR result and SMPL model.
    ir = gvhmr_pt_to_ir(args.input)
    smpl_model = smplx.SMPL("data/smpl/SMPL_NEUTRAL.pkl", batch_size=1)

    T = len(ir.timestamps)
    if args.max_frames is not None:
        T = min(T, args.max_frames)
    print(f"Loaded {T} frames from {args.input}")

    # Use mean betas over time to keep a single shape.
    betas = torch.from_numpy(ir.pose["betas"]).mean(dim=0, keepdim=True).float()
    body_pose = torch.from_numpy(ir.pose["body_pose"]).float()
    global_orient = torch.from_numpy(ir.pose["global_orient"]).float()
    transl = torch.from_numpy(ir.pose["transl"]).float()

    # Generate virtual cameras and precompute torch projection matrices.
    cameras = make_cameras_on_circle(args.n_views)
    K, R, t = cameras_to_tensors(cameras, device)
    P_list = [camera_projection_tensor(cam, device) for cam in cameras]

    # Load optional ray-attention v3 model.
    ray_v3 = None
    if args.ray_v3_checkpoint is not None:
        ray_v3 = load_ray_attention_v3(args.ray_v3_checkpoint, args.n_views, device)

    # Load metric-normalised ray-attention v1 plugin.
    if Path(args.ray_v1_checkpoint).exists():
        from motionflow_mv.fusion.ray_attention_module import RayAttentionFusionModule
        ray_v1_module = RayAttentionFusionModule(
            j=17,
            d=64,
            n_views=args.n_views,
            checkpoint_path=args.ray_v1_checkpoint,
            input_scale=args.ray_v1_input_scale,
        )
        ray_v1_module.model.to(device)
        ray_v1_module.model.eval()
        FUSION_REGISTRY._modules["ray_attention"] = ray_v1_module
        print(f"Loaded metric ray-attention v1 checkpoint: {args.ray_v1_checkpoint} "
              f"(input_scale={args.ray_v1_input_scale})")
    else:
        print(f"Warning: ray-attention v1 checkpoint not found: {args.ray_v1_checkpoint}")

    # Load optional legacy plugin checkpoints for comparison.
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
    per_plugin_errors["ray_attention_v3"] = []
    gt_joints_all = []
    single_view_errors = []

    rng = np.random.default_rng(2025)
    for t_idx in range(T):
        # SMPL forward to get 3D joints (world coordinates, meters).
        # Pad body_pose to smplx expected length (69) if it comes from GVHMR (63).
        bp = body_pose[t_idx:t_idx + 1]
        if bp.shape[1] < 69:
            bp_padded = torch.zeros(1, 69)
            bp_padded[:, :bp.shape[1]] = bp
        else:
            bp_padded = bp
        with torch.no_grad():
            output = smpl_model(
                betas=betas,
                body_pose=bp_padded,
                global_orient=global_orient[t_idx:t_idx + 1],
                transl=transl[t_idx:t_idx + 1],
            )
        joints_3d = output.joints[0, :17].cpu().numpy()  # (17, 3)
        gt_joints_all.append(joints_3d)

        # Project to each view and add noise.
        points_2d_list = []
        confidences_list = []
        for P_cam in P_list:
            x = project_points(joints_3d, P_cam)
            x += rng.normal(0, args.noise_std, size=x.shape)
            points_2d_list.append(x)
            confidences_list.append(rng.uniform(0.8, 1.0, size=joints_3d.shape[0]))
        points_2d = np.stack(points_2d_list, axis=0)  # (V, J, 2)
        confidences = np.stack(confidences_list, axis=0)  # (V, J)

        points_2d_px = points_2d[None]
        points_2d_norm = (points_2d / 1000.0)[None]
        confidences_batch = confidences[None]

        for name in sorted(FUSION_REGISTRY.names()):
            # Only run the learned attention/ray-attention plugins.  The other
            # built-ins (DLT, residual/temporal refiner, etc.) rely on NumPy
            # linear algebra that segfaults in this Anaconda environment.
            if name not in ("attention", "ray_attention"):
                continue
            module = FUSION_REGISTRY.get(name)
            try:
                if name in ("attention", "attention_v2", "ray_attention"):
                    input_2d = points_2d_norm if name == "attention" else points_2d_px
                elif name == "robust_triangulation":
                    input_2d = points_2d_px
                else:
                    input_2d = points_2d_px
                pred_3d = module.fuse(input_2d, confidences_batch, cameras)
                if pred_3d.ndim == 3:
                    pred_3d = pred_3d[0]
                # joints_3d is in meters; ray_attention/attention already output meters,
                # the other plugins output millimeters.
                if name in ("attention", "attention_v2", "ray_attention"):
                    pred_3d_m = pred_3d
                else:
                    pred_3d_m = pred_3d / 1000.0
                err = mpjpe_m(pred_3d_m[None], joints_3d[None])
                per_plugin_errors[name].append(err)
            except Exception as e:
                print(f"Plugin {name} failed on frame {t_idx}: {e}")

        # Ray-attention v3 direct inference (camera-conditioned embeddings + view/joint attention + weighted DLT).
        if ray_v3 is not None:
            try:
                x = np.concatenate([points_2d[None], confidences_batch[..., None]], axis=-1)  # (1, V, J, 3)
                x_tensor = torch.from_numpy(x).float().to(device)
                with torch.no_grad():
                    pred_v3, _ = ray_v3(x_tensor, K=K, R=R, t=t)
                    pred_v3 = pred_v3.cpu().numpy()[0]  # (J, 3)
                err_v3 = mpjpe_m(pred_v3[None], joints_3d[None])
                per_plugin_errors["ray_attention_v3"].append(err_v3)
            except Exception as e:
                print(f"ray_attention_v3 failed on frame {t_idx}: {e}")

        # Single-view baseline: use the original GVHMR world joints (the same reference we triangulate against).
        # This is zero by construction but is recorded explicitly for comparison.
        single_view_errors.append(0.0)

    print("\nMPJPE (m) vs GVHMR single-view world joints")
    print(f"{'Method':<25} {'MPJPE':>10}")
    print("-" * 36)
    for name in sorted(per_plugin_errors.keys()):
        if per_plugin_errors[name]:
            errors = np.array(per_plugin_errors[name])
            print(f"{name:<25} {errors.mean():>10.4f}")
        else:
            print(f"{name:<25} {'N/A':>10}")

    # Single-view vs multi-view summary.
    print("\nSingle-view vs multi-view summary")
    print("-" * 36)
    print(f"{'single-view (GVHMR)':<25} {np.mean(single_view_errors):>10.4f}")
    if per_plugin_errors["ray_attention_v3"]:
        print(f"{'ray_attention_v3':<25} {np.mean(per_plugin_errors['ray_attention_v3']):>10.4f}")
    if per_plugin_errors.get("ray_attention"):
        print(f"{'ray_attention (v1)':<25} {np.mean(per_plugin_errors['ray_attention']):>10.4f}")


if __name__ == "__main__":
    main()
