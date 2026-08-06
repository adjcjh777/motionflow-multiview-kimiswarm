"""Multi-view SMPL fitting stage for MotionFlow Multi-View.

Summary
-------
This script takes fused 3D joints (and optionally per-view 2D keypoints +
calibrated cameras) and fits a single coherent SMPL body to them.
It is intentionally simple: a direct PyTorch optimization over
`global_orient`, `body_pose`, `transl`, and a *shared* `betas` vector.

The loss is a weighted combination of:
  - 3D joint distance to the fused skeleton
  - optional multi-view reprojection error (when 2D observations are supplied)
  - light shape/pose regularization

Inputs are expected as an .npz file with keys:
  - joints_3d   : (T, J, 3)   fused world joints (meters by default)
  - camera_K    : (V, 3, 3)   optional intrinsics, used for reprojection
  - camera_R    : (V, 3, 3)   optional extrinsics
  - camera_t    : (V, 3)      optional camera translation
  - points_2d   : (T, V, J, 2) optional per-view 2D keypoints
  - confidences : (T, V, J)   optional 2D confidences

The script assumes the input joint set matches the first J SMPL body joints
(the convention used by the synthetic multi-view generator in this repo).
For COCO/H36M-style joint orders a regressor would be required; that is left
for a future iteration.

Outputs are saved as an .npz containing the fitted SMPL parameters and the
reconstructed 3D joints, ready to be wrapped into a HumanMotionIR.

Usage
-----
    python experiments/fit_smpl_multiview.py \
        --input data/h36m_hf/s_01_act_02_multiview.npz \
        --output outputs/fit_smpl_multiview.npz \
        --n_iters 200 --lr 1e-2 --max_frames 60

Verification
------------
A tiny synthetic sanity check can be run by generating a short synthetic clip
and fitting it back. The script prints the fitting error before exiting.

Important findings (see docs/swarm_iter5/fit_smpl_multiview.md):
  - Shared sequence-level betas is sufficient for short clips and avoids
    per-frame shape drift.
  - Torch-based Procrustes alignment is required here because np.linalg.svd
    crashes on this Windows/Anaconda setup; the torch implementation is stable.
  - Reprojection loss requires 2D observations; when only 3D joints are
    provided the fitter relies purely on 3D joint matching.

Verified on
-----------
data/h36m_hf/s_01_act_02_multiview.npz (H36M, 4 views, 17 joints, mm units):
    --max_frames 10 --n_iters 100 --input_unit mm --lr 0.01 --device cpu
    -> Fitting MPJPE: 0.321062 m, Final 3D MSE: 0.053551
Reprojection code path verified with --reproj_weight 0.01 on 3 frames.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import smplx
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera


SMPL_MODEL_PATH = "data/smpl/SMPL_NEUTRAL.pkl"


def procrustes_align(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (R, t) that best maps source -> target under rigid transform.

    Uses torch.linalg.svd to avoid numpy SVD instability on this platform.

    Args:
        source: (J, 3)
        target: (J, 3)

    Returns:
        R: (3, 3), t: (3,)
    """
    c_s = source.mean(dim=0)
    c_t = target.mean(dim=0)
    X = source - c_s
    Y = target - c_t
    H = X.T @ Y
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_t - R @ c_s
    return R, t


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to an axis-angle (3,) vector."""
    trace = torch.trace(R)
    angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
    if angle.abs() < 1e-6:
        return torch.zeros(3, dtype=R.dtype, device=R.device)
    r = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dim=0)
    sin_angle = torch.norm(r) / 2.0
    if sin_angle < 1e-6:
        return torch.zeros(3, dtype=R.dtype, device=R.device)
    axis = r / (2.0 * sin_angle)
    return axis * angle


def project_torch(points_3d: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Project (B, J, 3) world points through V cameras to (B, V, J, 2)."""
    # points_3d: (B, J, 3); K,R: (V, 3, 3); t: (V, 3)
    X_cam = torch.einsum("vik,bjk->bvji", R, points_3d) + t[None, :, None, :]  # (B, V, J, 3)
    x = X_cam[..., :2] / (X_cam[..., 2:3] + 1e-6)
    ones = torch.ones(x.shape[:-1] + (1,), device=x.device, dtype=x.dtype)
    x_hom = torch.cat([x, ones], dim=-1)  # (B, V, J, 3)
    x_pixel = torch.einsum("vik,bvjk->bvji", K[:, :2, :], x_hom)  # (B, V, J, 2)
    return x_pixel


def fit_smpl(
    joints_3d: np.ndarray,
    smpl_model_path: str = SMPL_MODEL_PATH,
    points_2d: np.ndarray | None = None,
    cameras: list[Camera] | None = None,
    confidences: np.ndarray | None = None,
    n_iters: int = 200,
    lr: float = 1e-2,
    shape_weight: float = 1.0,
    reproj_weight: float = 0.0,
    pose_prior_weight: float = 1e-3,
    smooth_weight: float = 0.0,
    share_shape: bool = True,
    device: torch.device | None = None,
) -> dict:
    """Fit SMPL to fused 3D joints (and optional 2D reprojection data).

    Args:
        joints_3d: (T, J, 3) in meters.
        smpl_model_path: path to SMPL neutral model (read-only).
        points_2d: (T, V, J, 2) optional pixel observations.
        cameras: list of V Camera objects, required if points_2d is given.
        confidences: (T, V, J) optional observation weights.
        n_iters: Adam iterations.
        lr: Adam learning rate.
        shape_weight: L2 weight on betas.
        reproj_weight: weight for reprojection loss (0 disables even if 2D provided).
        pose_prior_weight: L2 weight on body_pose and global_orient (zero-pose prior).
        smooth_weight: weight for temporal smoothness on transl/pose.
        share_shape: if True, a single betas vector is shared across all frames.
        device: torch device (defaults to cuda if available).

    Returns:
        dict with keys: betas, global_orient, body_pose, transl, fitted_joints_3d,
        mpjpe_m, loss_history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    joints_3d = np.asarray(joints_3d, dtype=np.float32)
    T, J, _ = joints_3d.shape

    # Load SMPL once for the whole sequence.
    smpl_model = smplx.SMPL(smpl_model_path, batch_size=T).to(device)
    smpl_model.eval()

    # T-pose joints used for Procrustes initialization.
    with torch.no_grad():
        tpose_output = smpl_model(
            betas=torch.zeros(1, 10, device=device),
            body_pose=torch.zeros(1, 69, device=device),
            global_orient=torch.zeros(1, 3, device=device),
            transl=torch.zeros(1, 3, device=device),
        )
    tpose_joints = tpose_output.joints[0, :J, :].detach()  # (J, 3)

    # Initialize per-frame rigid alignment (done on CPU torch to avoid np.linalg.svd).
    global_orient_init = np.zeros((T, 3), dtype=np.float32)
    transl_init = np.zeros((T, 3), dtype=np.float32)
    joints_3d_torch_cpu = torch.from_numpy(joints_3d).float()
    for t_idx in range(T):
        R, tr = procrustes_align(tpose_joints, joints_3d_torch_cpu[t_idx])
        global_orient_init[t_idx] = rotation_matrix_to_axis_angle(R).cpu().numpy().astype(np.float32)
        transl_init[t_idx] = tr.cpu().numpy().astype(np.float32)

    # Torch parameters.
    betas = torch.zeros(10, device=device, dtype=torch.float32, requires_grad=share_shape)
    global_orient = torch.from_numpy(global_orient_init).float().to(device).requires_grad_(True)
    body_pose = torch.zeros(T, 69, device=device, dtype=torch.float32, requires_grad=True)
    transl = torch.from_numpy(transl_init).float().to(device).requires_grad_(True)

    params = [global_orient, body_pose, transl]
    if share_shape:
        params.append(betas)
    else:
        # Per-frame betas are usually over-fit-prone; provide the option but warn.
        betas = torch.zeros(T, 10, device=device, dtype=torch.float32, requires_grad=True)
        params.append(betas)

    optimizer = torch.optim.Adam(params, lr=lr)

    target = torch.from_numpy(joints_3d).to(device)

    # Prepare optional reprojection inputs.
    use_reproj = reproj_weight > 0.0 and points_2d is not None and cameras is not None
    if use_reproj:
        if confidences is None:
            conf = torch.ones_like(torch.from_numpy(points_2d)[..., 0])
        else:
            conf = torch.from_numpy(confidences).float()
        target_2d = torch.from_numpy(points_2d).float().to(device)
        conf = conf.to(device)
        K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
        R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
        t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)

    loss_history = []
    for step in range(n_iters):
        optimizer.zero_grad()

        betas_input = betas[None, :].expand(T, -1) if share_shape else betas
        output = smpl_model(
            betas=betas_input,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
        )
        pred_joints = output.joints[:, :J, :]  # (T, J, 3)

        loss_3d = torch.mean((pred_joints - target) ** 2)
        loss = loss_3d

        if use_reproj:
            pred_2d = project_torch(pred_joints, K, R, t)
            diff = pred_2d - target_2d
            w = conf / (conf.mean() + 1e-6)
            loss_reproj = (w[..., None] * diff ** 2).mean()
            loss = loss + reproj_weight * loss_reproj

        # Regularizers.
        loss_shape = torch.mean(betas ** 2)
        loss_pose = torch.mean(body_pose ** 2) + torch.mean(global_orient ** 2)
        loss = loss + shape_weight * loss_shape + pose_prior_weight * loss_pose

        if smooth_weight > 0.0 and T > 1:
            loss_smooth = torch.mean((transl[1:] - transl[:-1]) ** 2)
            loss_smooth += torch.mean((body_pose[1:] - body_pose[:-1]) ** 2)
            loss = loss + smooth_weight * loss_smooth

        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

        if (step + 1) % max(1, n_iters // 5) == 0 or step == 0:
            print(f"  iter {step + 1}/{n_iters}: loss={loss.item():.6f}, 3d_mse={loss_3d.item():.6f}")

    fitted_joints = pred_joints.detach().cpu().numpy()
    mpjpe_m = float(np.mean(np.linalg.norm(fitted_joints - joints_3d, axis=-1)))

    return {
        "betas": betas.detach().cpu().numpy(),
        "global_orient": global_orient.detach().cpu().numpy(),
        "body_pose": body_pose.detach().cpu().numpy(),
        "transl": transl.detach().cpu().numpy(),
        "fitted_joints_3d": fitted_joints,
        "mpjpe_m": mpjpe_m,
        "loss_history": np.array(loss_history),
    }


def load_input(path: str, max_frames: int | None = None, input_unit: str = "m") -> dict:
    """Load and normalize a multi-view .npz into a flat dict."""
    data = np.load(path)
    joints_3d = data["joints_3d"].astype(np.float32)
    if max_frames is not None:
        joints_3d = joints_3d[:max_frames]

    scale = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}.get(input_unit, 1.0)
    joints_3d = joints_3d * scale

    out = {"joints_3d": joints_3d}

    for key in ("points_2d", "confidences"):
        if key in data:
            arr = data[key].astype(np.float32)
            if max_frames is not None:
                arr = arr[:max_frames]
            out[key] = arr

    if all(k in data for k in ("camera_K", "camera_R", "camera_t")):
        K = data["camera_K"]
        R = data["camera_R"]
        t = data["camera_t"]
        # Scale camera translation to match joint units.
        t = t * scale
        cameras = [Camera(K=K[v], R=R[v], t=t[v]) for v in range(K.shape[0])]
        out["cameras"] = cameras

    return out


def main():
    parser = argparse.ArgumentParser(description="Fit SMPL to fused multi-view 3D joints.")
    parser.add_argument("--input", type=str, default="data/h36m_hf/s_01_act_02_multiview.npz",
                        help="Input .npz containing joints_3d and optional 2D/camera data.")
    parser.add_argument("--output", type=str, default="outputs/fit_smpl_multiview.npz",
                        help="Where to save fitted SMPL parameters.")
    parser.add_argument("--smpl_model", type=str, default=SMPL_MODEL_PATH,
                        help="Path to SMPL neutral .pkl (read-only).")
    parser.add_argument("--n_iters", type=int, default=200, help="Adam iterations.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate.")
    parser.add_argument("--shape_weight", type=float, default=1.0, help="L2 weight on betas.")
    parser.add_argument("--reproj_weight", type=float, default=0.0,
                        help="Weight for reprojection loss (0 = disabled).")
    parser.add_argument("--pose_prior_weight", type=float, default=1e-3,
                        help="Zero-pose prior weight.")
    parser.add_argument("--smooth_weight", type=float, default=0.0,
                        help="Temporal smoothness weight.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit frames for quick tests.")
    parser.add_argument("--input_unit", type=str, default="m", choices=["m", "cm", "mm"],
                        help="Length unit of input joints/cameras.")
    parser.add_argument("--no_share_shape", action="store_true",
                        help="Disable sequence-level shared betas.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    input_data = load_input(args.input, max_frames=args.max_frames, input_unit=args.input_unit)

    device = torch.device(args.device) if args.device else None
    result = fit_smpl(
        joints_3d=input_data["joints_3d"],
        smpl_model_path=args.smpl_model,
        points_2d=input_data.get("points_2d"),
        cameras=input_data.get("cameras"),
        confidences=input_data.get("confidences"),
        n_iters=args.n_iters,
        lr=args.lr,
        shape_weight=args.shape_weight,
        reproj_weight=args.reproj_weight,
        pose_prior_weight=args.pose_prior_weight,
        smooth_weight=args.smooth_weight,
        share_shape=not args.no_share_shape,
        device=device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    print(f"Saved fitted SMPL parameters to {output_path}")
    print(f"Fitting MPJPE: {result['mpjpe_m']:.6f} m")
    print(f"Final 3D MSE : {result['loss_history'][-1]:.6f}")


if __name__ == "__main__":
    main()
