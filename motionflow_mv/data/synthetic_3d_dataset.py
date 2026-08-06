"""Synthetic 3D multi-view dataset generator with SMPL/AMASS augmentation.

This module produces calibrated multi-view 2D keypoints, confidences, and 3D
ground truth.  It can be used in two ways:

1. **Legacy simple mode** -- the original random-skeleton generator used by
   small fusion smoke tests (``make_cameras``, ``generate_sequence``,
   ``generate_dataset``).
2. **SMPL/AMASS mode** -- realistic human motion via SMPL (and optionally
   AMASS) with domain-randomized cameras, 2D noise, occlusion, outliers, and
   temporal augmentation (``CameraRigSampler``, ``MotionSampler``,
   ``SMPLSequenceGenerator``, ``SyntheticMultiViewDataset``).

All new SMPL helpers are written so that the code degrades gracefully when
optional dependencies (``smplx``) or data (AMASS) are absent: procedural motion
sampling is used as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.triangulation import triangulate_confidence_weighted


def _check_smplx():
    """Return the smplx module, or raise ImportError with a helpful message."""
    try:
        import smplx  # noqa: F401

        return smplx
    except ImportError as exc:
        raise ImportError(
            "smplx is required for SMPL/AMASS synthetic generation. "
            "Install it with: pip install smplx"
        ) from exc


# ---------------------------------------------------------------------------
# Legacy simple generator (kept for backward compatibility)
# ---------------------------------------------------------------------------

def make_cameras(n_views: int = 5, rng: np.random.Generator = None):
    """Return a list of calibrated pinhole cameras on a circle."""
    if rng is None:
        rng = np.random.default_rng(123)
    cameras = []
    for i in range(n_views):
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[:2, 2] = rng.uniform(300, 340, size=2)
        R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        theta = 2 * np.pi * i / n_views
        phi = np.pi / 3
        radius = 5.0
        c = radius * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        t = -R @ c
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def generate_sequence(
    n_frames: int = 10,
    n_views: int = 5,
    j: int = 17,
    rng: np.random.Generator = None,
    noise_std: float = 1.0,
):
    """Return one synthetic sequence: inputs (T, V, J, 3), baselines (T, J, 3), gt (T, J, 3)."""
    if rng is None:
        rng = np.random.default_rng(0)

    # Smooth base skeleton trajectory.
    base = rng.uniform(-1.0, 1.0, size=(j, 3)) * np.array([0.5, 0.8, 1.5])
    base[:, 2] += 3.0
    positions = [base.copy()]
    for _ in range(n_frames - 1):
        delta = rng.normal(0, 0.05, size=base.shape)
        base = base + delta
        positions.append(base.copy())
    positions = np.stack(positions, axis=0)  # (T, J, 3)

    cameras = make_cameras(n_views, rng)
    proj = [cam.projection_matrix for cam in cameras]

    inputs = []
    baselines = []
    for t in range(n_frames):
        X = positions[t]
        points_2d = []
        conf = []
        for cam in cameras:
            P = cam.projection_matrix
            X_h = np.hstack([X, np.ones((j, 1))])
            x_h = (P @ X_h.T).T
            x = x_h[:, :2] / x_h[:, 2:3]
            x += rng.normal(0, noise_std, size=x.shape)
            points_2d.append(x)
            conf.append(rng.uniform(0.5, 1.0, size=j))
        points_2d = np.stack(points_2d, axis=0)  # (V, J, 2)
        conf = np.stack(conf, axis=0)  # (V, J)
        inputs.append(np.concatenate([points_2d, conf[..., None]], axis=-1))

        # Baseline DLT from noisy points.
        baseline = np.zeros((j, 3), dtype=np.float64)
        proj_arr = np.stack(proj, axis=0)
        for joint_idx in range(j):
            baseline[joint_idx] = triangulate_confidence_weighted(
                points_2d[:, joint_idx, :],
                proj_arr,
                conf[:, joint_idx],
            )
        baselines.append(baseline)

    return (
        torch.tensor(np.stack(inputs, axis=0), dtype=torch.float32),
        torch.tensor(np.stack(baselines, axis=0), dtype=torch.float32),
        torch.tensor(positions, dtype=torch.float32),
        cameras,
    )


def generate_dataset(
    n_seq: int,
    n_frames: int = 10,
    n_views: int = 5,
    j: int = 17,
    seed: int = 0,
    noise_std: float = 1.0,
):
    rng = np.random.default_rng(seed)
    X, B, Y = [], [], []
    for _ in range(n_seq):
        inp, base, gt, _ = generate_sequence(n_frames, n_views, j, rng, noise_std)
        X.append(inp)
        B.append(base)
        Y.append(gt)
    return torch.stack(X), torch.stack(B), torch.stack(Y)


# ---------------------------------------------------------------------------
# SMPL / AMASS motion synthesis
# ---------------------------------------------------------------------------

@dataclass
class CameraRigSampler:
    """Sample randomized calibrated camera rigs.

    Supported modes:

    * ``legacy`` -- generic circular rigs in metres.
    * ``h36m``   -- match the Human3.6M four-camera distribution (millimetres).
    * ``mpiinf3dhp`` -- emulate the wider MPI-INF-3DHP-like ring (millimetres).
    * ``random`` -- fully randomized intrinsics/extrinsics for domain randomization.
    """

    mode: str = "h36m"
    # H36M camera statistics measured from h36m_hf preprocessed data.
    h36m_stats: dict = field(default_factory=lambda: {
        "distance_mm": (5318.75, 523.05),
        "z_mm": (1559.14, 41.83),
        "focal_mm": (1147.34, 2.08),
        "cx_mm": (512.04, 3.98),
        "cy_mm": (506.70, 5.69),
    })
    h36m_azimuths: np.ndarray = field(
        default_factory=lambda: np.array([1.215, -1.237, 1.911, -2.020], dtype=np.float64)
    )

    def sample(self, n_views: int, rng: np.random.Generator) -> List[Camera]:
        """Sample a rig with ``n_views`` cameras."""
        if self.mode == "legacy":
            return self._legacy_rig(n_views, rng)
        if self.mode == "h36m":
            return self._h36m_rig(n_views, rng)
        if self.mode == "mpiinf3dhp":
            return self._mpi_rig(n_views, rng)
        if self.mode == "random":
            return self._random_rig(n_views, rng)
        raise ValueError(f"Unknown camera mode: {self.mode}")

    def _legacy_rig(self, n_views: int, rng: np.random.Generator) -> List[Camera]:
        radius = rng.uniform(3.0, 6.0)
        height = rng.uniform(0.5, 2.5)
        focal = rng.uniform(600.0, 1200.0)
        cx = rng.uniform(300.0, 340.0)
        cy = rng.uniform(220.0, 260.0)
        phi_base = rng.uniform(np.pi / 6.0, np.pi / 3.0)
        thetas = 2.0 * np.pi * np.arange(n_views) / n_views + rng.uniform(-0.1, 0.1)
        return self._spherical_rig(thetas, phi_base, radius, height, focal, cx, cy, rng)

    def _h36m_rig(self, n_views: int, rng: np.random.Generator) -> List[Camera]:
        focal = rng.normal(*self.h36m_stats["focal_mm"])
        cx = rng.normal(*self.h36m_stats["cx_mm"])
        cy = rng.normal(*self.h36m_stats["cy_mm"])
        distance = rng.normal(*self.h36m_stats["distance_mm"])
        z_height = rng.normal(*self.h36m_stats["z_mm"])
        distance = max(distance, z_height + 100.0)
        r_xy = np.sqrt(max(distance ** 2 - z_height ** 2, 0.0))

        if n_views == 4:
            yaw = rng.uniform(0.0, 2.0 * np.pi)
            thetas = self.h36m_azimuths + yaw
        else:
            thetas = 2.0 * np.pi * np.arange(n_views) / n_views + rng.uniform(0.0, 2.0 * np.pi)
        return self._spherical_rig(thetas, None, None, None, focal, cx, cy, rng, r_xy=r_xy, z_height=z_height)

    def _mpi_rig(self, n_views: int, rng: np.random.Generator) -> List[Camera]:
        # MPI-INF-3DHP uses cameras on a wider ring, ~4-7 m from the subject.
        radius = rng.uniform(4.0, 7.0)
        height = rng.uniform(0.8, 2.5)
        focal = rng.uniform(700.0, 1300.0)
        cx = rng.uniform(300.0, 370.0)
        cy = rng.uniform(200.0, 280.0)
        phi_base = rng.uniform(np.pi / 8.0, np.pi / 4.0)
        thetas = 2.0 * np.pi * np.arange(n_views) / n_views + rng.uniform(-0.15, 0.15)
        return self._spherical_rig(thetas, phi_base, radius, height, focal, cx, cy, rng)

    def _random_rig(self, n_views: int, rng: np.random.Generator) -> List[Camera]:
        cameras = []
        for _ in range(n_views):
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = rng.uniform(500.0, 1500.0)
            K[1, 1] = K[0, 0] * rng.uniform(0.98, 1.02)
            K[0, 2] = rng.uniform(250.0, 400.0)
            K[1, 2] = rng.uniform(180.0, 320.0)
            R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            if np.linalg.det(R) < 0:
                R[:, 0] *= -1
            c = rng.normal(0.0, 1.0, size=3)
            c[2] = abs(c[2]) * 2.0 + 0.5
            t = -R @ c
            cameras.append(Camera(K=K, R=R, t=t))
        return cameras

    @staticmethod
    def _spherical_rig(
        thetas: np.ndarray,
        phi_base: Optional[float],
        radius: Optional[float],
        height: Optional[float],
        focal: float,
        cx: float,
        cy: float,
        rng: np.random.Generator,
        r_xy: Optional[float] = None,
        z_height: Optional[float] = None,
    ) -> List[Camera]:
        cameras = []
        for theta in thetas:
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = K[1, 1] = focal
            K[0, 2] = cx
            K[1, 2] = cy

            if r_xy is not None and z_height is not None:
                c = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), z_height], dtype=np.float64)
            else:
                phi = phi_base + rng.uniform(-0.1, 0.1)
                c = radius * np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                ])
                c[2] += height

            forward = -c / np.linalg.norm(c)
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            R = np.stack([right, up, -forward], axis=0)
            t = -R @ c
            cameras.append(Camera(K=K, R=R, t=t))
        return cameras


@dataclass
class MotionSampler:
    """Sample SMPL-compatible motion parameters.

    If ``amass_root`` is provided and contains ``*_poses.npz`` files, real
    AMASS poses are loaded and sampled.  Otherwise, smooth procedural poses
    are generated via Brownian motion on latent joint angles.
    """

    amass_root: Optional[str] = None
    smpl_model_path: Optional[str] = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self):
        if self.amass_root is not None:
            self.amass_files = sorted(Path(self.amass_root).glob("*_poses.npz"))
        else:
            self.amass_files = []
        self._amass_cache: dict = {}

    def sample(
        self,
        n_frames: int,
        rng: np.random.Generator,
        smpl_model,
        betas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (body_pose, global_orient, transl) each of shape (n_frames, ...).

        The tensors are torch arrays ready to be fed to ``smpl_model``.
        """
        if self.amass_files and rng.random() < 0.7:
            try:
                return self._sample_amass(n_frames, rng, smpl_model, betas)
            except Exception:
                # Fall back to procedural if AMAASS sample is malformed.
                pass
        return self._sample_procedural(n_frames, rng)

    def _sample_amass(
        self,
        n_frames: int,
        rng: np.random.Generator,
        smpl_model,
        betas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file = rng.choice(self.amass_files)
        key = str(file)
        if key in self._amass_cache:
            data = self._amass_cache[key]
        else:
            data = np.load(file)
            self._amass_cache[key] = data

        poses = data["poses"]  # (T, 72) or (T, ?)
        trans = data["trans"]  # (T, 3)
        total = poses.shape[0]

        if n_frames >= total:
            start = 0
            clip_len = total
        else:
            start = rng.integers(0, total - n_frames + 1)
            clip_len = n_frames
        end = start + clip_len

        body_pose = torch.from_numpy(poses[start:end, 3:72]).float().to(self.device)
        global_orient = torch.from_numpy(poses[start:end, :3]).float().to(self.device)
        transl = torch.from_numpy(trans[start:end]).float().to(self.device)
        return body_pose, global_orient, transl

    def _sample_procedural(
        self,
        n_frames: int,
        rng: np.random.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        body_pose = []
        global_orient = []
        transl = []
        for _ in range(n_frames):
            body_pose.append(rng.normal(0, 0.3, size=69).astype(np.float32))
            global_orient.append(rng.normal(0, 0.2, size=3).astype(np.float32))
            transl.append((rng.normal(0, 0.2, size=3) + np.array([0.0, 0.0, 1.0])).astype(np.float32))
        return (
            torch.from_numpy(np.stack(body_pose, axis=0)).to(self.device),
            torch.from_numpy(np.stack(global_orient, axis=0)).to(self.device),
            torch.from_numpy(np.stack(transl, axis=0)).to(self.device),
        )


@dataclass
class AugmentConfig:
    """Configuration for per-frame 2D augmentation."""

    noise_std: float = 1.0
    occlusion_rate: float = 0.1
    outlier_rate: float = 0.02
    outlier_scale: float = 100.0
    mirror_prob: float = 0.0
    scale_jitter: float = 0.0
    camera_jitter: float = 0.0


def augment_2d_keypoints(
    points_2d: np.ndarray,
    confidences: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply noise, occlusion, outliers, mirror, and scale jitter to (V, J, 2).

    Args:
        points_2d: ``(V, J, 2)`` array of 2D keypoints.
        confidences: ``(V, J)`` array of confidence scores.
        rng: numpy random generator.
        config: augmentation configuration.

    Returns:
        Augmented ``points_2d`` and ``confidences``.
    """
    points_2d = points_2d.copy()
    confidences = confidences.copy()
    V, J, _ = points_2d.shape

    # Gaussian noise.
    if config.noise_std > 0:
        points_2d += rng.normal(0, config.noise_std, size=points_2d.shape)

    # Occlusion.
    if config.occlusion_rate > 0:
        occ_mask = rng.random((V, J)) < config.occlusion_rate
        points_2d[occ_mask] = 0.0
        confidences[occ_mask] = 0.0

    # Outliers.
    if config.outlier_rate > 0:
        outlier_mask = rng.random((V, J)) < config.outlier_rate
        num_outliers = outlier_mask.sum()
        if num_outliers:
            points_2d[outlier_mask] += rng.normal(
                0, config.outlier_scale, size=(num_outliers, 2)
            )
            confidences[outlier_mask] = 0.0

    # Horizontal mirror (flip x around principal point cx; approximate by image width/2).
    if config.mirror_prob > 0 and rng.random() < config.mirror_prob:
        points_2d[..., 0] = -points_2d[..., 0]

    # Scale jitter (rare; simulates focal-length / subject-distance uncertainty).
    if config.scale_jitter > 0:
        scale = rng.lognormal(0.0, config.scale_jitter)
        points_2d *= scale

    return points_2d, confidences


@dataclass
class SMPLSequenceGenerator:
    """Generate one synthetic multi-view sequence from a SMPL model.

    Example::

        gen = SMPLSequenceGenerator(camera_mode="h36m")
        points_2d, confidences, joints_3d, cameras = gen.generate(
            smpl_model, betas, n_frames=30, n_views=4, rng=rng
        )
    """

    camera_sampler: CameraRigSampler = field(default_factory=lambda: CameraRigSampler("h36m"))
    motion_sampler: MotionSampler = field(default_factory=MotionSampler)
    augment_config: AugmentConfig = field(default_factory=AugmentConfig)
    world_scale: float = 1000.0
    use_triangulated_baseline: bool = True
    n_joints: int = 17

    def generate(
        self,
        smpl_model,
        betas: torch.Tensor,
        n_frames: int,
        n_views: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Camera], np.ndarray]:
        """Generate a sequence.

        Returns:
            points_2d: ``(T, V, J, 2)``
            confidences: ``(T, V, J)``
            joints_3d: ``(T, J, 3)``
            cameras: list of ``Camera`` objects.
            baseline_3d: ``(T, J, 3)`` triangulated baseline from noisy 2D points.
        """
        body_pose, global_orient, transl = self.motion_sampler.sample(
            n_frames, rng, smpl_model, betas
        )

        cameras = self.camera_sampler.sample(n_views, rng)

        joints_3d_list = []
        points_2d_list = []
        confidences_list = []
        baselines = []

        for f in range(n_frames):
            with torch.no_grad():
                output = smpl_model(
                    betas=betas,
                    body_pose=body_pose[f:f + 1],
                    global_orient=global_orient[f:f + 1],
                    transl=transl[f:f + 1],
                )
            joints_3d = output.joints[0, : self.n_joints].cpu().numpy()  # (J, 3)
            joints_3d = joints_3d * self.world_scale
            joints_3d_list.append(joints_3d)

            p2d_frame = []
            conf_frame = []
            for cam in cameras:
                x = project_points(joints_3d, cam)
                conf = rng.uniform(0.8, 1.0, size=self.n_joints)
                x, conf = augment_2d_keypoints(x[None], conf[None], rng, self.augment_config)
                p2d_frame.append(x[0])
                conf_frame.append(conf[0])

            points_2d_list.append(np.stack(p2d_frame, axis=0))
            confidences_list.append(np.stack(conf_frame, axis=0))

            if self.use_triangulated_baseline:
                baseline = triangulate_joints_baseline(
                    points_2d_list[-1], confidences_list[-1], cameras
                )
                baselines.append(baseline)

        joints_3d = np.stack(joints_3d_list, axis=0)
        points_2d = np.stack(points_2d_list, axis=0)
        confidences = np.stack(confidences_list, axis=0)
        baseline_3d = np.stack(baselines, axis=0) if baselines else joints_3d.copy()

        return points_2d, confidences, joints_3d, cameras, baseline_3d


def project_points(points_3d: np.ndarray, camera: Camera) -> np.ndarray:
    """Project ``(J, 3)`` to ``(J, 2)`` using a ``Camera``.

    Uses torch internally to avoid a numpy BLAS/MKL crash observed on the
    Windows + Git Bash python runner while keeping the public API unchanged.
    """
    K = torch.from_numpy(camera.K).float()
    R = torch.from_numpy(camera.R).float()
    t = torch.from_numpy(camera.t).float()
    X = torch.from_numpy(np.asarray(points_3d, dtype=np.float64)).float()
    X_h = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    Rt = torch.cat([R, t.unsqueeze(1)], dim=1)
    P = K @ Rt
    x_h = (P @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return x.numpy()


def triangulate_joints_baseline(
    points_2d: np.ndarray,
    confidences: np.ndarray,
    cameras: List[Camera],
) -> np.ndarray:
    """Triangulate ``(V, J, 2)`` keypoints to ``(J, 3)`` using confidence weights."""
    V, J, _ = points_2d.shape
    proj = np.stack([cam.projection_matrix for cam in cameras], axis=0)
    baseline = np.zeros((J, 3), dtype=np.float64)
    for j in range(J):
        baseline[j] = triangulate_confidence_weighted(
            points_2d[:, j, :], proj, confidences[:, j]
        )
    return baseline


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class SyntheticMultiViewDataset(Dataset):
    """PyTorch Dataset that generates synthetic multi-view sequences on the fly.

    This is useful for training augmentation: each call generates a fresh
    sequence, so the model sees an effectively infinite stream of domain
    randomized data.  The returned tuple is compatible with
    ``temporal_clip_dataset.collate_fn``::

        x, y, K, R, t

    where ``x`` has shape ``(T, V, J, 3)`` and contains the 2D keypoints plus
    confidence channel.
    """

    def __init__(
        self,
        smpl_model_path: str = "data/smpl/SMPL_NEUTRAL.pkl",
        n_sequences: int = 500,
        n_frames: int = 30,
        n_views: int = 4,
        camera_mode: str = "h36m",
        amass_root: Optional[str] = None,
        augment_config: Optional[AugmentConfig] = None,
        world_scale: float = 1000.0,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        n_joints: int = 17,
    ):
        _check_smplx()
        self.smpl_model_path = smpl_model_path
        self.n_sequences = n_sequences
        self.n_frames = n_frames
        self.n_views = n_views
        self.world_scale = world_scale
        self.n_joints = n_joints
        self.seed = seed

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.smpl_model = self._load_smpl(smpl_model_path)

        self.generator = SMPLSequenceGenerator(
            camera_sampler=CameraRigSampler(camera_mode),
            motion_sampler=MotionSampler(amass_root=amass_root, device=self.device),
            augment_config=augment_config or AugmentConfig(),
            world_scale=world_scale,
            n_joints=n_joints,
        )

    def _load_smpl(self, path: str):
        import smplx

        model = smplx.SMPL(path, batch_size=1)
        return model.to(self.device)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        # Use a deterministic but unique seed per sample so that different
        # workers in a DataLoader do not produce identical sequences.
        seed = (self.seed + idx) if self.seed is not None else idx
        rng = np.random.default_rng(seed)

        betas = torch.randn(1, 10, device=self.device) * 0.1
        points_2d, confidences, joints_3d, cameras, _ = self.generator.generate(
            self.smpl_model, betas, self.n_frames, self.n_views, rng
        )

        K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float()
        R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float()
        t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float()

        x = torch.from_numpy(np.concatenate([points_2d, confidences[..., None]], axis=-1)).float()
        y = torch.from_numpy(joints_3d).float()
        return x, y, K, R, t


# ---------------------------------------------------------------------------
# Convenience: generate a canonical .npz file from the new generator.
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    output: str,
    smpl_model_path: str = "data/smpl/SMPL_NEUTRAL.pkl",
    n_sequences: int = 500,
    n_frames: int = 30,
    n_views: int = 4,
    camera_mode: str = "h36m",
    amass_root: Optional[str] = None,
    augment_config: Optional[AugmentConfig] = None,
    world_scale: float = 1000.0,
    seed: int = 2025,
    device: Optional[torch.device] = None,
    n_joints: int = 17,
) -> Path:
    """Generate a canonical ``.npz`` synthetic dataset using the new generator.

    Returns the path to the saved ``.npz`` file.
    """
    _check_smplx()
    import smplx

    rng = np.random.default_rng(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smpl_model = smplx.SMPL(smpl_model_path, batch_size=1).to(device)
    generator = SMPLSequenceGenerator(
        camera_sampler=CameraRigSampler(camera_mode),
        motion_sampler=MotionSampler(amass_root=amass_root, device=device),
        augment_config=augment_config or AugmentConfig(),
        world_scale=world_scale,
        n_joints=n_joints,
    )

    all_points_2d = []
    all_confidences = []
    all_joints_3d = []
    all_camera_K = []
    all_camera_R = []
    all_camera_t = []

    for seq_idx in range(n_sequences):
        betas = torch.randn(1, 10, device=device) * 0.1
        points_2d, confidences, joints_3d, cameras, _ = generator.generate(
            smpl_model, betas, n_frames, n_views, rng
        )

        K_arr = np.stack([cam.K for cam in cameras], axis=0)
        R_arr = np.stack([cam.R for cam in cameras], axis=0)
        t_arr = np.stack([cam.t for cam in cameras], axis=0)

        all_points_2d.append(points_2d)
        all_confidences.append(confidences)
        all_joints_3d.append(joints_3d)
        all_camera_K.append(np.tile(K_arr[None], (n_frames, 1, 1, 1)))
        all_camera_R.append(np.tile(R_arr[None], (n_frames, 1, 1, 1)))
        all_camera_t.append(np.tile(t_arr[None], (n_frames, 1, 1)))

    data = {
        "points_2d": np.concatenate(all_points_2d, axis=0),
        "confidences": np.concatenate(all_confidences, axis=0),
        "joints_3d": np.concatenate(all_joints_3d, axis=0),
        "camera_K": np.concatenate(all_camera_K, axis=0),
        "camera_R": np.concatenate(all_camera_R, axis=0),
        "camera_t": np.concatenate(all_camera_t, axis=0),
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)
    return output_path
