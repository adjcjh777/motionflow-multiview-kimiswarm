"""CPU smoke test for the multi-model 6-axis robustness matrix.

Creates a tiny synthetic MPI-INF-3DHP-style .npz, saves trivial checkpoints for a
couple of registered models, and runs
``experiments/eval_robustness_matrix_pp_mpiinf3dhp.py`` end-to-end on CPU.
The whole test should finish in well under two minutes.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.eval_full_metrics import MODEL_CLASSES


def _make_synthetic_npz(path: Path, n_frames: int = 25, n_views: int = 2, n_joints: int = 17) -> None:
    """Write a minimal multiview .npz in the format expected by the matrix."""
    rng = np.random.default_rng(42)
    points_2d = rng.normal(size=(n_frames, n_views, n_joints, 2)).astype(np.float32) * 0.1
    confidences = rng.uniform(0.5, 1.0, size=(n_frames, n_views, n_joints)).astype(np.float32)
    joints_3d = rng.normal(size=(n_frames, n_joints, 3)).astype(np.float32) * 0.1

    camera_K = np.zeros((n_views, 3, 3), dtype=np.float32)
    camera_R = np.zeros((n_views, 3, 3), dtype=np.float32)
    camera_t = np.zeros((n_views, 3), dtype=np.float32)
    for v in range(n_views):
        camera_K[v] = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
        camera_R[v] = np.eye(3, dtype=np.float32)

    np.savez(
        path,
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=joints_3d,
        camera_K=camera_K,
        camera_R=camera_R,
        camera_t=camera_t,
    )


def _make_checkpoint(model_name: str, ckpt_path: Path) -> None:
    model = MODEL_CLASSES[model_name](
        j=17,
        d=32,
        n_views=2,
        n_st_layers=1,
        residual_hidden=64,
    )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)


def _run_matrix(model_name: str, tmp_path: Path) -> None:
    data_path = tmp_path / "smoke.npz"
    ckpt_path = tmp_path / f"{model_name}.pth"
    out_json = tmp_path / f"robustness_{model_name}.json"
    out_md = tmp_path / f"robustness_{model_name}.md"

    _make_synthetic_npz(data_path)
    _make_checkpoint(model_name, ckpt_path)

    cmd = [
        sys.executable,
        "experiments/eval_robustness_matrix_pp_mpiinf3dhp.py",
        "--model",
        model_name,
        "--checkpoint",
        str(ckpt_path),
        "--dataset",
        str(data_path),
        "--out_json",
        str(out_json),
        "--out_md",
        str(out_md),
        "--clip_len",
        "5",
        "--batch_size",
        "2",
        "--val_stride",
        "5",
        "--d",
        "32",
        "--n_st_layers",
        "1",
        "--residual_hidden",
        "64",
        "--device",
        "cpu",
    ]
    if model_name == "hierarchical_view_temporal_joint_pp":
        cmd += ["--n_view_groups", "2", "--n_view_layers", "1", "--n_temporal_layers", "1", "--n_joint_graph_layers", "1"]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(Path(__file__).parent.parent))
    if result.returncode != 0:
        raise AssertionError(f"Robustness matrix failed for {model_name}:\n{result.stdout}\n{result.stderr}")

    assert out_json.exists(), f"JSON output missing for {model_name}"
    assert out_md.exists(), f"Markdown output missing for {model_name}"
    content = out_md.read_text()
    assert "clean" in content
    assert "rot_1.0_deg" in content


if __name__ == "__main__":
    for name in ["crossview_residual_pp", "bayesian_tri_pp"]:
        with tempfile.TemporaryDirectory() as td:
            print(f"Running smoke test for {name}...")
            _run_matrix(name, Path(td))
            print(f"  {name}: OK")
