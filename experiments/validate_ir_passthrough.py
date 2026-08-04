"""Validate that GVHMR <-> HumanMotionIR <-> GVHMR is a zero-loss passthrough.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/validate_ir_passthrough.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.ir.gvhmr_adapter import gvhmr_pt_to_ir, ir_to_gvhmr_pt


def make_synthetic_gvhmr_pt(path: Path, T: int = 120, betas_dim: int = 10):
    """Create a minimal hmr4d_results.pt for testing."""
    smpl_params = {
        "body_pose": torch.randn(T, 63) * 0.1,
        "global_orient": torch.randn(T, 3) * 0.1,
        "transl": torch.randn(T, 3) * 0.1,
        "betas": torch.randn(T, betas_dim) * 0.01,
    }
    pred = {
        "smpl_params_global": smpl_params,
        "smpl_params_incam": {k: v.clone() for k, v in smpl_params.items()},
        "K_fullimg": torch.eye(3).unsqueeze(0).repeat(T, 1, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pred, path)
    return pred


def main():
    pt_path = Path("outputs/synthetic_hmr4d_results.pt")
    print("Creating synthetic GVHMR output...")
    original = make_synthetic_gvhmr_pt(pt_path)

    print("Converting to HumanMotionIR...")
    ir = gvhmr_pt_to_ir(pt_path, sequence_id="synthetic_test")

    print(f"IR schema: {ir.schema_version}, frames: {len(ir.timestamps)}, fps: {ir.fps}")

    print("Converting back to GVHMR-compatible dict...")
    reconstructed = ir_to_gvhmr_pt(ir)

    print("Checking pose key round-trip consistency...")
    max_errors = {}
    for key in original["smpl_params_global"]:
        orig = original["smpl_params_global"][key]
        recon = reconstructed["smpl_params_global"][key]
        err = (orig - recon).abs().max().item()
        max_errors[key] = err
        print(f"  {key}: max error {err:.6f}")

    overall_max = max(max_errors.values())
    if overall_max < 1e-5:
        print(f"\n[PASSED] Passthrough validation passed (max error {overall_max:.2e}).")
    else:
        print(f"\n[FAILED] Passthrough validation failed (max error {overall_max:.2e}).")
        sys.exit(1)


if __name__ == "__main__":
    main()
