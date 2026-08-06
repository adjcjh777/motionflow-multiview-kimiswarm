"""Convert a GVHMR hmr4d_results.pt artifact into HumanMotionIR.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/convert_gvhmr_to_ir.py \
        --input data/gvhmr_demo/hmr4d_results.pt \
        --output outputs/humanmotion_ir_gvhmr_demo.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.ir.gvhmr_adapter import gvhmr_pt_to_ir


def main():
    parser = argparse.ArgumentParser(description="Convert GVHMR output to HumanMotionIR.")
    parser.add_argument("--input", type=str, default="data/gvhmr_demo/hmr4d_results.pt")
    parser.add_argument("--output", type=str, default="outputs/humanmotion_ir_gvhmr_demo.pt")
    args = parser.parse_args()

    ir = gvhmr_pt_to_ir(args.input, sequence_id=Path(args.input).stem)
    print(f"Converted {args.input} -> {args.output}")
    print(f"  sequence_id: {ir.sequence_id}")
    print(f"  frames: {len(ir.timestamps)}")
    print(f"  human_model: {ir.human_model}")
    for key, arr in ir.pose.items():
        print(f"  pose[{key!r}]: {arr.shape}")

    import torch
    torch.save(ir, args.output)
    print(f"Saved HumanMotionIR to {args.output}")


if __name__ == "__main__":
    main()
