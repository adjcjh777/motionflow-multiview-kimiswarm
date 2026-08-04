"""Demonstrate converting a real GVHMR hmr4d_results.pt to HumanMotionIR.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/demo_ir_from_gvhmr.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.ir.gvhmr_adapter import gvhmr_pt_to_ir


def main():
    pt_path = Path("data/gvhmr_demo/hmr4d_results.pt")
    if not pt_path.exists():
        print(f"Missing {pt_path}; copy a GVHMR demo output first.")
        return

    ir = gvhmr_pt_to_ir(pt_path, sequence_id="gvhmr_demo_block")
    print("IR schema:", ir.schema_version)
    print("Sequence id:", ir.sequence_id)
    print("Frames:", len(ir.timestamps))
    print("FPS:", ir.fps)
    print("Human model:", ir.human_model)
    print("Pose keys:", list(ir.pose.keys()))
    for key, arr in ir.pose.items():
        print(f"  {key}: shape {arr.shape}, dtype {arr.dtype}")
    print("Coordinate system:", ir.coordinate_system)


if __name__ == "__main__":
    main()
