"""Convert a multi-view .npz dataset to metric (meters).

Human3.6M data is stored in millimeters (camera_t norm ~5e3, joints_3d ~1e3--2e4),
Shelf/Campus in centimeters (camera_t norm ~3e2--1.2e3).  This script lets you
pick the source unit and writes a new .npz with camera_t and joints_3d in meters.
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--scale", type=float, required=True,
                        help="Factor by which camera_t and joints_3d are divided to get meters (e.g. 1000 for mm, 100 for cm).")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    data = dict(np.load(args.input))
    data["camera_t"] = data["camera_t"] / args.scale
    data["joints_3d"] = data["joints_3d"] / args.scale
    # points_2d stays in pixels; K, R unchanged.

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)
    print(f"Saved {args.output} (scale=1/{args.scale})")


if __name__ == "__main__":
    main()
