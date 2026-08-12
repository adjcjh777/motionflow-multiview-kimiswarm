#!/usr/bin/env python
"""Patch VoxelPose lib/core/function.py to support the H36M true-GT adapter."""

from __future__ import absolute_import, division, print_function

import sys

TRAIN_H36M = """        elif 'h36m' in config.DATASET.TEST_DATASET:
            pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(meta=meta, targets_3d=targets_3d[0],
                                                                              input_heatmaps=input_heatmap)
"""

VAL_H36M = """        elif 'h36m' in config.DATASET.TEST_DATASET:
            pred, heatmaps, grid_centers, _, _, _ = model(meta=meta, targets_3d=targets_3d[0],
                                                          input_heatmaps=input_heatmap)
"""

METRIC_H36M = """    elif 'h36m' in config.DATASET.TEST_DATASET:
        mpjpe = loader.dataset.evaluate(preds)
        msg = 'MPJPE: {mpjpe:.2f}mm'.format(mpjpe=mpjpe)
        print(msg)
        metric = -mpjpe
"""


def main(path='lib/core/function.py'):
    with open(path, 'r') as f:
        content = f.read()

    if "elif 'h36m' in config.DATASET.TEST_DATASET:" in content:
        print('{} already patched for H36M true-GT'.format(path))
        return

    lines = content.splitlines(keepends=True)

    current_func = None
    out_lines = []
    for line in lines:
        if line.startswith('def '):
            name = line.split('(')[0].replace('def ', '').strip()
            current_func = name

        out_lines.append(line)

        if "elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:" in line:
            if current_func == 'train_3d':
                out_lines.insert(-1, TRAIN_H36M)
            elif current_func == 'validate_3d':
                out_lines.insert(-1, VAL_H36M)

        if "elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:" in line and "actor_pcp" in line:
            out_lines.insert(-1, METRIC_H36M)

    with open(path, 'w') as f:
        f.writelines(out_lines)
    print('Patched {} for H36M true-GT'.format(path))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'lib/core/function.py')
