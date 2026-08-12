#!/usr/bin/env python
"""Patch VoxelPose lib/core/function.py to support the H36M true-GT adapter."""

from __future__ import absolute_import, division, print_function

import sys

TRAIN_H36M_BODY = """pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(meta=meta, targets_3d=targets_3d[0],
                                                                              input_heatmaps=input_heatmap)"""

VAL_H36M_BODY = """pred, heatmaps, grid_centers, _, _, _ = model(meta=meta, targets_3d=targets_3d[0],
                                                          input_heatmaps=input_heatmap)"""

METRIC_H36M = """elif 'h36m' in config.DATASET.TEST_DATASET:
        mpjpe = loader.dataset.evaluate(preds)
        msg = 'MPJPE: {mpjpe:.2f}mm'.format(mpjpe=mpjpe)
        print(msg)
        metric = -mpjpe
"""


def _indent(level, text):
    """Indent a multi-line block by ``level`` spaces."""
    return ''.join(' ' * level + line for line in text.splitlines(True)) + '\n'


def _make_h36m_branch(indent_level, body):
    branch = ' ' * indent_level + "elif 'h36m' in config.DATASET.TEST_DATASET:\n"
    branch += ' ' * (indent_level + 4) + body + '\n'
    return branch


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
            # Match the indentation of the surrounding elif block so the
            # inserted branch is syntactically valid regardless of upstream
            # whitespace changes.
            indent = len(line) - len(line.lstrip())
            if current_func == 'train_3d':
                out_lines.insert(-1, _make_h36m_branch(indent, TRAIN_H36M_BODY))
            elif current_func == 'validate_3d':
                out_lines.insert(-1, _make_h36m_branch(indent, VAL_H36M_BODY))

        if "elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:" in line and "actor_pcp" in line:
            indent = len(line) - len(line.lstrip())
            out_lines.insert(-1, _indent(indent, METRIC_H36M))

    with open(path, 'w') as f:
        f.writelines(out_lines)
    print('Patched {} for H36M true-GT'.format(path))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'lib/core/function.py')
