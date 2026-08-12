# ------------------------------------------------------------------------------
# VoxelPose dataset package initialisation with H36M true-GT support.
# This file replaces lib/dataset/__init__.py in the upstream repo.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.panoptic import Panoptic as panoptic
from dataset.shelf_synthetic import ShelfSynthetic as shelf_synthetic
from dataset.campus_synthetic import CampusSynthetic as campus_synthetic
from dataset.shelf import Shelf as shelf
from dataset.campus import Campus as campus
from dataset.h36m_true_gt import H36MTrueGT as h36m_true_gt
