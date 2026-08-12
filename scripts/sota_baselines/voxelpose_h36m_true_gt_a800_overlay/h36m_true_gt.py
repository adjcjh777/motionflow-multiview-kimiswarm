# ------------------------------------------------------------------------------
# VoxelPose dataset adapter for the corrected H36M true-GT protocol.
#
# This file is copied into the upstream VoxelPose repo as
# lib/dataset/h36m_true_gt.py by the A800 run script.
#
# It loads the annotations produced by
# scripts/sota_baselines/convert_to_voxelpose_format.py from
# data/h36m_true_gt/ and presents them in the format VoxelPose expects.
# Because the project stores 2D/3D keypoints (not raw H36M video frames),
# the adapter feeds ground-truth 2D points as the per-view input heatmaps
# and uses a single blank placeholder image for the image pipeline.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import pickle
import logging

import numpy as np
import cv2

from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)


def _convert_camera(cam_in):
    """Convert our (K, R, t) camera to VoxelPose (R, T, fx/fy/cx/cy, k, p)."""
    K = cam_in['K']
    R = cam_in['R']
    t = cam_in['t']
    # Camera centre in world coordinates. Our convention: x = K (R X + t),
    # so the centre is C = -R^T t. VoxelPose uses x_cam = R (X - T),
    # therefore T = C.
    T = -R.T.dot(t)
    return {
        'R': R,
        'T': T,
        'fx': float(K[0, 0]),
        'fy': float(K[1, 1]),
        'cx': float(K[0, 2]),
        'cy': float(K[1, 2]),
        'k': np.zeros((3, 1), dtype=np.float32),
        'p': np.zeros((2, 1), dtype=np.float32),
    }


class H36MTrueGT(JointsDataset):
    """H36M true-GT single-person dataset for VoxelPose."""

    def __init__(self, cfg, image_set, is_train, transform=None):
        self.pixel_std = 200.0
        self.joints_def = {}
        super().__init__(cfg, image_set, is_train, transform)

        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.limbs = []

        # Number of views is fixed by the H36M multiview capture.
        self.num_views = cfg.DATASET.CAMERA_NUM
        self.cam_list = list(range(self.num_views))

        anno_path = osp.join(self.dataset_root, 'h36m_true_gt_annotations.pkl')
        if not osp.exists(anno_path):
            raise FileNotFoundError(
                'Run the data conversion step first: {}'.format(anno_path))

        with open(anno_path, 'rb') as f:
            annotations = pickle.load(f)

        split_map = {
            'train': 'train',
            'training': 'train',
            'validation': 'val',
            'val': 'val',
            'test': 'val',
        }
        if image_set not in split_map:
            raise ValueError('Unknown image_set: {}'.format(image_set))
        split_key = split_map[image_set]

        self.blank_image_path = osp.join(self.dataset_root, 'blank_h36m.png')
        self._ensure_blank_image()

        self.db = self._get_db(annotations[split_key])
        self.db_size = len(self.db)
        logger.info('=> H36MTrueGT {}: {} frames x {} views = {} records'.format(
            image_set, self.db_size // self.num_views, self.num_views, self.db_size))

    def _ensure_blank_image(self):
        if osp.exists(self.blank_image_path):
            return
        # H36M standard image resolution is 1000 x 1002. 1024 x 1024 is a
        # safe square canvas that contains all projected keypoints.
        img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        cv2.imwrite(self.blank_image_path, img)
        logger.info('=> created blank placeholder image: {}'.format(
            self.blank_image_path))

    def _get_db(self, sequences):
        db = []
        for seq in sequences:
            points_2d = seq['points_2d']      # (F, V, J, 2)
            confidences = seq['confidences']    # (F, V, J)
            joints_3d = seq['joints_3d']        # (F, J, 3)
            cameras = seq['cameras']            # list of V dicts

            num_frames = points_2d.shape[0]
            for f in range(num_frames):
                for v in range(self.num_views):
                    cam = _convert_camera(cameras[v])
                    pose3d = joints_3d[f]              # (J, 3)
                    pose2d = points_2d[f, v]           # (J, 2)
                    vis = confidences[f, v]            # (J,)

                    # VoxelPose expects binary visibility flags per coordinate.
                    vis_2d = np.stack([vis, vis], axis=1)  # (J, 2)
                    vis_3d = np.stack([vis, vis, vis], axis=1)  # (J, 3)

                    # GT 2D used as the "detected" input heatmap.
                    pred2d = np.concatenate([pose2d, vis[:, None]], axis=1)  # (J, 3)

                    db.append({
                        'image': self.blank_image_path,
                        'joints_3d': [pose3d],
                        'joints_3d_vis': [vis_3d],
                        'joints_2d': [pose2d],
                        'joints_2d_vis': [vis_2d],
                        'camera': cam,
                        'pred_pose2d': [pred2d],
                    })
        return db

    def __getitem__(self, idx):
        input, target_heatmap, target_weight, target_3d, meta, input_heatmap = [], [], [], [], [], []
        for k in range(self.num_views):
            i, th, tw, t3, m, ih = super().__getitem__(self.num_views * idx + k)
            input.append(i)
            target_heatmap.append(th)
            target_weight.append(tw)
            input_heatmap.append(ih)
            target_3d.append(t3)
            meta.append(m)
        return input, target_heatmap, target_weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views

    def evaluate(self, preds, recall_threshold=500.0):
        """Compute MPJPE (mm) over all frames."""
        gt_num = self.db_size // self.num_views
        if len(preds) != gt_num:
            raise ValueError('Prediction count mismatch: {} vs {}'.format(
                len(preds), gt_num))

        mpjpes = []
        for i in range(gt_num):
            db_rec = self.db[i * self.num_views]
            gt = db_rec['joints_3d'][0]  # (J, 3)

            pred = preds[i]  # (num_cand, J, 5)
            pred = pred[pred[:, 0, 3] >= 0]
            if len(pred) == 0:
                continue
            # Single-person data: take the first candidate.
            p = pred[0, :, :3]
            mpjpes.append(np.mean(np.linalg.norm(p - gt, axis=1)))

        mpjpe = float(np.mean(mpjpes)) * 1000.0 if mpjpes else float('inf')
        return mpjpe
