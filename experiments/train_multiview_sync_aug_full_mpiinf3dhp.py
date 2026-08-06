"""Full MPI-INF-3DHP training with view-synchronised temporal jitter.

This script is a thin wrapper around ``train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py``.
It monkey-patches the base ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``
class so that every training batch is passed through :class:`MultiViewSyncAugmentation`
before the original forward.  The patch is applied only to the model class used by the
``temporal`` model_type, leaving the rest of the training script unchanged.

Usage is identical to the base script; use ``--model_type temporal`` (the default)::

    python experiments/train_multiview_sync_aug_full_mpiinf3dhp.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz ... \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --model_type temporal --clip_len 13 --d 64 ...

Augmentation hyperparameters can be controlled through environment variables:
    MV_AUG_SUBCLIP_LEN, MV_AUG_TRANSLATION_STD, MV_AUG_ROTATION_STD,
    MV_AUG_SCALE_STD, MV_AUG_NOISE_STD, MV_AUG_VIEW_DROPOUT_RATE, MV_AUG_MIN_VIEWS.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.multiview_temporal_jitter import MultiViewSyncAugmentation
import experiments.train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp as _base


def _aug_config_from_env():
    return {
        "subclip_len": int(os.environ.get("MV_AUG_SUBCLIP_LEN", "5")),
        "translation_std": float(os.environ.get("MV_AUG_TRANSLATION_STD", "2.0")),
        "rotation_std_deg": float(os.environ.get("MV_AUG_ROTATION_STD", "1.0")),
        "scale_std": float(os.environ.get("MV_AUG_SCALE_STD", "0.02")),
        "noise_std": float(os.environ.get("MV_AUG_NOISE_STD", "0.5")),
        "view_dropout_rate": float(os.environ.get("MV_AUG_VIEW_DROPOUT_RATE", "0.1")),
        "min_views": int(os.environ.get("MV_AUG_MIN_VIEWS", "2")),
    }


def main():
    aug_config = _aug_config_from_env()

    base_cls = _base.RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint

    class MultiViewSyncAugPrincipalPointModel(base_cls):
        """Temporal cross-view residual PP model with view-synced augmentation."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.aug = MultiViewSyncAugmentation(**aug_config)

        def forward(self, x, *args, **kwargs):
            if self.training:
                x = self.aug(x)
            return super().forward(x, *args, **kwargs)

    MultiViewSyncAugPrincipalPointModel.__name__ = f"{base_cls.__name__}MultiViewSyncAug"
    _base.RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint = MultiViewSyncAugPrincipalPointModel

    # Call the original training entry point with the same CLI arguments.
    _base.main()


if __name__ == "__main__":
    main()
