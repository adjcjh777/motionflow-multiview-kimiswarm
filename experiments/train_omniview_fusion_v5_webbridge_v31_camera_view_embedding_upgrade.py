"""v31 camera-view-embedding upgrade: training entry wrapper.

This script runs the standard ``train_omniview_fusion_v5_webbridge_multi.py``
loop, but swaps in ``CameraConditionedViewEmbeddingV31`` as the implementation
of ``CameraConditionedViewEmbedding``.  The monkey-patch is performed before the
OmniMultiViewFusionV5 model is imported, so the new embedding is instantiated
when ``--use_camera_view_embedding`` is passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Patch the camera embedding class *before* OmniMultiViewFusionV5 is imported.
from motionflow_mv.fusion import camera_conditioned_view_embedding
from motionflow_mv.fusion.camera_view_embedding_v31 import (
    CameraConditionedViewEmbeddingV31,
)

camera_conditioned_view_embedding.CameraConditionedViewEmbedding = (
    CameraConditionedViewEmbeddingV31
)

# Now import and run the standard v5 trainer.
from experiments import train_omniview_fusion_v5_webbridge_multi as _train  # noqa: E402

if __name__ == "__main__":
    _train.main()
