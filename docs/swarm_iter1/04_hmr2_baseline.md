# HMR2.0 / 4DHumans Baseline

## TL;DR

4DHumans (ICCV 2023) uses HMR2.0, a fully-transformerized Human Mesh Recovery network. It regresses SMPL parameters from a single image using a ViT-H/16 backbone and a cross-attention transformer decoder. The video pipeline feeds per-frame HMR2.0 outputs into PHALP for 4D tracking.

## 关键结论

### 1. 网络结构

- **Backbone**: ViT-H/16 (`hmr2/models/backbones/vit.py`)
  - Input ~256×192, patch=16, embed_dim=1280, depth=32, heads=16
  - Outputs image token sequence as decoder context.
- **Head**: `SMPLTransformerDecoderHead` (`hmr2/models/heads/smpl_head.py`)
  - 6-layer, 8-head cross-attention transformer decoder (dim=1024)
  - Iterative Error Feedback (IEF): refines pose/shape/cam residuals
  - Joint rotation default 6D → rotmat
- **Outputs**: via `forward_step` in `hmr2/models/hmr2.py`
  - `pred_smpl_params`: global_orient, body_pose, betas
  - `pred_cam`: weak-perspective camera (scale, tx, ty)
  - `pred_cam_t`: 3D camera translation
  - `pred_vertices`: SMPL vertices
  - `pred_keypoints_3d` / `pred_keypoints_2d`: 3D/2D joints

### 2. 使用方式

Environment setup:

```bash
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans
pip install -e ".[all]"
# Download SMPL neutral model to ./data/
```

Single-image inference:

```bash
python demo.py \
  --img_folder example_data/images \
  --out_folder demo_out \
  --batch_size=48 --side_view --save_mesh --full_frame
```

Video tracking:

```bash
pip install git+https://github.com/brjathu/PHALP.git
python track.py video.source="example_data/videos/gymnasts.mp4"
```

Python snippet:

```python
from hmr2.models import load_hmr2
import torch

model, cfg = load_hmr2()
model.eval()
# batch = {"img": tensor}
with torch.no_grad():
    out = model(batch)
    verts = out["pred_vertices"]
    smpl_params = out["pred_smpl_params"]
```

### 3. MotionFlow 关联

- HMR2.0 is single-image and single-view; no built-in multi-view fusion.
- Output format (SMPL params + camera + vertices/keypoints) is a natural initialization for multi-view consistency or motion flow stages.

## 参考

- Repo: https://github.com/shubham-goel/4D-Humans
- Paper: https://arxiv.org/abs/2305.20091
- Project page: https://shubham-goel.github.io/4dhumans/
