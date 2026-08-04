# VIBE baseline: input/output, dependencies, how to run on video

## TL;DR

VIBE（CVPR 2020）从单目视频逐帧估计 SMPL 参数，作为 MotionFlow 多视角扩展的单视角基线。

## 关键结论

- 输入：任意 RGB 视频（mp4/YouTube URL），支持多人和 CPU/GPU。
- 输出：vibe_output.pkl（每帧 pose 72-d、betas 10-d、camera 3-d、joints/vertices），可转 FBX/glTF。
- 依赖：Python ≥3.7、PyTorch、OpenCV、SMPL；prepare_data.sh 下载模型。
- 运行：demo.py 一条命令完成视频推理。

## 参考链接 / 代码片段

- 仓库：https://github.com/mkocabas/VIBE
- 论文：https://arxiv.org/abs/1912.05656

~~~bash
git clone https://github.com/mkocabas/VIBE.git && cd VIBE
source scripts/install_conda.sh
source scripts/prepare_data.sh

python demo.py --vid_file sample_video.mp4 --output_folder output/ --display

import joblib
results = joblib.load("output/sample_video/vibe_output.pkl")
~~~
