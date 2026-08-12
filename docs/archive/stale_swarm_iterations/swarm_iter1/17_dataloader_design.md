# Data Loader Design for Synchronized Multiview Frames

## TL;DR
Use one canonical frame index as the single source of truth; store per-view frames in a  layout and a central manifest ( or ) that maps . Sampling is deterministic by frame index; time sync relies on pre-aligned frame extraction, with fallback to nearest-frame lookup when frame rates differ slightly. For the MotionFlow multi-view extension, keep the loader PyTorch-native, avoid decoding video on the fly, and cache pre-extracted frames as JPG/PNG for fast random access.

## Key Conclusions

### 1. File structure: one manifest, not many folder conventions

A multiview clip with  cameras and  frames should be organized as:



Why this layout:
- Keeps each view independently replaceable (different frame rates, resolutions).
- Lets the loader pick a view subset at runtime for augmentation/ablation.
- Manifest is the only place that knows frame-to-file mapping; view directories stay dumb.

### 2. Sampling: frame-index driven, not timestamp driven

For synchronized multiview training, sample by **frame index**.



Rules:
- Batch collation should stack  across views, not across time.
- Shuffle only the time index; never shuffle views independently within a frame.
- Subsampling (e.g. every 4th frame) is applied to  before dataset construction.

### 3. Time synchronization

Preferred options in order:

1. **Hardware/genlock sync**: already aligned; use frame index directly.
2. **Pre-processed alignment**: extract frames with  using a shared timestamp file; all views share the same frame index.
   
3. **Nearest-frame fallback**: if view frame counts differ, look up the frame whose timestamp is closest to the canonical timestamp. Store  in the manifest.



### 4. Integration with MotionFlow

Since MotionFlow expects a monocular video or frame sequence, the multiview loader should return a list of single-view tensors. The per-view baseline wrapper is applied independently, and the fusion step consumes the per-view 2D/3D outputs. Avoid decoding the original videos inside the training loop; it is slow and non-deterministic.

## Reference Links / Code

- PyTorch  docs: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- OpenCV image I/O: https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html
-  frame extraction guide: https://ffmpeg.org/ffmpeg.html#image2
- Related project notes:
  -  §3 (preprocessing / cache layout)
  -  §4 (core module APIs)

