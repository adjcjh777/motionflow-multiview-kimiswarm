# PyTorch GPU/CPU Environment Check

**Date checked:** 2026-08-09

**Environment:** `.venv` (WSL path: `/mnt/d/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`)

**Python executable:** `.venv/bin/python`

## Verification command

```bash
cd /mnt/d/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm
.venv/bin/python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN available:", torch.backends.cudnn.is_available())
print("cuDNN version:", torch.backends.cudnn.version())
PY
```

## Result

| Item | Value |
|------|-------|
| torch | `2.4.0+cu121` |
| CUDA available | `True` |
| CUDA version | `12.1` |
| cuDNN available | `True` |
| cuDNN version | `90100` |

Relevant pip packages:

```text
torch                    2.4.0+cu121
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12        9.1.0.70
... (other nvidia-*-cu12 packages)
```

## Conclusion

The project virtual environment is already using a **GPU build of PyTorch** (`torch 2.4.0+cu121`) with CUDA 12.1 and cuDNN available. No `pip install` switch is required.
