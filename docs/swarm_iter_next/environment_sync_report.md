# Environment Sync Report (A800 vs Local)

**Generated:** 2026-08-08 05:15:18 UTC
**Script:** `scripts/check_env_sync.py`

## 1. Python / CUDA / GPU

### Local

- Python: `Python 3.13.9`
- PyTorch: `2.7.1+cu118`
- CUDA (torch): `11.8`
- GPUs:
- `NVIDIA GeForce RTX 4090, 595.79, 24564 MiB`

## 2. requirements.txt Compliance

- `numpy` >=1.24.0 - local `2.4.4` OK
- `scipy` >=1.11.0 - local `1.16.3` OK
- `pyyaml` >=6.0 - local `6.0.3` OK
- `pytest` >=7.0.0 - local `8.4.2` OK
- `torch` ==2.4.0+cu121 - local `2.7.1+cu118` **does not satisfy**

## 3. Critical Package Version Comparison

| Package | Local | Remote | Match? |
|---------|-------|--------|--------|
| `torch` | `2.7.1+cu118` | `missing` | - |
| `torchvision` | `0.22.1+cu118` | `missing` | - |
| `numpy` | `2.4.4` | `missing` | - |
| `scipy` | `1.16.3` | `missing` | - |
| `pyyaml` | `6.0.3` | `missing` | - |
| `pytest` | `8.4.2` | `missing` | - |

## 4. v25 Module Import Check

- Local: OK - OK (D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm\motionflow_mv\fusion\multiview_geometry_fusion_v25.py)

## 5. Recommendations

- Resolve the errors above before launching a full A800 run.
- Run the v25 small smoke config before scheduling the full A800 run.
- Re-run this checker after any conda/pip update on either host.