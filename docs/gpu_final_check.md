# GPU Final Check

## Timestamp

- Check performed: 2026-08-11 (local system time)
- Command: `nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit,fan.speed,pstate --format=csv`

## GPU Status

| Metric | Value |
|--------|-------|
| GPU Index | 0 |
| GPU Name | NVIDIA GeForce RTX 4090 |
| Temperature | 48 °C |
| Utilization | 2 % (idle) |
| Memory used | 1377 MiB |
| Memory total | 24564 MiB |
| Power draw | 20.90 W |
| Power limit | 450.00 W |
| Fan speed | 45 % |
| Performance state | P8 (idle/low power) |

## Compute Processes

- Only one Python process is listed under compute apps: `D:\anaconda3\python.exe` (PID 1428) with `[N/A]` GPU memory usage.
- No active CUDA training/eval process detected.
- GPU utilization is at 2 %, indicating the v57 H36M true-GT medium run has completed and the GPU is free.

## Conclusion

The RTX 4090 is **free** and ready for the next single-GPU task. GPU memory usage is at baseline (1.4 GB out of 24 GB), and no model training/evaluation process is consuming GPU resources.
