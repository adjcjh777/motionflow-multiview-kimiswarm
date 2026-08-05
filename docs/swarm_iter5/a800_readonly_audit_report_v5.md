# A800-D Read-Only Audit Report (Swarm Iteration 5)

**Date:**
2026-08-05 20:35:41 UTC

**Scope:**
`/mnt/nvme0n1/zhangzy/projects` on host `a800-D`

**Constraint:** Read-only operations only.

**Audit script:**
[`scripts/audit_a800_readonly_v5.py`](../scripts/audit_a800_readonly_v5.py)


## 1. Host & Storage

### Top-level directory listing

```text
total 180
drwxrwxr-x 26 zhangzy zhangzy  4096 Jul 28 08:57 .
drwxr-x--- 36 zhangzy zhangzy  4096 Aug  4 05:50 ..
drwxr-xr-x  3 zhangzy zhangzy  4096 Jun 22 01:58 .agents
-rw-rw-r--  1 zhangzy zhangzy  1043 Jun 22 01:58 AGENTS.md
drwxrwxr-x  2 zhangzy zhangzy  4096 Jun 22 01:58 .aris
-rw-rw-r--  1 zhangzy zhangzy 13548 Jun 22 02:12 aris-elf3-gmr-safety-handoff-2026-06-22.md
-rw-rw-r--  1 zhangzy zhangzy 19233 Jun 22 03:42 aris-elf3-serial-runbook-and-interface-audit-2026-06-22.md
-rw-rw-r--  1 zhangzy zhangzy 20043 Jun 22 03:41 ARIS_ELF3_运行手册.md
drwxrwxr-x 15 zhangzy zhangzy  4096 Jun 22 03:13 Auto-claude-code-research-in-sleep
drwxrwxr-x  3 zhangzy zhangzy  4096 Jun 22 01:56 .claude
-rw-rw-r--  1 zhangzy zhangzy   634 Jun 22 01:56 CLAUDE.md
drwxr-xr-x  2 zhangzy zhangzy  4096 Jun 22 01:55 .codex
drwxrwxr-x 11 zhangzy zhangzy  4096 Jul 20 10:25 elf3-video-to-policy
drwxr-xr-x  2 zhangzy zhangzy  4096 Jun 22 01:55 .git
drwxrwxr-x 10 zhangzy zhangzy  4096 Jun 22 06:10 GMR
drwxrwxr-x 10 zhangzy zhangzy  4096 Jun 23 01:45 gmr-motionlab
drwxrwxr-x 11 zhangzy zhangzy  4096 May 27 06:54 GVHMR
drwxrwxr-x  3 zhangzy zhangzy  4096 Jun 24 02:20 melodyai
drwxrwxr-x 14 zhangzy zhangzy  4096 May 28 02:27 mjlab
drwxrwxr-x 15 zhangzy zhangzy  4096 Jun 23 07:09 mjlab_elf3
drwxrwxr-x 11 zhangzy zhangzy  4096 Jul  9 03:54 mjlab-elf3_beyongmimic
drwxrwxr-x 10 zhangzy zhangzy  4096 Jul 29 09:41 motionflow-6df139c-build
drwx------  7 zhangzy zhangzy  4096 Jul 23 09:39 motionflow-f49d93e-build-KieqEr
drwxrwxr-x 10 zhangzy zhangzy  4096 Jul 28 09:00 motionflow-research-multiview-easymocap-robot-profiles
drwxrwxr-x  3 zhangzy zhangzy  4096 Jul 10 01:18 .pytest_cache
drwxrwxr-x 13 zhangzy zhangzy  4096 May  9 06:36 pytorch3d
drwxrwxr-x 11 zhangzy zhangzy  4096 May 21 06:56 smplx
drwxrwxr-x  6 zhangzy zhangzy  4096 Jul 10 08:30 summercamp
drwxrwxr-x  6 zhangzy zhangzy  4096 Jun  4 06:20 urdf2mjcf
-rw-rw-r--  1 zhangzy zhangzy 10201 Jun 22 06:34 video_to_robot_motion_pipeline.md
drwxrwxr-x  5 zhangzy zhangzy  4096 May 15 07:52 whole_body_tracking
drwxrwxr-x  9 zhangzy zhangzy  4096 Jun 25 08:49 xiaoqibot
```

### Top-level directory sizes

```text
4.0K	/mnt/nvme0n1/zhangzy/projects/AGENTS.md
16K	/mnt/nvme0n1/zhangzy/projects/aris-elf3-gmr-safety-handoff-2026-06-22.md
20K	/mnt/nvme0n1/zhangzy/projects/aris-elf3-serial-runbook-and-interface-audit-2026-06-22.md
20K	/mnt/nvme0n1/zhangzy/projects/ARIS_ELF3_运行手册.md
62M	/mnt/nvme0n1/zhangzy/projects/Auto-claude-code-research-in-sleep
4.0K	/mnt/nvme0n1/zhangzy/projects/CLAUDE.md
855M	/mnt/nvme0n1/zhangzy/projects/elf3-video-to-policy
2.3G	/mnt/nvme0n1/zhangzy/projects/GMR
1.8G	/mnt/nvme0n1/zhangzy/projects/gmr-motionlab
7.1G	/mnt/nvme0n1/zhangzy/projects/GVHMR
342M	/mnt/nvme0n1/zhangzy/projects/melodyai
1.7G	/mnt/nvme0n1/zhangzy/projects/mjlab
3.5G	/mnt/nvme0n1/zhangzy/projects/mjlab_elf3
216M	/mnt/nvme0n1/zhangzy/projects/mjlab-elf3_beyongmimic
299M	/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build
11M	/mnt/nvme0n1/zhangzy/projects/motionflow-f49d93e-build-KieqEr
157M	/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles
244M	/mnt/nvme0n1/zhangzy/projects/pytorch3d
6.1M	/mnt/nvme0n1/zhangzy/projects/smplx
672M	/mnt/nvme0n1/zhangzy/projects/summercamp
305M	/mnt/nvme0n1/zhangzy/projects/urdf2mjcf
12K	/mnt/nvme0n1/zhangzy/projects/video_to_robot_motion_pipeline.md
227M	/mnt/nvme0n1/zhangzy/projects/whole_body_tracking
8.0G	/mnt/nvme0n1/zhangzy/projects/xiaoqibot
```

### Disk usage for project/data volumes

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1  3.5T  3.2T   92G  98% /mnt/nvme0n1p1
/dev/nvme1n1p1  3.5T  2.7T  644G  81% /mnt/nvme1n1p1
```


## 2. Docker Containers & Images

### Docker containers

```text
NAMES                                                   IMAGE                                          STATUS                     PORTS
motionflow                                              elf3-trainer:20260729-auto-equal-sample        Up 7 days                  0.0.0.0:8000->8000/tcp, :::8000->8000/tcp, 8080/tcp
motionflow-pre-auto-equal-sample-20260729T085958Z       elf3-trainer:20260729-equal-sample-hotfix      Exited (143) 7 days ago    
motionflow-pre-equal-sample-hotfix-20260729T083057Z     elf3-trainer:20260729-multigpu-tuning          Exited (143) 7 days ago    
xiaoqibot-preview-latest                                asimov-mjlab:latest                            Up 8 days                  0.0.0.0:8081->8080/tcp, :::8081->8080/tcp
xiaoqibot-dashboard                                     xiaoqibot-dashboard:20260728                   Up 8 days                  0.0.0.0:8050->8050/tcp, :::8050->8050/tcp, 8080/tcp
xiaoqibot-multigpu                                      asimov-mjlab:latest                            Up 8 days                  0.0.0.0:8080->8080/tcp, :::8080->8080/tcp
motionflow-pre-multigpu-tuning-20260729T075059Z         elf3-trainer:20260728-live-log-eta-v2          Exited (143) 7 days ago    
motionflow-pre-live-log-eta-v2-20260728T063526Z         elf3-trainer:20260728-live-log-eta             Exited (143) 8 days ago    
motionflow-pre-live-log-eta-20260728T063041Z            elf3-trainer:20260728-global-env-split         Exited (143) 8 days ago    
keen_rosalind                                           artifixer:cuda12                               Up 9 days                  6006/tcp, 8888/tcp
motionflow-pre-global-env-split-20260728T055705Z        elf3-trainer:20260724-preview-runtime-hotfix   Exited (143) 8 days ago    
motionflow-pre-preview-runtime-hotfix-20260724T175420   elf3-trainer:20260724-smart-gpu-scheduler      Exited (143) 12 days ago   
asimov-train-container                                  asimov-mjlab:latest                            Exited (0) 13 days ago     
coder_wuzy                                              coder-server-ai                                Up 8 days                  5550-5570/tcp, 0.0.0.0:18841-18848->18841-18848/tcp, :::18841-18848->18841-18848/tcp, 0.0.0.0:18840->8080/tcp, :::18840->8080/tcp, 0.0.0.0:18849->33533/tcp, :::18849->33533/tcp
motionflow-pre-smart-scheduler-20260724T033247Z         elf3-trainer:20260720-a800-single-port         Exited (143) 12 days ago   
motionflow-pre-single-port-20260720                     elf3-trainer:20260717-concurrent-queue         Exited (0) 2 weeks ago     
motionflow-gpu7-stopped-20260717                        elf3-trainer:20260717-concurrent-queue         Exited (137) 2 weeks ago   
motionflow-pre-concurrent-20260717                      elf3-trainer:20260717-system-settings          Exited (0) 2 weeks ago     
elf3-trainer-b300-webui                                 elf3-trainer:b300                              Exited (0) 2 weeks ago     
elf3-trainer-b300-webui-pre-seek-final-20260715         46288aef4eb0                                   Exited (0) 3 weeks ago     
elf3-trainer-b300-webui-pre-timeline-20260715           1291662d15ac                                   Exited (0) 3 weeks ago     
elf3-trainer-b300-webui-pre-one-env-20260715            d7b7e7a4ed86                                   Exited (0) 3 weeks ago     
elf3-trainer-b300-webui-pre-preview-fix-20260715        2380390ce760                                   Exited (0) 3 weeks ago     
elf3-trainer-b300-webui-pre-final-20260715              a7116e37cdf4                                   Exited (0) 3 weeks ago     
elf3-trainer-b300-webui-backup-20260715                 0c2cb4d89c6c                                   Exited (137) 3 weeks ago   
elf3-trainer                                            elf3-trainer:latest                            Exited (0) 3 weeks ago     
funny_ishizaka                                          d7ecb1a21d4c                                   Exited (1) 3 weeks ago     
trusting_pike                                           d7ecb1a21d4c                                   Exited (1) 3 weeks ago     
strange_newton                                          ghcr.io/nerfstudio-project/nerfstudio:1.1.5    Exited (127) 3 weeks ago   
coder_freemocap                                         coder_hermes                                   Exited (255) 9 days ago    5550-5570/tcp, 11110-11199/tcp, 0.0.0.0:12210-12299->12210-12299/tcp, :::12210-12299->12210-12299/tcp, 0.0.0.0:12200->8080/tcp, :::12200->8080/tcp, 0.0.0.0:12201->33533/tcp, :::12201->33533/tcp
jixunying-2                                             nvcr.io/nvidia/pytorch:23.05-py3               Exited (137) 9 days ago    
jixunying-1                                             nvcr.io/nvidia/pytorch:23.05-py3               Exited (137) 9 days ago    
coder_fyt_gpu                                           coder_hermes                                   Exited (255) 9 days ago    5550-5570/tcp, 11110-11199/tcp, 0.0.0.0:12110-12199->12110-12199/tcp, :::12110-12199->12110-12199/tcp, 0.0.0.0:12100->8080/tcp, :::12100->8080/tcp, 0.0.0.0:12101->33533/tcp, :::12101->33533/tcp
coder_113                                               coder_hermes                                   Exited (255) 9 days ago    5550-5570/tcp, 11110-11199/tcp, 0.0.0.0:11310-11399->11310-11399/tcp, :::11310-11399->11310-11399/tcp, 0.0.0.0:11300->8080/tcp, :::11300->8080/tcp, 0.0.0.0:11301->33533/tcp, :::11301->33533/tcp
coder_112                                               coder_hermes                                   Exited (255) 9 days ago    5550-5570/tcp, 11110-11199/tcp, 0.0.0.0:11210-11299->11210-11299/tcp, :::11210-11299->11210-11299/tcp, 0.0.0.0:11200->8080/tcp, :::11200->8080/tcp, 0.0.0.0:11201->33533/tcp, :::11201->33533/tcp
coder_hermes                                            coder-server-ai                                Up 2 days                  5550-5570/tcp, 0.0.0.0:11110-11199->11110-11199/tcp, :::11110-11199->11110-11199/tcp, 0.0.0.0:11100->8080/tcp, :::11100->8080/tcp, 0.0.0.0:11101->33533/tcp, :::11101->33533/tcp
stt3                                                    5adce7ec4e6f                                   Up 9 days                  0.0.0.0:15576->8080/tcp, :::15576->8080/tcp
tts2                                                    c4c07b9daa29                                   Up 9 days                  0.0.0.0:15577->18827/tcp, :::15577->18827/tcp
stt                                                     stt2                                           Up 9 days                  0.0.0.0:7806->8080/tcp, :::7806->8080/tcp
coder_fyt                                               coder-server-ai                                Exited (255) 9 days ago    5550-5570/tcp, 0.0.0.0:18831-18838->18831-18838/tcp, :::18831-18838->18831-18838/tcp, 0.0.0.0:18830->8080/tcp, :::18830->8080/tcp, 0.0.0.0:18839->33533/tcp, :::18839->33533/tcp
coder_shf                                               coder-server-ai                                Exited (255) 9 days ago    5550-5570/tcp, 0.0.0.0:18821-18828->18821-18828/tcp, :::18821-18828->18821-18828/tcp, 0.0.0.0:18820->8080/tcp, :::18820->8080/tcp, 0.0.0.0:18829->33533/tcp, :::18829->33533/tcp
mysql                                                   mysql:5.6                                      Up 9 days                  0.0.0.0:3306->3306/tcp, :::3306->3306/tcp
```

### Docker images (top 40)

```text
REPOSITORY                                                           TAG                                   SIZE
elf3-trainer                                                         20260729-auto-equal-sample-portable   34GB
elf3-trainer                                                         motionflow-latest-portable            34GB
elf3-trainer                                                         20260729-auto-equal-sample            35GB
elf3-trainer                                                         motionflow-latest                     35GB
elf3-trainer                                                         20260729-equal-sample-hotfix          35GB
elf3-trainer                                                         20260729-multigpu-tuning              35GB
xiaoqibot-dashboard                                                  20260728                              29.7GB
elf3-trainer                                                         20260728-live-log-eta-v2              35GB
elf3-trainer                                                         20260728-live-log-eta                 35GB
elf3-trainer                                                         20260728-global-env-split             35GB
elf3-trainer                                                         20260724-preview-runtime-hotfix       35GB
<none>                                                               <none>                                35GB
<none>                                                               <none>                                35GB
elf3-trainer                                                         20260724-smart-gpu-scheduler          35GB
<none>                                                               <none>                                35GB
elf3-trainer                                                         20260723-f49d93e-a800-single-port     34.9GB
asimov-mjlab                                                         latest                                29.7GB
<none>                                                               <none>                                29.6GB
artifixer                                                            cuda12                                37.7GB
elf3-trainer                                                         20260720-a800-single-port             34.9GB
elf3-trainer                                                         a800-single-port                      34.9GB
elf3-trainer                                                         20260717-concurrent-queue             34.1GB
elf3-trainer                                                         20260717-system-settings              34.1GB
elf3-trainer                                                         20260717-npz-sim-hotfix               34.1GB
elf3-trainer                                                         20260715-b300                         34.1GB
elf3-trainer                                                         b300                                  34.1GB
<none>                                                               <none>                                34.1GB
<none>                                                               <none>                                34.1GB
<none>                                                               <none>                                34.1GB
<none>                                                               <none>                                34.1GB
<none>                                                               <none>                                34.1GB
<none>                                                               <none>                                34.1GB
elf3-trainer                                                         20260715-b300-base                    34.1GB
<none>                                                               <none>                                34.1GB
<none>                                                               <none>                                34.1GB
<none>                                                               <none>                                34.1GB
nerfstudio-sai                                                       1.1.5                                 11.7GB
spectacularai                                                        latest                                11.7GB
elf3-trainer                                                         latest                                15.5GB
```

### Running motionflow container config

```json
{
  "Entrypoint": [
    "/entrypoint-single-port.sh"
  ],
  "Cmd": null,
  "WorkingDir": "/app",
  "Env": [
    "ELF3_GPU_BUSY_THRESHOLD_PCT=100",
    "PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "NVARCH=x86_64",
    "NVIDIA_REQUIRE_CUDA=cuda>=12.8 brand=unknown,driver>=470,driver<471 brand=grid,driver>=470,driver<471 brand=tesla,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=vapps,driver>=470,driver<471 brand=vpc,driver>=470,driver<471 brand=vcs,driver>=470,driver<471 brand=vws,driver>=470,driver<471 brand=cloudgaming,driver>=470,driver<471 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=560,driver<561 brand=grid,driver>=560,driver<561 brand=tesla,driver>=560,driver<561 brand=nvidia,driver>=560,driver<561 brand=quadro,driver>=560,driver<561 brand=quadrortx,driver>=560,driver<561 brand=nvidiartx,driver>=560,driver<561 brand=vapps,driver>=560,driver<561 brand=vpc,driver>=560,driver<561 brand=vcs,driver>=560,driver<561 brand=vws,driver>=560,driver<561 brand=cloudgaming,driver>=560,driver<561 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566",
    "NV_CUDA_CUDART_VERSION=12.8.90-1",
    "CUDA_VERSION=12.8.1",
    "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
    "NVIDIA_VISIBLE_DEVICES=all",
    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
    "NV_CUDA_LIB_VERSION=12.8.1-1",
    "NV_NVTX_VERSION=12.8.90-1",
    "NV_LIBNPP_VERSION=12.3.3.100-1",
    "NV_LIBNPP_PACKAGE=libnpp-12-8=12.3.3.100-1",
    "NV_LIBCUSPARSE_VERSION=12.5.8.93-1",
    "NV_LIBCUBLAS_PACKAGE_NAME=libcublas-12-8",
    "NV_LIBCUBLAS_VERSION=12.8.4.1-1",
    "NV_LIBCUBLAS_PACKAGE=libcublas-12-8=12.8.4.1-1",
    "NV_LIBNCCL_PACKAGE_NAME=libnccl2",
    "NV_LIBNCCL_PACKAGE_VERSION=2.25.1-1",
    "NCCL_VERSION=2.25.1-1",
    "NV_LIBNCCL_PACKAGE=libnccl2=2.25.1-1+cuda12.8",
    "NVIDIA_PRODUCT_NAME=CUDA",
    "NV_CUDA_CUDART_DEV_VERSION=12.8.90-1",
    "NV_NVML_DEV_VERSION=12.8.90-1",
    "NV_LIBCUSPARSE_DEV_VERSION=12.5.8.93-1",
    "NV_LIBNPP_DEV_VERSION=12.3.3.100-1",
    "NV_LIBNPP_DEV_PACKAGE=libnpp-dev-12-8=12.3.3.100-1",
    "NV_LIBCUBLAS_DEV_VERSION=12.8.4.1-1",
    "NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-12-8",
    "NV_LIBCUBLAS_DEV_PACKAGE=libcublas-dev-12-8=12.8.4.1-1",
    "NV_CUDA_NSIGHT_COMPUTE_VERSION=12.8.1-1",
    "NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-12-8=12.8.1-1",
    "NV_NVPROF_VERSION=12.8.90-1",
    "NV_NVPROF_DEV_PACKAGE=cuda-nvprof-12-8=12.8.90-1",
    "NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev",
    "NV_LIBNCCL_DEV_PACKAGE_VERSION=2.25.1-1",
    "NV_LIBNCCL_DEV_PACKAGE=libnccl-dev=2.25.1-1+cuda12.8",
    "LIBRARY_PATH=/usr/local/cuda/lib64/stubs",
    "PYTORCH_VERSION=2.9.1",
    "DEBIAN_FRONTEND=noninteractive",
    "PYTHONUNBUFFERED=1",
    "MUJOCO_GL=egl",
    "PYOPENGL_PLATFORM=egl",
    "PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/",
    "PIP_TRUSTED_HOST=mirrors.aliyun.com",
    "PIP_CONFIG_FILE=/etc/pip.conf",
    "TORCH_CUDA_ARCH_LIST=8.0;9.0;10.0+PTX",
    "ELF3_JOBS_DIR=/var/elf3/jobs",
    "ELF3_DB_PATH=/var/elf3/data/elf3.db",
    "ELF3_GVHMR_ROOT=/opt/GVHMR",
    "ELF3_GMR_ROOT=/opt/GMR",
    "ELF3_MJLAB_ROOT=/opt/mjlab-elf3_beyongmimic",
    "ELF3_WEIGHTS_ROOT=/opt/GVHMR/inputs/checkpoints",
    "ELF3_FRONTEND_DIST=/app/frontend/dist",
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
    "ELF3_APP_VERSION=2026.07.29-auto-equal-sample",
    "ELF3_VISER_PORT=8080",
    "ELF3_GIF_PLAYBACK_SPEED=0.5",
    "ELF3_GIF_OUTPUT_FPS=30",
    "ELF3_VISER_PUBLIC_URL=/viser/"
  ]
}
```


## 3. Checkpoints / Weights

### Checkpoints / weights (.pth, .pt, .ckpt)

```text
===== motionflow-research-multiview-easymocap-robot-profiles =====

  sizes (MB):
  count=0 total_bytes=0 total_MB=0.00
===== motionflow-6df139c-build =====

  sizes (MB):
  count=0 total_bytes=0 total_MB=0.00
===== motionflow-f49d93e-build-KieqEr =====

  sizes (MB):
  count=0 total_bytes=0 total_MB=0.00
===== GVHMR =====
/mnt/nvme0n1/zhangzy/projects/GVHMR/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt	2709494041
/mnt/nvme0n1/zhangzy/projects/GVHMR/inputs/checkpoints/yolo/yolov8x.pt	136867539
/mnt/nvme0n1/zhangzy/projects/GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt	163508011
/mnt/nvme0n1/zhangzy/projects/GVHMR/inputs/checkpoints/dpvo/dpvo.pth	14167743
/mnt/nvme0n1/zhangzy/projects/GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth	2549075546
/mnt/nvme0n1/zhangzy/projects/GVHMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt	208935
/mnt/nvme0n1/zhangzy/projects/GVHMR/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt	662187
/mnt/nvme0n1/zhangzy/projects/GVHMR/hmr4d/utils/body_model/smpl_3dpw14_J_regressor_sparse.pt	2791
/mnt/nvme0n1/zhangzy/projects/GVHMR/hmr4d/utils/body_model/coco_aug_dict.pth	1759
/mnt/nvme0n1/zhangzy/projects/GVHMR/hmr4d/utils/body_model/smplx_verts437.pt	4203
/mnt/nvme0n1/zhangzy/projects/GVHMR/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt	469227
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/freemocap_rotated/hmr4d_results.betas10.pt	3133303
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/freemocap_rotated/hmr4d_results.pt	3167615
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/xiaoxaunf/hmr4d_results.betas10.pt	2204279
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/xiaoxaunf/hmr4d_results.pt	2227583
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/galbot/hmr4d_results.betas10.pt	3063863
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/galbot/hmr4d_results.pt	3097407
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/block_long/hmr4d_results.pt	1101375
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/beatutiful_mythos/hmr4d_results.pt	4244095
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/unitree_classic_movie_dancing/hmr4d_results.betas10.pt	5216255
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/unitree_classic_movie_dancing/hmr4d_results.pt	5273855
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/7motion/hmr4d_results.pt	5835967
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/squat/hmr4d_results.pt	662975
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/quanji/hmr4d_results.betas10.pt	1618231
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/quanji/hmr4d_results.pt	1635263
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/block/hmr4d_results.pt	667327
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/taiji/hmr4d_results.betas10.pt	3710327
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/taiji/hmr4d_results.pt	3750847
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/love_waltz/hmr4d_results.betas10.pt	3007031
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/love_waltz/hmr4d_results.pt	3039423
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/wushu/hmr4d_results.betas10.pt	3436471
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/wushu/hmr4d_results.pt	3474239
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/From-standing-to-crawling-forward/hmr4d_results.betas10.pt	2880951
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/From-standing-to-crawling-forward/hmr4d_results.pt	2912575
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/1783905861112_VID_442/hmr4d_results.betas10.pt	1865463
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/1783905861112_VID_442/hmr4d_results.pt	1885823
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/pufu/hmr4d_results.betas10.pt	4829623
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/pufu/hmr4d_results.pt	4882751
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/walk/hmr4d_results.pt	1422463
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/xiaoxuanfeng/hmr4d_results.betas10.pt	2203263

  sizes (MB):
  count=46 total_bytes=5669332932 total_MB=5406.70
===== GMR =====

  sizes (MB):
  count=0 total_bytes=0 total_MB=0.00
===== gmr-motionlab =====

  sizes (MB):
  count=0 total_bytes=0 total_MB=0.00
```


## 4. Datasets / Demo Artifacts

### Datasets / demo artifacts

```text
===== GVHMR/outputs/demo =====
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/freemocap_rotated/hmr4d_results.betas10.pt	3133303
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/freemocap_rotated/hmr4d_results.pt	3167615
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/xiaoxaunf/hmr4d_results.betas10.pt	2204279
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/xiaoxaunf/hmr4d_results.pt	2227583
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/galbot/hmr4d_results.betas10.pt	3063863
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/galbot/hmr4d_results.pt	3097407
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/block_long/hmr4d_results.pt	1101375
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/beatutiful_mythos/hmr4d_results.pt	4244095
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/unitree_classic_movie_dancing/hmr4d_results.betas10.pt	5216255
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/unitree_classic_movie_dancing/hmr4d_results.pt	5273855
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/7motion/hmr4d_results.pkl	897723
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/7motion/hmr4d_results.pt	5835967
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/squat/hmr4d_results.pt	662975
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/quanji/hmr4d_results.betas10.pt	1618231
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/quanji/hmr4d_results.pt	1635263
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/block/block.pkl	102131
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/block/hmr4d_results.pt	667327
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/taiji/hmr4d_results.betas10.pt	3710327
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/taiji/hmr4d_results.pt	3750847
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/love_waltz/hmr4d_results.betas10.pt	3007031
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/love_waltz/hmr4d_results.pt	3039423
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/wushu/hmr4d_results.betas10.pt	3436471
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/wushu/hmr4d_results.pt	3474239
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/From-standing-to-crawling-forward/hmr4d_results.betas10.pt	2880951
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/From-standing-to-crawling-forward/hmr4d_results.pt	2912575
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/1783905861112_VID_442/hmr4d_results.betas10.pt	1865463
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/1783905861112_VID_442/hmr4d_results.pt	1885823
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/pufu/hmr4d_results.betas10.pt	4829623
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/pufu/hmr4d_results.pt	4882751
/mnt/nvme0n1/zhangzy/projects/GVHMR/outputs/demo/walk/hmr4d_results.pt	1422463
===== GMR/save =====
/mnt/nvme0n1/zhangzy/projects/GMR/save/freemocap_rotated_elf3.pkl	207387
/mnt/nvme0n1/zhangzy/projects/GMR/save/beatutiful_mythos_elf3.pkl	281115
/mnt/nvme0n1/zhangzy/projects/GMR/save/data_elf3.npz	928258
/mnt/nvme0n1/zhangzy/projects/GMR/save/xiaoxuanfeng_elf3.pkl	145755
/mnt/nvme0n1/zhangzy/projects/GMR/save/g7ms_elf3.pkl	386819
/mnt/nvme0n1/zhangzy/projects/GMR/save/7ms_elf3.pkl	386819
/mnt/nvme0n1/zhangzy/projects/GMR/save/block_long_elf3.pkl	72599
/mnt/nvme0n1/zhangzy/projects/GMR/save/From-standing-to-crawling-forward_elf3.pkl	190683
/mnt/nvme0n1/zhangzy/projects/GMR/save/data_g1.npz	928258
/mnt/nvme0n1/zhangzy/projects/GMR/save/squat_elf3.pkl	43502
/mnt/nvme0n1/zhangzy/projects/GMR/save/squat2_elf3.pkl	51278
/mnt/nvme0n1/zhangzy/projects/GMR/save/data_g1.pkl	89882
/mnt/nvme0n1/zhangzy/projects/GMR/save/wushu_elf3.pkl	227547
/mnt/nvme0n1/zhangzy/projects/GMR/save/unitree_classic_movie_dancing_elf3.pkl	345636
/mnt/nvme0n1/zhangzy/projects/GMR/save/squat_newelf3.pkl	43502
/mnt/nvme0n1/zhangzy/projects/GMR/save/love_waltz_elf3.pkl	199035
/mnt/nvme0n1/zhangzy/projects/GMR/save/walk.pkl	93914
/mnt/nvme0n1/zhangzy/projects/GMR/save/7motion_elf3.pkl	386820
/mnt/nvme0n1/zhangzy/projects/GMR/save/dance1_subject1_BXI.npz	11782402
/mnt/nvme0n1/zhangzy/projects/GMR/save/block_elf3.pkl	43790
/mnt/nvme0n1/zhangzy/projects/GMR/save/quanji_elf3.pkl	106875
/mnt/nvme0n1/zhangzy/projects/GMR/save/xiaoxaunf_elf3.pkl	145755
/mnt/nvme0n1/zhangzy/projects/GMR/save/pufu_elf3.pkl	319995
/mnt/nvme0n1/zhangzy/projects/GMR/save/data_elf3.pkl	89882
/mnt/nvme0n1/zhangzy/projects/GMR/save/galbot_elf3.pkl	202779
/mnt/nvme0n1/zhangzy/projects/GMR/save/1783905861112_VID_442_elf3.pkl	123291
/mnt/nvme0n1/zhangzy/projects/GMR/save/stand.pkl	45518
/mnt/nvme0n1/zhangzy/projects/GMR/save/taiji_elf3.pkl	245691
===== motionflow-research-multiview-easymocap-robot-profiles/vendor/mjlab-elf3_beyongmimic/npz =====
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/vendor/mjlab-elf3_beyongmimic/npz/taiji_elf3_smooth.npz	3753760
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/vendor/mjlab-elf3_beyongmimic/npz/dance1_subject1_BXI.npz	11782402
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/vendor/mjlab-elf3_beyongmimic/npz/dance2_final_slow.npz	2461468
```


## 5. Run Configs / Scripts / Dockerfiles

### Run configs / scripts / Dockerfiles

```text
===== motionflow-research-multiview-easymocap-robot-profiles =====
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/scripts/export_20260716_hotfix.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/scripts/export_20260717_npz_sim_hotfix.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/scripts/export_20260717_release.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/scripts/build_b300_hotfix_image.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/scripts/export_20260717_concurrent_queue.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/scripts/build_b300_image.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/scripts/export_b300_image.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/scripts/export_20260724_smart_gpu_scheduler.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/scripts/setup.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/scripts/setup_weights.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/scripts/export_20260717_system_settings.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/vendor/GMR/inference.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/vendor/GVHMR/pyrightconfig.json
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/docker/Dockerfile.hotfix
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/docker/Dockerfile.cuda-graph-hotfix
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/docker/entrypoint.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/docker/Dockerfile.system-settings
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/docker/Dockerfile.concurrent-queue.dockerignore
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/docker/Dockerfile.system-settings.dockerignore
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/docker/Dockerfile.smart-gpu-scheduler
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/docker/Dockerfile
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/docker/Dockerfile.concurrent-queue
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/frontend/tsconfig.json
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/frontend/package.json
/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/frontend/package-lock.json

===== motionflow-6df139c-build =====
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/scripts/export_20260716_hotfix.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/scripts/export_20260717_npz_sim_hotfix.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/scripts/export_20260717_release.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/scripts/build_b300_hotfix_image.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/scripts/export_20260717_concurrent_queue.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/scripts/build_b300_image.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/scripts/export_b300_image.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/scripts/export_20260724_smart_gpu_scheduler.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/scripts/setup.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/scripts/setup_weights.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/scripts/export_20260717_system_settings.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/vendor/GMR/inference.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/vendor/GVHMR/pyrightconfig.json
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/docker/Dockerfile.hotfix
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/docker/Dockerfile.cuda-graph-hotfix
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/docker/entrypoint.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/docker/Dockerfile.system-settings
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/docker/Dockerfile.concurrent-queue.dockerignore
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/docker/Dockerfile.system-settings.dockerignore
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/docker/Dockerfile.smart-gpu-scheduler
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/docker/Dockerfile
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/docker/Dockerfile.concurrent-queue
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/frontend/tsconfig.json
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/frontend/package.json
/mnt/nvme0n1/zhangzy/projects/motionflow-6df139c-build/frontend/node_modules/.package-lock.json

===== motionflow-f49d93e-build-KieqEr =====
/mnt/nvme0n1/zhangzy/projects/motionflow-f49d93e-build-KieqEr/docker/Dockerfile.hotfix
/mnt/nvme0n1/zhangzy/projects/motionflow-f49d93e-build-KieqEr/docker/Dockerfile.cuda-graph-hotfix
/mnt/nvme0n1/zhangzy/projects/motionflow-f49d93e-build-KieqEr/docker/entrypoint.sh
/mnt/nvme0n1/zhangzy/projects/motionflow-f49d93e-build-KieqEr/docker/Dockerfile.system-settings
/mnt/nvme0n1/zhangzy/projects/motionflow-f49d93e-build-KieqEr/docker/Dockerfile.concurrent-queue.dockerignore
/mnt/nvme0n1/zhangzy/projects/motionflow-f49d93e-build-KieqEr/docker/Dockerfile.system-settings.dockerignore
/mnt/nvme0n1/zhangzy/projects/motionflow-f49d93e-build-KieqEr/docker/Dockerfile
/mnt/nvme0n1/zhangzy/projects/motionflow-f49d93e-build-KieqEr/docker/Dockerfile.concurrent-queue
/mnt/nvme0n1/zhangzy/projects/motionflow-f49d93e-build-KieqEr/frontend/tsconfig.json
/mnt/nvme0n1/zhangzy/projects/motionflow-f49d93e-build-KieqEr/frontend/package.json
/mnt/nvme0n1/zhangzy/projects/motionflow-f49d93e-build-KieqEr/frontend/package-lock.json

===== GVHMR =====
/mnt/nvme0n1/zhangzy/projects/GVHMR/tools/video_to_elf3_pkl.sh
/mnt/nvme0n1/zhangzy/projects/GVHMR/hmr4d/configs/train.yaml
/mnt/nvme0n1/zhangzy/projects/GVHMR/hmr4d/configs/siga24_release.yaml
/mnt/nvme0n1/zhangzy/projects/GVHMR/hmr4d/configs/demo.yaml
/mnt/nvme0n1/zhangzy/projects/GVHMR/pyrightconfig.json

===== GMR =====
/mnt/nvme0n1/zhangzy/projects/GMR/assets/berkeley_humanoid_lite/config.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_n1.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_r1pro.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/bvh_xsens_to_h1_2.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_kuavo.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_bhl.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_gr3.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_tienkung.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_pm01.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_toddy.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_t1_29dof.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/bvh_nokov_to_g1.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_toddy.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/bvh_to_talos.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_h1.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_adam.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_openloong.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_elf3.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_n1.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/xrobot_to_g1.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_hi.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/xsens_mvn_to_g1.json
/mnt/nvme0n1/zhangzy/projects/GMR/general_motion_retargeting/ik_configs/fbx_to_g1.json

===== gmr-motionlab =====
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/scripts/run_web_editor.sh
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/run.sh
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/web_pkl_editor/frontend/package.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/web_pkl_editor/frontend/package-lock.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/assets/berkeley_humanoid_lite/config.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/smplx_to_n1.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/smplx_to_r1pro.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/bvh_xsens_to_h1_2.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/smplx_to_kuavo.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/smplx_to_bhl.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/smplx_to_gr3.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/smplx_to_tienkung.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/bvh_lafan1_to_pm01.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/smplx_to_toddy.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/bvh_lafan1_to_t1_29dof.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/smplx_to_g1.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/bvh_nokov_to_g1.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/bvh_lafan1_to_toddy.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/bvh_to_talos.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/smplx_to_h1.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/smplx_to_adam.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/smplx_to_openloong.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/smplx_to_elf3.json
/mnt/nvme0n1/zhangzy/projects/gmr-motionlab/general_motion_retargeting/ik_configs/bvh_lafan1_to_n1.json
```


## 6. Implications for MotionFlow-MultiView


- **GVHMR weights are present** (`gvhmr_siga24_release.ckpt`, `hmr2`,
  `vitpose`, `yolo`, `dpvo`, SMPL/SMPL-X body models). These can be used
  read-only for local per-view feature extraction and for adapting the IR
  pipeline.
- **GVHMR demo outputs** (`outputs/demo/*/hmr4d_results.pt`) are available for
  single-view IR adapter validation, but they are not multi-view calibrated
  data.
- **Multi-view / mocap NPZ files** live in `vendor/mjlab-elf3_beyongmimic/npz`
  and `GMR/save/`. They are robot retargeting artifacts, not standard 3D-HPE
  benchmarks, so they cannot directly train the MPI-INF-3DHP temporal fusion
  baseline without conversion/label verification.
- **Standard 3D-HPE datasets** (Human3.6M, MPI-INF-3DHP canonical .npz,
  Shelf/Campus, CMU Panoptic, 3DPW, AMASS) are **not present** in the audited
  tree. The existing local `data/webbridge/mpi_inf_3dhp/` remains the primary
  training source for the RayAttentionFusionModelTemporal smoke tests.
- **Docker images** (`elf3-trainer:*`, ~34 GB) target the ELF3 video-to-policy
  pipeline (CUDA 12.8 / PyTorch 2.9.1). They are not currently wired to the
  multiview fusion repo, but the Dockerfile documents the dependency baseline
  (PyTorch cu128, GVHMR, GMR, mjlab-elf3_beyongmimic, SMPL-X).
- **No new dependencies** are required to run this audit; the script only uses
  the Python standard library and an existing `ssh` binary.
