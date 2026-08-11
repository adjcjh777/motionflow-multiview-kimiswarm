# A800 Docker MotionFlow Service Status

> **Snapshot time (A800 host):** 2026-08-11T02:46:09+00:00  
> **Checked from:** local WSL workspace (`D:/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm`)  
> **Method:** read-only SSH inspection (`ssh a800-D`)

## 1. Docker Daemon

```text
● docker.service - Docker Application Container Engine
     Loaded: loaded (/lib/systemd/system/docker.service; enabled; vendor preset: enabled)
     Active: active (running) since Mon 2026-07-27 02:17:34 UTC; 2 weeks 1 day ago
   Main PID: 3084 (dockerd)
      Tasks: 2376
     Memory: 53.0G
        CPU: 49min 29s
```

- **Status:** `active (running)`
- **Enabled:** yes (`enabled`)
- **Uptime:** ~2 weeks

## 2. MotionFlow Container

```text
CONTAINER ID   NAME        IMAGE                                    STATUS        PORTS
a4c3398033d6   motionflow  elf3-trainer:20260729-auto-equal-sample  Up 12 days    0.0.0.0:8000->8000/tcp, :::8000->8000/tcp, 8080/tcp
```

| Field | Value |
|-------|-------|
| Container ID | `a4c3398033d6` |
| Name | `motionflow` |
| Image | `elf3-trainer:20260729-auto-equal-sample` |
| Status | `running` (Up 12 days) |
| Created | `2026-07-29T08:59:58Z` |
| Started | `2026-07-29T08:59:59Z` |
| Restart policy | `unless-stopped` |
| Host port | `8000` → container `8000` |
| Entrypoint | `/entrypoint-single-port.sh` |
| OOM Killed | No |
| Dead | No |

### Container runtime stats

```text
CONTAINER ID   NAME         CPU %   MEM USAGE / LIMIT   MEM %   NET I/O          BLOCK I/O      PIDS
a4c3398033d6   motionflow   0.09%   310MiB / 1008GiB    0.03%   800MB / 12.7GB   98MB / 691MB   123
```

### Inside-container processes (top)

```text
PID   USER      %CPU  COMMAND
  1   root       0.0  bash /entrypoint-single-port.sh
 10   root      21.7  /usr/bin/python3 /usr/local/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8001
 11   root       0.0  nginx: master process nginx -g daemon off;
```

- **Uvicorn/FastAPI backend:** `127.0.0.1:8001`
- **nginx reverse proxy:** exposed on host port `8000`

### Volume mounts

| Host path | Container path |
|-----------|----------------|
| `/mnt/nvme1n1p1/datas/zhangzy/motionflow-runtime/jobs` | `/var/elf3/jobs` |
| `/mnt/nvme1n1p1/datas/zhangzy/motionflow-runtime/data` | `/var/elf3/data` |

### GPU request

- `DeviceRequests`: all available GPUs (`Count: -1`, capability `gpu`)
- Container runtime: `runc`

## 3. Service Endpoints

| Endpoint | Response | Notes |
|----------|----------|-------|
| `http://localhost:8000/` | MotionFlow web UI HTML | Frontend loads correctly |
| `http://localhost:8000/api/jobs` | JSON list of 13 jobs | See job summary below |
| `http://localhost:8000/api/health` | Returns web UI HTML | No dedicated JSON health endpoint observed |

## 4. Job Summary (from `/api/jobs`)

| Status | Count |
|--------|-------|
| Running | 0 |
| Queued | 0 |
| Succeeded | 8 |
| Cancelled | 4 |
| Failed | 1 |
| **Total** | **13** |

### Recent job detail

| Job ID | Video | Status | Progress |
|--------|-------|--------|----------|
| `238f79baa0a94af18d61923347ef2b61` | `CR7_SIU_1-1.mp4` | cancelled | 3% |
| `60e1b85ec93644bc83833ca5a8270af8` | `CR7_SIU_2-1.mp4` | succeeded | 100% |
| `6016db01622a48fdb4c17d116231333f` | `CR7_SIU_1-1.mp4` | cancelled | 97% |
| `d75301491e1641c4b6e9a97408c43c14` | `jianshen_2.gif` | cancelled | 76% |
| `ad9f0ef02e8441728153037ccd28a3bb` | `jianshen_1.gif` | cancelled | 74% |
| `7b40444b97e74937a199b2caea6be671` | `taiji.mp4` | failed | 3% |
| `51f2ce77696e49959125e9636132420e` | `nuanfeng.mp4` | succeeded | 100% |
| `5bbf192953f04d32806f691ecbc8a15a` | `dl.mp4` | succeeded | 100% |
| `b5d1dde805f04ab395c5a156d7681549` | `mechannical_dance.mp4` | succeeded | 100% |
| `6218e3caf97046f7ba3a4efbbf9243cd` | `child dance.mp4` | succeeded | 100% |
| `0758b2e8fb3c4933bb5298c190a8cee9` | `beatutiful_mythos.mp4` | succeeded | 100% |
| `7c9c63a69d3343128003ad0237708127` | `sistar.mp4` | succeeded | 100% |
| `bcbd703de20d45b095c96dfcad48b570` | `taiji_song.mp4` | succeeded | 100% |

> **Observation:** No active training is running inside the MotionFlow Docker container at this time. GPUs 0-3 still hold allocations from other workloads (likely the separate tmux/nohup training sessions tracked elsewhere).

## 5. GPU Status on A800 Host

```text
Tue Aug 11 02:45:39 2026
NVIDIA-SMI 580.173.02    Driver Version: 580.173.02    CUDA Version: 13.0
8 x NVIDIA A800-SXM4-80GB
```

| GPU | Memory usage | Notes |
|-----|--------------|-------|
| 0 | 76135 MiB / 81920 MiB | In use (~76 GB) |
| 1 | 76135 MiB / 81920 MiB | In use (~76 GB) |
| 2 | 76135 MiB / 81920 MiB | In use (~76 GB) |
| 3 | 76135 MiB / 81920 MiB | In use (~76 GB) |
| 4 | 0 MiB / 81920 MiB | Free |
| 5 | 0 MiB / 81920 MiB | Free |
| 6 | 0 MiB / 81920 MiB | Free |
| 7 | 12375 MiB / 81920 MiB | Partial use (~12 GB) |

All GPUs at 0% volatile GPU-Util, low power (55-68 W).

## 6. Other Notable Containers on the Same Host

The Docker daemon hosts several other services; only the motionflow container is relevant to this project:

| Container | Image | Status | Ports |
|-----------|-------|--------|-------|
| `motionflow` | `elf3-trainer:20260729-auto-equal-sample` | **Up 12 days** | `8000:8000` |
| `xiaoqibot-multigpu` | `asimov-mjlab:latest` | Up 3 days | `8080:8080` |
| `coder_wuzy` | `coder-server-ai` | Up 13 days | various |
| `coder_hermes` | `coder-server-ai` | Up 7 days | various |
| `stt3`, `tts2`, `stt` | various | Up ~2 weeks | `15576`, `15577`, `7806` |
| `mysql` | `mysql:5.6` | Up ~2 weeks | `3306:3306` |

Many historical/previous `motionflow-pre-*` containers are present in `Exited` state and do not affect the current service.

## 7. Read-Only Audit Notes

- No files or containers on `a800-D` were modified.
- Only inspection commands were run: `systemctl status`, `docker ps`, `docker inspect`, `docker stats`, `docker logs`, `nvidia-smi`, and `curl` to local service endpoints.

## 8. Summary

- **Docker daemon:** healthy and active.
- **MotionFlow Docker service:** running normally (`motionflow`, image `elf3-trainer:20260729-auto-equal-sample`, Up 12 days, port 8000).
- **API:** responsive; web UI and `/api/jobs` return data.
- **Workload:** no active/queued MotionFlow jobs; 8 succeeded, 4 cancelled, 1 failed job in history.
- **GPU:** 8x A800 present; GPUs 0-3 and 7 are partially utilized by other workloads, GPUs 4-6 are free.
