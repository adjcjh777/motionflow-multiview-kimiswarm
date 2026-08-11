# CVPR 2027 Submission Checklist — MotionFlow-MultiView

> **Project:** MotionFlow-MultiView (sparse-view / cross-domain robust multi-view 3D pose estimation)  
> **Target venue:** CVPR 2027  
> **Estimated deadline:** ~November 13, 2026 (Friday, mid-November; confirm on official CVPR 2027 website once announced)  
> **Last updated:** 2026-08-11  
> **Repository:** `D:\WSL_workspace\about_eassys\motionflow-multivie-kimiswarm`

---

## How to use this checklist

- Each item has a **deadline**, an **owner**, and a **status**.
- Status legend:  
  - ✅ Done  
  - 🟡 In progress / partial  
  - 🔴 Not started / blocked  
  - ⚪ Not applicable / optional

---

## 1. Pre-submission milestone timeline

| Week ending | Milestone | Hard deliverables | Exit criteria | Status |
|-------------|-----------|-------------------|---------------|--------|
| **2026-08-18** | Data foundation locked | Non-circular `.npz` for H36M, MPI, Shelf/Campus, AIST++; baseline scripts reproducible | `direct MJE > 0` on every benchmark; DLT baselines re-run | 🟡 H36M/Shelf/Campus/AIST++ ready; MPI detected-2D ready but alignment issue pending |
| **2026-08-25** | Baseline leaderboard complete | H36M/Shelf/Campus true-GT results for DLT, Iskakov, v25, v57, v80; AIST++ medium | Stable rank order; v57 result recorded; divergence diagnosis started | 🟡 v57 done (75.16 mm observed / 81.47 mm ckpt); v25 ablation 1 running; v25/v80 still diverge |
| **2026-09-01** | Divergence fix + smoke ablations | New regularization recipe; short-epoch smoke results show convergence | v25/v80/v57 no longer diverge on true GT | 🔴 Not started |
| **2026-09-15** | Full medium runs + cross-domain mix | Medium-length runs on H36M, AIST++, and mixed-dataset training | Per-domain val MPJPE reported | 🔴 Not started |
| **2026-10-01** | Robustness experiments complete | Sparse-view `MPJPE@k` curves, noise/occlusion/calibration perturbations | Figures + tables ready | 🔴 Not started |
| **2026-10-15** | SOTA comparisons complete | VoxelPose / MVPose or equivalent baselines on same split | Related-work table filled | 🔴 Not started |
| **2026-10-31** | Paper draft v1.0 complete | Full draft with real citations, corrected tables, new story | Internally reviewable |  Not started |
| **2026-11-07** | Camera-ready polish week | Rebuttal-ready figures, video, checksum, compliance checks | Zero known blockers | 🔴 Not started |
| **2026-11-13** | Submission day | CMT upload + PDF + supplementary | Submission confirmation email received | 🔴 Not started |

---

## 2. CVPR 2027 CMT / administrative requirements

| # | Item | Deadline | Owner | Status | Notes |
|---|------|----------|-------|--------|-------|
| 2.1 | Create / update CMT author accounts | 2026-11-06 | unassigned | 🔴 | All co-authors must have active CMT accounts |
| 2.2 | Finalize author list + order + affiliations | 2026-11-06 | unassigned | 🔴 | Lock before abstract submission |
| 2.3 | Confirm no dual-submission / arXiv conflicts | 2026-11-06 | unassigned | 🔴 | Review CVPR dual-submission policy |
| 2.4 | Submit title + abstract (if CMT opens early) | per CMT | unassigned | 🔴 | CVPR often allows abstract update until deadline |
| 2.5 | Verify PDF size / page limit (8 pages + refs) | 2026-11-13 | unassigned | 🔴 | CVPR 2027 will publish exact limits |
| 2.6 | Validate CMT PDF upload + supplementary upload | 2026-11-13 | unassigned | 🔴 | Do a dry-run 24 h before deadline |
| 2.7 | Check CMT submission ID / confirmation email | 2026-11-13 | unassigned | 🔴 | Keep screenshot/forward to team |

---

## 3. Paper content

### 3.1 Abstract & story

| # | Item | Deadline | Owner | Status | Notes |
|---|------|----------|-------|--------|-------|
| 3.1.1 | Lock paper pitch: sparse-view + cross-domain robustness | 2026-08-25 | unassigned |  | Draft in `docs/cvpr2027_status.md` |
| 3.1.2 | Write abstract (max 150 words) | 2026-10-31 | unassigned | 🔴 | Emphasize honest non-circular benchmarking |
| 3.1.3 | Write intro / motivation around circular-label problem | 2026-10-31 | unassigned | 🔴 | Cite `docs/results_true_gt_h36m.md` |
| 3.1.4 | Write contribution bullet list | 2026-10-31 | unassigned | 🔴 | Map to `docs/roadmap_cvpr2027.md` §3 |

### 3.2 Method

| # | Item | Deadline | Owner | Status | Notes |
|---|------|----------|-------|--------|-------|
| 3.2.1 | Final method section (v25/v57/v80 architecture) | 2026-10-31 | unassigned | 🔴 | Keep geometry-first fusion narrative |
| 3.2.2 | Sparse-view adaptive selector / variable-view module | 2026-09-30 | unassigned | 🔴 | `eval_variable_views.py` baseline exists |
| 3.2.3 | Calibration-robust fusion description | 2026-10-31 | unassigned | 🔴 | Link to `docs/roadmap_cvpr2027.md` §3 |

### 3.3 Experiments & results

| # | Item | Deadline | Owner | Status | Notes |
|---|------|----------|-------|--------|-------|
| 3.3.1 | H36M true-GT final leaderboard | 2026-08-25 | unassigned |  | v25 72.80 mm, v80 39.98 mm, v57 75.16 mm observed (81.47 mm ckpt); v25 ablation 1 running |
| 3.3.2 | MPI-INF-3DHP real-detected 2D leaderboard | 2026-09-15 | unassigned |  | Detected 2D ready, but DLT baseline ~326–400 mm due to camera/label alignment; blocked until fixed |
| 3.3.3 | Shelf/Campus detected leaderboard | 2026-08-25 | unassigned | ✅ | `docs/results_true_gt_shelf_campus.md` |
| 3.3.4 | AIST++ medium results | 2026-09-15 | agent-67 | 🟡 | Smoke done; full medium pending |
| 3.3.5 | Sparse-view `MPJPE@k` curves | 2026-10-01 | unassigned |  | Reuse `eval_variable_views.py` |
| 3.3.6 | Cross-domain transfer table | 2026-10-15 | unassigned |  | H36M ↔ MPI ↔ AIST++ ↔ Shelf/Campus |
| 3.3.7 | SOTA comparison (VoxelPose / MVPose / Iskakov) | 2026-10-15 | unassigned | 🟡 | Iskakov done; VoxelPose/MVPose not |
| 3.3.8 | Ablation study table | 2026-10-15 | unassigned | 🔴 | Regularization, augmentation, view count |

### 3.4 Citations & related work

| # | Item | Deadline | Owner | Status | Notes |
|---|------|----------|-------|--------|-------|
| 3.4.1 | Replace all placeholder / fabricated citations | 2026-08-25 | unassigned | 🟡 | `docs/paper_draft_icra_cvpr_2027.md` placeholders removed |
| 3.4.2 | Add real Iskakov ICCV 2019 citation & numbers | 2026-08-25 | unassigned | ✅ | `docs/results_iskakov_h36m_true_gt.md` |
| 3.4.3 | Add VoxelPose / MVPose / SOTA citations | 2026-10-15 | unassigned | 🔴 | Requires baseline runs |
| 3.4.4 | Check all references compile (BibTeX) | 2026-11-06 | unassigned | 🔴 | Use CVPR .bst |

### 3.5 Figures & tables

| # | Item | Deadline | Owner | Status | Notes |
|---|------|----------|-------|--------|-------|
| 3.5.1 | Main results table (true-GT H36M) | 2026-09-01 | unassigned | 🟡 | Pending v57 / divergence fix |
| 3.5.2 | Sparse-view robustness figure | 2026-10-01 | unassigned | 🔴 | `MPJPE@k` curves |
| 3.5.3 | Cross-domain transfer figure | 2026-10-15 | unassigned | 🔴 | Per-dataset bars |
| 3.5.4 | Qualitative failure / success examples | 2026-10-31 | unassigned | 🔴 | 2–3 high-res figures |
| 3.5.5 | Camera-ready figure resolution check (≥300 dpi) | 2026-11-06 | unassigned | 🔴 | |

---

## 4. Supplementary material

| # | Item | Deadline | Owner | Status | Notes |
|---|------|----------|-------|--------|-------|
| 4.1 | Supplementary PDF (proofs, extra tables, architecture details) | 2026-11-06 | unassigned | 🔴 | Optional but strongly recommended |
| 4.2 | Video / demo (≤5 min, per CVPR rules) | 2026-11-06 | unassigned | ⚪ | Optional; nice for visibility |
| 4.3 | Code / checkpoint release plan statement | 2026-11-06 | unassigned | 🔴 | State GitHub release timeline |
| 4.4 | Per-dataset evaluation scripts & manifests | 2026-10-15 | unassigned | 🟡 | `configs/splits/` growing |

---

## 5. Data & code reproducibility

| # | Item | Deadline | Owner | Status | Notes |
|---|------|----------|-------|--------|-------|
| 5.1 | Freeze non-circular `.npz` generation scripts | 2026-08-18 | unassigned | ✅ | H36M, Shelf/Campus, AIST++ |
| 5.2 | Document exact data preprocessing pipeline | 2026-09-01 | unassigned | 🟡 | Partial in `docs/data_audit_summary_2026-08-11.md` |
| 5.3 | Pin environment / dependency versions | 2026-10-01 | unassigned | 🔴 | `requirements.txt` / conda env |
| 5.4 | Add training + evaluation README | 2026-10-15 | unassigned | 🔴 | One-command reproduction |
| 5.5 | Archive final checkpoints with clear naming | 2026-10-31 | unassigned | 🔴 | Include best-epoch info |

---

## 6. Open blockers (must be resolved before submission)

| ID | Blocker | Priority | Impact | Status | Next action |
|---|---------|----------|--------|--------|-------------|
| P0-4 | H36M true-GT learned-model divergence / overfitting | P0 | Cannot trust learned results | 🟡 In progress | v57 done; v25 ablation 1 (`v25_true_gt_baseline_fix`) running |
| P0-5 | v57 H36M true-GT medium result recorded | P0 | — | ✅ Done | v57 best **75.16 mm** @ epoch 3 (saved ckpt **81.47 mm**), early-stopped @ epoch 5; leaderboard updated |
| P1-1 | MPI-INF-3DHP detected-2D / camera / label alignment | P1 | Blocks MPI training/eval | 🟡 In progress | Diagnose/fix coordinate-frame mismatch; re-run DLT until ~20–30 mm |
| P1-2 | v25 crash on Shelf/Campus non-circular smoke | P1 | Blocks cross-domain training |  In progress | Debug CUDA assert in `motionflow_mv/fusion/epipolar_attention_bias.py` |
| P1-3 | AIST++ full medium validation | P1 | No convergence proof | 🟡 In progress | agent-67 to run when GPU free |
| P1-4 | Additional SOTA baselines (VoxelPose, MVPose) | P1 | Weak related-work table | 🔴 Open | Add configs/scripts; run when GPU free |
| P1-5 | Cross-domain training manifest and recipe | P1 | Cross-domain claim unsupported | 🔴 Open | Create `configs/splits/h36m_aist_shelf_campus_mix.yaml` |
| P1-6 | Paper rewrite around sparse-view / cross-domain robustness | P1 | Old narrative invalid | 🟡 In progress | Update `docs/paper_draft_icra_cvpr_2027.md` |

> Full blocker details: `docs/open_blockers.md`

---

## 7. Compute & resource guardrails

| # | Rule | Status |
|---|------|--------|
| 7.1 | **RTX 4090: only one GPU training at a time** |  Currently running v25 true-GT ablation 1 (`v25_true_gt_baseline_fix`) |
| 7.2 | **Do not start new GPU training while ablation 1 is running** |  Active |
| 7.3 | **A800-D and Docker `motionflow` are read-only** | ✅ No writes/starts/modifications |
| 7.4 | CPU-only work (detection, data prep, plotting) allowed in parallel | ✅ |
| 7.5 | Verify `nvidia-smi` before every new GPU launch | 🔴 Manual check required |

---

## 8. Definition of submission-ready

- [ ] All P0 blockers resolved.
- [ ] H36M true-GT leaderboard has final numbers for v25, v57, v80, DLT, and Iskakov.
- [ ] MPI real-detected-2D data complete and validated.
- [ ] At least one cross-domain training experiment completed.
- [ ] Sparse-view `MPJPE@k` curves included.
- [ ] Paper draft reviewed internally with zero placeholder citations.
- [ ] Supplementary materials uploaded and verified in CMT.
- [ ] Submission confirmation email received.

---

## 9. Quick-reference commands

```bash
# Check GPU status before any launch
nvidia-smi

# Monitor v57 medium
tail -f outputs/omniview_fusion_v57_h36m_true_gt_medium.log

# H36M true-GT Iskakov baseline (reproducible)
python experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol h36m --epochs 10 --batch_size 8 --lr 1e-3 --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --log_path outputs/iskakov_h36m_true_gt.log \
    --ckpt_path outputs/iskakov_h36m_true_gt.pth
```

---

## 10. Related docs

- `docs/cvpr2027_status.md` — day-to-day status
- `docs/roadmap_cvpr2027.md` — strategic roadmap
- `docs/open_blockers.md` — P0/P1 blockers
- `docs/results_true_gt_h36m.md` — H36M true-GT numbers
- `docs/results_true_gt_shelf_campus.md` — Shelf/Campus true-GT numbers
- `docs/paper_corrected_outline_cvpr2027.md` — current corrected outline (supersedes `paper_draft_icra_cvpr_2027.md`)
