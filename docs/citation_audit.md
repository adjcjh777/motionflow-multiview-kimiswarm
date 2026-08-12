# Citation and Narrative Audit: `docs/paper_draft_icra_cvpr_2027.md`

**Scope:** `docs/paper_draft_icra_cvpr_2027.md`  
**Produced:** 2026-08-12  
**Status:** Read-only audit; paper draft was not modified.

---

## 1. Executive summary

- **No fabricated citations were found.** All seven numbered references in the draft are real publications and are correctly formatted.
- One citation (RUMPL, [4]) was independently re-verified via arXiv and confirmed real.
- Two previously corrected venue typos (SmoothNet and Stacked Hourglass listed as CVPR instead of ECCV) are already fixed in the draft.
- Several **accuracy / narrative issues** were found in the claims around runtime, AIST++ non-circularity, and experimental status. These are detailed below with line numbers and suggested rewrites.
- The overall pivot to **sparse-view / cross-domain robustness on honest, non-circular benchmarks** is well-supported by the project data and is the correct narrative frame.

---

## 2. Citation audit

### 2.1 Numbered references

| # | Reference in draft | Status | Notes / verification |
|---|--------------------|--------|----------------------|
| 1 | Hartley, R. and Zisserman, A. *Multiple View Geometry in Computer Vision*. CUP, 2004. | ✅ Real | Standard textbook; unchanged. |
| 2 | Hartley, R. and Sturm, P. “Triangulation.” *CVIU*, 68(2):146–157, 1997. | ✅ Real | Replaced an earlier duplicate of [1]; verified against Elsevier CVIU index. |
| 3 | Iskakov, K., Burkov, E., Lempitsky, V., and Malkov, Y. “Learnable triangulation of human pose.” *ICCV*, 2019. | ✅ Real | arXiv:1905.05754; ICCV 2019. |
| 4 | Ghasemzadeh, S. A. and Alahi, A. “RUMPL: Ray-based transformers for universal multi-view 2D to 3D human pose lifting.” arXiv:2512.15488, 2025. | ✅ Real | Independently fetched `https://arxiv.org/abs/2512.15488`; title, authors, and arXiv ID match. v1 submitted 17 Dec 2025. |
| 5 | Zhu, W., Ma, X., Liu, Z., Liu, L., Wu, W., and Wang, Y. “MotionBERT: A Unified Perspective on Learning Human Motion Representations.” *ICCV*, 2023. | ✅ Real | arXiv:2210.06551; ICCV 2023. |
| 6 | Zeng, A., Yang, L., Ju, X., Li, J., Wang, J., and Xu, Q. “SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos.” *ECCV*, 2022. | ✅ Real | arXiv:2112.13715; ECCV 2022 (venue corrected from an earlier CVPR typo). |
| 7 | Newell, A., Yang, K., and Deng, J. “Stacked hourglass networks for human pose estimation.” *ECCV*, 2016. | ✅ Real | arXiv:1603.06937; ECCV 2016 (venue corrected from an earlier CVPR typo). |

**Conclusion:** the reference list is clean. No fabricated or non-existent citations are present.

### 2.2 Inline citation usage

- `[1,2]` in Related Work (Section 2) points to two distinct triangulation references — valid.
- `[3]` and `[4]` are used for learnable triangulation and ray-aware transformer related work — valid.
- `[5,6]` and `[7]` are used for temporal pose refinement and 2D pose backbone context — valid.

**Caveat:** the draft describes RUMPL as representing "ray-aware attention" in related work. This is acceptable because RUMPL does introduce a ray-based 3D representation and a View Fusion Transformer. However, the draft should not imply that RUMPL has been reproduced as a baseline; no RUMPL config or checkpoint exists in the repo (confirmed in `docs/paper_corrected_outline_cvpr2027.md`).

---

## 3. Accuracy and narrative issues

### 3.1 Runtime range mismatch

**Location:** Abstract (line 14) and Conclusion (line 371).  
**Issue:** the draft states throughput of **"12.7–194 clips/s"**. The runtime table in Section 5.7 (lines 316–323) reports **12.8 clips/s** at batch size 1 and **194.8 clips/s** at batch size 16. The lower bound in the prose (12.7) does not match the table (12.8). This is a small inconsistency but should be aligned.

**Suggested rewrite:**
> "The system runs at **12.8–194.8 clips/s** on an RTX 4090 …"  
> (or round to **~12.8–195 clips/s** if the exact figure is not needed).

---

### 3.2 AIST++ "DLT direct MJE ≈ 44 mm" is unsupported

**Location:** Section 5.1 (line 190) and Section 5.5.1 (line 292).  
**Issue:** the draft claims AIST++ is non-circular with "DLT direct MJE ≈ 44 mm". The actual full AIST++ DLT baseline, computed on all 1,408 canonical clips, gives:

- Confidence-weighted DLT: **15.93 mm** MPJPE
- Unweighted DLT: **38.11 mm** MPJPE

Neither matches the ≈44 mm figure. The smoke-split numbers are even lower (conf-weighted 6.52 mm, unweighted 12.66 mm). The 44 mm value appears to be an outdated or misremembered rough estimate with no identifiable source in the current outputs.

**Suggested rewrite:**
> "The canonical `.npz` are built from the 9-camera AIST++ annotations and are non-circular (full-set unweighted DLT direct MPJPE ≈ **38 mm**; confidence-weighted ≈ **16 mm**), making it a useful cross-domain stress test."

This ties the claim to the reproducible baseline in `outputs/aistpp_full_dlt_baseline.json`.

---

### 3.3 AIST++ skeleton description is imprecise

**Location:** Section 5.1 (line 190) and Section 5.5.1 (line 292).  
**Issue:** the draft says AIST++ "uses the same 17-joint skeleton as H36M." AIST++ has its own original skeleton; the project maps it to a **17-joint H36M-compatible skeleton**. Stating it is "the same skeleton" is imprecise and could confuse reviewers.

**Suggested rewrite:**
> "It uses a **17-joint H36M-compatible skeleton mapping** …"

---

### 3.4 Sparse-view narrative should not overstate current results

**Location:** Section 3.10 (lines 150–154) and Section 5.4 (lines 259–284).  
**Issue:** the draft correctly reports that v25/v81/v82 produce catastrophic k=2/k=3 errors on true-GT H36M, and that a wrapper fix and DLT-fallback re-evaluation are in progress. This is appropriate, but the narrative should avoid implying that the sparse-view failure is itself a contribution. Currently the draft frames it as evidence for the robustness pivot, which is fine, but the prose should clearly distinguish:

- **Confirmed result:** all current learned variants fail catastrophically at k<4 on true-GT H36M.
- **Preliminary result:** MPI-INF-3DHP smoke curves (Section 5.4, bottom table) were measured on GT-projected 2D and should not be used for final model selection — the draft already says this, but the caveat could be stronger.

**Suggested rewrite for Section 5.4 intro:**
> "On true-GT H36M, every learned variant we have evaluated so far degrades catastrophically when fewer than four views are supplied (Table X). The MPI-INF-3DHP curves below are a smoke diagnostic on GT-projected 2D; they illustrate architectural differences in low-view behavior but are not a substitute for a true detected-2D sparse-view benchmark."

---

### 3.5 Cross-domain AIST++ and Shelf/Campus claims should emphasize preliminary status

**Location:** Section 5.5.1 (lines 286–303) and Section 5.5.2 (lines 304–306).  
**Issue:** the draft says the learned models "are far from convergence after only three smoke epochs" and treats cross-domain transfer as "an active research direction." This is accurate. However, the abstract and conclusion could be tightened to avoid implying that cross-domain validation is already a strong result.

**Suggested rewrite (abstract):**
> "We evaluate cross-domain behavior on AIST++ and Shelf/Campus, and expose the fusion module as a pluggable `MultiViewFusionPlugin` inside MotionFlow."

This removes any implication that cross-domain transfer has been solved.

---

### 3.6 Calibration-robustness claims need true-GT verification

**Location:** Section 3.8 (lines 105–128) and Section 5.6 (lines 308–313).  
**Issue:** the draft describes calibration-perturbation training and a learned intrinsic-correction layer. It notes that earlier perturbation matrices were measured on the old circular protocol and should be re-measured on true GT. This caveat is correct and should be retained. Before submission, the absolute claims about rotation / principal-point robustness should be backed by true-GT numbers or framed purely as a method description.

**Suggested rewrite for Section 5.6:**
> "The perturbation recipe and `IntrinsicCorrection` layer described in Section 3.8 are implemented, but absolute robustness numbers on true-GT H36M are still being collected. Until those runs complete, we report the architectural design as a contribution and do not claim a calibrated-robustness improvement over the baselines."

---

## 4. Suggested rewrites around sparse-view / cross-domain robustness

### 4.1 Title / abstract framing

The current abstract is strong. One tweak would make the accuracy gap clearer:

> "On true-GT Human3.6M, our best learned variant (v25 stability) reaches **30.83 mm**, still behind Iskakov (**23.40 mm**) and confidence-weighted DLT (**25.67 mm**), **showing that the real remaining challenge is not incremental accuracy on circular leaderboards but robustness under domain shift and view scarcity.**"

This keeps the pivot front-and-center.

### 4.2 Contribution list (lines 26–30)

The contribution list is already aligned with the robustness pivot. Suggested minor edit to item 4:

> "4. Empirical validation on **Human3.6M true-GT**, **Campus sparse-view**, and **AIST++ cross-domain transfer**, including efficiency numbers on an RTX 4090, **with explicit failure-mode analysis when views are removed**."

### 4.3 Related Work (Section 2)

Add a sentence that positions RUMPL correctly:

> "Ray-aware transformer methods such as RUMPL [4] propose universal multi-view 2D-to-3D lifting with ray-based representations; we cite them as recent related work and do not reproduce their model."

### 4.4 Conclusion (lines 365–372)

The conclusion is factually consistent with `docs/results_true_gt_h36m.md`. To reinforce the robustness frame, consider ending with:

> "The honest leaderboards reset expectations: on true-GT H36M, geometric and learnable-triangulation baselines still outperform our current learned variants. The paper's contribution is therefore not a new MPJPE record, but a geometry-first fusion architecture, an honest evaluation protocol, and a set of failure-mode diagnostics that point toward sparse-view and cross-domain robustness as the central research problems."

---

## 5. Corrected reference list (no changes required)

All seven references are accurate. For completeness, the verified list is repeated below exactly as it appears in the draft:

1. Hartley, R. and Zisserman, A. *Multiple View Geometry in Computer Vision*. Cambridge University Press, 2004.
2. Hartley, R. and Sturm, P. “Triangulation.” *Computer Vision and Image Understanding*, 68(2):146–157, 1997.
3. Iskakov, K., Burkov, E., Lempitsky, V., and Malkov, Y. “Learnable triangulation of human pose.” *ICCV*, 2019.
4. Ghasemzadeh, S. A. and Alahi, A. “RUMPL: Ray-based transformers for universal multi-view 2D to 3D human pose lifting.” arXiv:2512.15488, 2025.
5. Zhu, W., Ma, X., Liu, Z., Liu, L., Wu, W., and Wang, Y. “MotionBERT: A Unified Perspective on Learning Human Motion Representations.” *ICCV*, 2023.
6. Zeng, A., Yang, L., Ju, X., Li, J., Wang, J., and Xu, Q. “SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos.” *ECCV*, 2022.
7. Newell, A., Yang, K., and Deng, J. “Stacked hourglass networks for human pose estimation.” *ECCV*, 2016.

---

## 6. Files consulted

- `docs/paper_draft_icra_cvpr_2027.md`
- `docs/citation_verification.md` (prior verification)
- `docs/results_true_gt_h36m.md`
- `docs/results_true_gt_shelf_campus.md`
- `docs/results_aistpp_dlt_baseline.md`
- `docs/results_aistpp_train_val_test_mixed_dlt_baseline.md`
- `docs/paper_corrected_outline_cvpr2027.md`
- `outputs/aistpp_full_dlt_baseline.json`
- `https://arxiv.org/abs/2512.15488` (RUMPL verification)

---

## 7. Blockers / next steps

- **None for the audit itself.** The deliverable is complete.
- **For the paper:** apply the suggested rewrites, replace the AIST++ 44 mm claim with the verified baseline numbers, and align the runtime range. Before any camera-ready submission, re-run a fresh citation pass after any new related-work references are added.
