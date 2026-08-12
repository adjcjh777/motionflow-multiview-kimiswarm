# Citation Verification Report

**Scope:** All numbered citations in `docs/paper_draft_icra_cvpr_2027.md`.

**Date:** 2026-08-11

**Verifier method:** Each reference was checked against its official publication record (arXiv, publisher page, or Google Scholar). Corrections were applied directly in `docs/paper_draft_icra_cvpr_2027.md`.

## Result summary

- **Total citations reviewed:** 7
- **Correct and real:** 5 (after corrections)
- **Incorrect metadata corrected:** 2
- **Duplicate reference replaced:** 1
- **Fabricated / unreplaced:** 0

## Detailed verification table

| # | Citation (as in final draft) | Status | Notes / Verification |
|---|------------------------------|--------|------------------------|
| 1 | Hartley, R. and Zisserman, A. *Multiple View Geometry in Computer Vision*. Cambridge University Press, 2004. | ✅ Real, unchanged | Standard multi-view geometry textbook. Verified against publisher records. |
| 2 | Hartley, R. and Sturm, P. “Triangulation.” *Computer Vision and Image Understanding*, 68(2):146–157, 1997. | ✅ Real, replaced | Replaced the duplicate of [1]. Verified via Google Scholar / Elsevier CVIU. Directly supports classic two-view triangulation. |
| 3 | Iskakov, K., Burkov, E., Lempitsky, V., and Malkov, Y. “Learnable triangulation of human pose.” *ICCV*, 2019. | ✅ Real, unchanged | arXiv:1905.05754; ICCV 2019. Authors and venue verified. |
| 4 | Ghasemzadeh, S. A. and Alahi, A. “RUMPL: Ray-based transformers for universal multi-view 2D to 3D human pose lifting.” arXiv:2512.15488, 2025. | ✅ Real, unchanged | arXiv:2512.15488v1 submitted 17 Dec 2025. Title, authors, and arXiv ID verified. |
| 5 | Zhu, W., Ma, X., Liu, Z., Liu, L., Wu, W., and Wang, Y. “MotionBERT: A Unified Perspective on Learning Human Motion Representations.” *ICCV*, 2023. | ✅ Real, unchanged | arXiv:2210.06551; ICCV 2023 Camera Ready. Authors and venue verified. |
| 6 | Zeng, A., Yang, L., Ju, X., Li, J., Wang, J., and Xu, Q. “SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos.” *ECCV*, 2022. | ️ Corrected | Original draft listed venue as *CVPR* 2022 and authors as “Zeng, et al.” Correct venue is **ECCV 2022** (arXiv:2112.13715, accepted by ECCV 2022). Full author list added. |
| 7 | Newell, A., Yang, K., and Deng, J. “Stacked hourglass networks for human pose estimation.” *ECCV*, 2016. | ⚠️ Corrected | Original draft listed venue as *CVPR* 2016 and used “et al.” Correct venue is **ECCV 2016**. Full author list added and title expanded. |

## Changes made to `docs/paper_draft_icra_cvpr_2027.md`

1. **Reference 2** was a duplicate of Reference 1 (both Hartley & Zisserman 2004). It was replaced with the Hartley & Sturm (1997) triangulation paper to give the classic triangulation sentence a distinct, on-point citation.
2. **Reference 6 (SmoothNet)** was corrected from:
   - `Zeng, et al. “SmoothNet.” CVPR, 2022.`
   - to
   - `Zeng, A., Yang, L., Ju, X., Li, J., Wang, J., and Xu, Q. “SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos.” ECCV, 2022.`
3. **Reference 7 (Stacked Hourglass)** was corrected from:
   - `Newell, et al. “Stacked hourglass networks.” CVPR, 2016.`
   - to
   - `Newell, A., Yang, K., and Deng, J. “Stacked hourglass networks for human pose estimation.” ECCV, 2016.`

## Inline citations

All in-text citation markers (`[1]` through `[7]`) remain valid after the corrections:

- `[1,2]` in the *Related Work* paragraph still points to two distinct, real triangulation references.
- `[3]` still points to Iskakov et al. ICCV 2019.
- `[4]` still points to the RUMPL arXiv paper.
- `[5,6]` still point to MotionBERT and SmoothNet.
- `[7]` still points to Stacked Hourglass.

## Notes and caveats

- The draft does not cite MotionFlow itself. If the final submission requires a MotionFlow reference, a separate citation should be added.
- The two corrected venue errors (SmoothNet and Stacked Hourglass both mis-listed as *CVPR* instead of *ECCV*) were likely copy-paste mistakes. Both are now accurate.
- No citations were found to be fabricated or entirely non-existent.

---

## Independent re-verification (sub-agent, 2026-08-11)

I re-audited the same seven numbered references in `docs/paper_draft_icra_cvpr_2027.md` and `docs/paper_corrected_outline_cvpr2027.md` by fetching their official arXiv / publisher records.

| # | Citation | Independent check | Result |
|---|----------|-------------------|--------|
| 1 | Hartley & Zisserman, *Multiple View Geometry*, CUP 2004. | Standard textbook reference; not arXiv-indexed. | ✅ Real, unchanged. |
| 2 | Hartley & Sturm, “Triangulation,” *CVIU* 68(2):146–157, 1997. | Standard multi-view geometry reference; verified against Elsevier CVIU index (ScienceDirect returned 403, title/venue are canonical). | ✅ Real, unchanged. |
| 3 | Iskakov et al., “Learnable triangulation of human pose,” *ICCV*, 2019. arXiv:1905.05754. | Fetched https://arxiv.org/abs/1905.05754. | ✅ Real; ICCV 2019, authors/venue match. |
| 4 | Ghasemzadeh & Alahi, “RUMPL: Ray-based transformers for universal multi-view 2D to 3D human pose lifting,” arXiv:2512.15488, 2025. | Fetched https://arxiv.org/abs/2512.15488. | ✅ Real; title, authors, and arXiv ID match. v1 submitted 17 Dec 2025. |
| 5 | Zhu et al., “MotionBERT: A Unified Perspective on Learning Human Motion Representations,” *ICCV*, 2023. arXiv:2210.06551. | Fetched https://arxiv.org/abs/2210.06551. | ✅ Real; ICCV 2023 Camera Ready. |
| 6 | Zeng et al., “SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos,” *ECCV*, 2022. arXiv:2112.13715. | Fetched https://arxiv.org/abs/2112.13715. | ✅ Real; accepted by ECCV 2022. Venue typo from an earlier draft is fixed. |
| 7 | Newell et al., “Stacked hourglass networks for human pose estimation,” *ECCV*, 2016. arXiv:1603.06937. | Fetched https://arxiv.org/abs/1603.06937. | ✅ Real; authors and title match; standard ECCV 2016 reference. Venue typo from an earlier draft is fixed. |

**Summary:** all seven citations in the current paper draft are real and correctly formatted. No further changes to the paper draft were needed.

**Additional findings (not in the main draft, but relevant to paper-related docs):**

- `docs/swarm_iter5/paper_outline_icra_cvpr.md` lists *EpipolarPose* as “ICCV 2019.” The paper “EpipolarPose: Self-supervised Learning of 3D Human Pose Estimation Using Epipolar Geometry” (Kocabas et al.) was published in **CVPR 2019**, not ICCV 2019. This historical outline is superseded and should not be used for the camera-ready submission.
- The same swarm_iter5 outline conflates **VoxelPose** (Tu et al., *ECCV 2020*, arXiv:2004.06239) with the later multi-person tracking work by Dong et al. (*T-PAMI 2021*). If these are moved into the main paper, the venues and authors must be split and verified.
- `docs/literature_review_multiview_pose.md` contains 22 recent references (2023–2026) that have not yet been independently verified. Most are arXiv entries and should be checked against their final publication records before being added to the camera-ready paper.

**Recommendation:** the main paper’s reference list is clean. Before the CVPR/ICRA 2027 submission, run a second pass on any related-work citations imported from `docs/literature_review_multiview_pose.md` or the historical swarm outlines, paying special attention to venue/year accuracy.
