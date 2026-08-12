# Agent-18: In-the-Wild Dataset Review for v49 (AIST++, 3DPW, EHF)

**Owner:** Agent-18  
**Scope:** ANALYZE — review additional real-world datasets beyond the current 3DPW pseudo-mode pipeline, with an eye toward v49 domain generalization.  
**Branch:** `v48-domain`  
**Related files:**
- `motionflow_mv/data/webbridge_mixed_dataset.py`
- `motionflow_mv/data/webbridge_loader.py`
- `experiments/convert_3dpw_multiview.py`
- `scripts/download_aistpp.py`
- `docs/proposals/v48_domain_generalization.md`

---

## Executive Summary

The v48 pipeline already consumes five canonical WebBridge sources: H36M, MPI-INF-3DHP, AIST++, Shelf/Campus, and 3DPW (pseudo-mode). For v49 we need data that tests *true* in-the-wild generalization: moving cameras, unconstrained environments, noisy detections, and motion styles not present in dance/studio captures.

| Dataset | Type | Multi-view? | Ground-truth 3D | Current status | v49 priority |
|---------|------|-------------|-----------------|----------------|--------------|
| 3DPW pseudo | synthetic rig re-projection | yes (virtual) | yes | used for training/eval | baseline in-the-wild proxy |
| 3DPW actual | monocular video + IMU | no (single moving camera) | yes (SMPL joints) | loader exists, not yet in mixed manifest | **high** |
| AIST++ | studio multi-view dance | yes (9 rig) | yes | already in manifest | medium (motion diversity) |
| EHF | **unidentified / not canonical** | — | — | not present | **blocked — needs clarification** |

---

## 1. AIST++

### What it is
AIST++ (Google, 2021) is a large-scale 3D dance-motion dataset. It provides 9 calibrated static cameras, 2D keypoint tracks, and 3D joint positions derived from multi-view triangulation. The `scripts/download_aistpp.py` downloader already fetches the annotation-only archives, and `motionflow_mv/data/webbridge_loader.py` implements `convert_aistpp`.

### Relevance to v49
- **Not truly in-the-wild.** Lighting, background, and camera rig are highly controlled.
- **High motion complexity.** Extreme limb articulations, fast rotations, and self-occlusion are far richer than H36M/MPI, making it valuable for *motion-domain* generalization even if not *environment-domain*.
- **Skeleton alignment.** `webbridge_mixed_dataset.py` maps AIST++ directly to the canonical 17-joint layout (`"aist": np.arange(17)`), so integration cost is zero.

### Pros
- Already wired into the mixed loader and manifests.
- Large volume of dynamic motion reduces overfitting to H36M/MPI walking/posing.
- 9 views let us stress-test sparse-view behavior at `k = 2..9`.

### Cons
- Studio lighting and chroma-key backgrounds create a distribution gap with real deployment.
- No moving camera; cannot test monocular/egocentric scenarios.

### v49 recommendation
Keep AIST++ as a **motion-diversity domain**, but do not rely on it for in-the-wild camera/generalization metrics. Consider using it as an source-only domain for domain-adversarial training because its motion distribution is distinct from H36M/MPI.

---

## 2. 3DPW

### What it is
3DPW is the closest real-world 3D human-pose benchmark currently available. It consists of monocular video + IMU, with per-frame moving-camera parameters and SMPL 3D joints. The project already has `experiments/convert_3dpw_multiview.py`, which can output either:

1. **pseudo mode** — builds a static virtual multi-view rig around the actor (default, currently used).
2. **actual mode** — keeps the real moving camera as a single-view sequence.

### Relevance to v49
- **actual mode is the real test.** It is the only existing source with a genuine moving camera, making it the natural benchmark for v49 generalization.
- The v48 design doc explicitly calls out `v48_3dpw_actual_val_paths` and per-domain dropout (`3DPW p=0.15`).
- Loader side: `webbridge_mixed_dataset.py` can already load 3DPW `pseudo` `.npz` files because the skeleton map exists (`"3dpw": np.array(...)`). `actual` mode requires handling per-frame `camera_K_frames`, `camera_R_frames`, and `camera_t_frames`, which the current `WebBridgeCanonical17Dataset` ignores.

### Pros
- True outdoor/indoor unconstrained environments.
- Moving camera = realistic monocular deployment scenario.
- 24 SMPL joints map cleanly to the canonical 17-joint skeleton.

### Cons
- Small dataset size (~1k short clips) compared to studio datasets.
- Single-person, single-camera only; cannot evaluate multi-view geometry fusion directly in `actual` mode.
- SMPL joint noise and IMU drift add label uncertainty.

### v49 recommendation
- **Promote 3DPW actual mode to a first-class val/test domain** (separate from pseudo-mode training).
- Add a lightweight wrapper in `motionflow_mv/data/webbridge_3dpw_actual_loader.py` as sketched in `docs/proposals/v48_domain_generalization.md`.
- Use `actual` clips for v49 zero-shot monocular evaluation and `pseudo` clips only as auxiliary training.

---

## 3. EHF

### Status
**EHF could not be unambiguously identified as a canonical 3D human-pose dataset.** No reference to "EHF" exists in the current codebase, the v48 proposal, or the AGENTS notes. Web searches did not return a recognized body-pose benchmark with this acronym.

### Possible interpretations
| Acronym | Dataset | Relevance |
|---------|---------|-----------|
| EHF | EgoHands & Face / MediaPipe EHF | hand/face focused, not body pose; unlikely relevant |
| EHF | Event-based High-Frequency dataset | not a standard 3D pose benchmark |
| EHF | Internal project codename | unknown |

### v49 recommendation
**Do not integrate EHF until the acronym is clarified.** If the team intended a specific dataset (e.g., ExPI, InstaVariety, UPI-SAPT, or an internal capture), open a follow-up issue to confirm the name, download URL, and skeleton format. Until then, leave EHF out of v49 planning to avoid scope creep.

---

## 4. Comparative integration effort

| Concern | 3DPW actual | AIST++ | EHF |
|---------|-------------|--------|-----|
| Data already available | partial (raw `.pkl` required) | yes (download script) | no |
| Loader work | medium (per-frame cameras) | none | unknown |
| Skeleton map | exists | exists | unknown |
| Domain ID slot | `5` in `DATASET_IDS` | `2` | unknown |
| Multi-view | no (actual) / yes (pseudo) | yes | unknown |
| In-the-wild value | very high | low (motion only) | unknown |
| Risk | low (known format) | low | high (undefined) |

---

## 5. Recommendations for v49

1. **Double down on 3DPW actual mode.** It is the highest-value, lowest-ambiguity in-the-wild source the project already owns. Finish the v48 actual-mode loader/eval and carry it into v49.
2. **Use AIST++ for motion-domain diversity**, not for environment-domain generalization. Keep it in the mixed manifest but weight it as a studio domain.
3. **Resolve EHF before committing engineering effort.** If it is intended as a real dataset, produce a separate analysis with download links, skeleton layout, and a conversion script.
4. **Consider additional in-the-wild candidates once 3DPW actual is stable:**
   - **ExPI** — extreme poses, outdoor.
   - **UPI-SAPT / InstaVariety** — Internet video pseudo-labels for pre-training.
   - **Ego4D** — egocentric in-the-wild video (no 3D GT; only weak supervision).

---

## 6. Open questions / blockers

- **EHF definition:** What dataset does "EHF" refer to? This report is blocked on clarification.
- **3DPW actual-mode licensing:** Confirm the raw 3DPW download is present on A800-D and locally.
- **Per-frame camera storage:** Decide whether to store actual-mode cameras as extra arrays in the `.npz` or to restructure `WebBridgeCanonical17Dataset` to read them.
- **Evaluation metric:** For v49 monocular in-the-wild evaluation, define whether `MPJPE@1` on 3DPW actual is the primary benchmark or whether additional datasets are required.

---

## Conclusion

For v49, the project should **prioritize 3DPW actual-mode integration** and treat AIST++ as a complementary motion-diversity studio domain. The EHF dataset is currently undefined and should not be integrated until its identity and format are confirmed. This keeps v49 scoped, avoids wasted conversion work, and aligns with the v48 goal of true cross-domain generalization.
