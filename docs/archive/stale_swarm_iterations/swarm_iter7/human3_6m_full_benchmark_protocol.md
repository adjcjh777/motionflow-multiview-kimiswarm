# Human3.6M full benchmark protocol

**Summary**

I investigated the Human3.6M benchmark story in the repo. The headline numbers (MPJPE 5.74 mm / PA-MPJPE 3.99 mm) are real, but they are only for a **single action** (S1 actions 2–16 → S5 action 02). The canonical full protocol—train on S1/S5/S6/S7/S8 all actions, test on S9/S11 all actions—is not implemented. Worse, I found that the existing S9/S11 `.npz`s are corrupted: triangulating the stored 2D points with the stored cameras gives ~736 mm and ~71,402 mm DLT-vs-GT error, so the test-subject preprocessing is broken and must be fixed before any full benchmark can run.

The report I wrote describes the current state, the gap, a concrete next step (fix preprocessing, generate all subject/action `.npz`s, add `experiments/run_h36m_full_benchmark.py`), expected targets (mean MPJPE ≤ 6 mm / PA-MPJPE ≤ 5 mm), and risks (HuggingFace archive camera-mapping bug, storage, compute, do-not-commit large files).

**Report path:** `docs/swarm_iter7/human36m_full_benchmark_protocol.md`