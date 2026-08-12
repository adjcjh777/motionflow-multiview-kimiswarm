#!/usr/bin/env python3
"""Generate configs/splits/mix_true_gt_v2.yaml with domain-balanced weights and a test split.

This is an improved version of ``scripts/generate_mix_true_gt_v2.py``.  It keeps
the same H36M true-GT / AIST++ / Shelf/Campus sources but:

1. Labels Shelf and Campus files as separate domains names (``"shelf"`` and
   ``"campus"``) so they are compatible with
   :class:`motionflow_mv.data.webbridge_mixed_dataset.WebBridgeMixedDataset`.
2. Splits a deterministic fraction of each domain's validation pool into a
   held-out ``test`` split.
3. Adds ``domain_balancing_weights`` computed as inverse-frequency over the
   training file counts, normalised to a mean of 1.0.

Split design:
- H36M true-GT: standard protocol S1,S5-S8 -> train; S9,S11 -> val, with a
  deterministic 10% of the val files moved to test.
- AIST++: train genres gBR, gHO, gJB, gJS, gKR, gLH; val genres
  gLO, gMH, gPO, gWA.  10% of the val-genre files are moved to test.
- Shelf/Campus (detected): train files -> train; val files -> val, with 10% of
  the val files moved to test (minimum one file per domain).

Output manifest is written to configs/splits/mix_true_gt_v2.yaml.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]

TEST_FRAC = 0.10
TEST_SEED = 42


def _relative(files: List[Path]) -> List[str]:
    return [str(f.relative_to(ROOT)).replace("\\", "/") for f in files]


def _label_for(file_list: List[Path], label: str) -> List[str]:
    return [label] * len(file_list)


def h36m_splits(
    test_frac: float = TEST_FRAC,
    test_seed: int = TEST_SEED,
) -> Tuple[List[Path], List[Path], List[Path]]:
    train, val = [], []
    for f in sorted((ROOT / "data/h36m_true_gt").glob("*_multiview_m.npz")):
        subject = f.stem.split("_")[1]
        if subject in {"09", "11"}:
            val.append(f)
        else:
            train.append(f)
    val, test = _split_val_to_test(val, "h36m", test_frac, test_seed)
    return train, val, test


def aist_splits(
    test_frac: float = TEST_FRAC,
    test_seed: int = TEST_SEED,
) -> Tuple[List[Path], List[Path], List[Path]]:
    train_genres = {"gBR", "gHO", "gJB", "gJS", "gKR", "gLH"}
    val_genres = {"gLO", "gMH", "gPO", "gWA"}
    train, val = [], []
    for f in sorted((ROOT / "data/webbridge/aistpp_canonical").glob("*_multiview.npz")):
        genre = f.stem.split("_")[0]
        if genre in val_genres:
            val.append(f)
        elif genre in train_genres:
            train.append(f)
        else:
            print(f"[warn] Unrecognised AIST++ genre '{genre}' for {f.name}; adding to train.")
            train.append(f)
    val, test = _split_val_to_test(val, "aist", test_frac, test_seed)
    return train, val, test


def shelf_campus_splits(
    test_frac: float = TEST_FRAC,
    test_seed: int = TEST_SEED,
) -> Tuple[List[Path], List[Path], List[Path], List[Path], List[Path]]:
    """Return train/val/test for shelf and campus separately.

    Returns
    -------
    shelf_train, shelf_val, shelf_test, campus_train, campus_val, campus_test
    """
    shelf_train, shelf_val, shelf_test = [], [], []
    campus_train, campus_val, campus_test = [], [], []
    for f in sorted((ROOT / "data/webbridge/shelf_campus_detected").glob("*_m.npz")):
        if "shelf" in f.name:
            target_train, target_val = shelf_train, shelf_val
        elif "campus" in f.name:
            target_train, target_val = campus_train, campus_val
        else:
            print(f"[warn] Unknown shelf/campus file {f.name}; skipping.")
            continue

        if "_val_" in f.name:
            target_val.append(f)
        elif "_train_" in f.name:
            target_train.append(f)
        else:
            # Fallback: first seq of each scene goes to train.
            if f.name.startswith(("shelf_seq1_train", "campus_seq1_train")):
                target_train.append(f)
            else:
                target_val.append(f)

    shelf_val, shelf_test = _split_val_to_test(shelf_val, "shelf", test_frac, test_seed)
    campus_val, campus_test = _split_val_to_test(campus_val, "campus", test_frac, test_seed)
    return shelf_train, shelf_val, shelf_test, campus_train, campus_val, campus_test


def _split_val_to_test(
    val: List[Path],
    domain: str,
    test_frac: float = TEST_FRAC,
    test_seed: int = TEST_SEED,
) -> Tuple[List[Path], List[Path]]:
    """Move a deterministic fraction of the validation pool to test.

    Uses ``test_frac`` and ``test_seed``.  Always moves at least one file when
    the validation pool is non-empty.
    """
    if not val:
        return [], []

    rng = random.Random(test_seed ^ hash(domain) ^ 0x9E3779B9)
    shuffled = val.copy()
    rng.shuffle(shuffled)

    n_test = max(1, int(round(len(shuffled) * test_frac)))
    # Always keep at least one file in val.  If the pool is a single file,
    # do not hold it out, since a separate test split is impossible without
    # leaving val empty.
    if n_test >= len(shuffled):
        n_test = max(0, len(shuffled) - 1)

    test = sorted(shuffled[:n_test])
    new_val = sorted(shuffled[n_test:])
    return new_val, test


def _compute_domain_weights(train_names: List[str]) -> Dict[str, float]:
    """Inverse-frequency weights normalised to a mean of 1.0."""
    from collections import Counter

    counts = Counter(train_names)
    if not counts:
        return {}

    n_domains = len(counts)
    total = sum(counts.values())
    weights = {domain: total / (n_domains * count) for domain, count in counts.items()}
    return {domain: round(w, 6) for domain, w in weights.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the balanced mix_true_gt_v2 split manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs/splits/mix_true_gt_v2.yaml",
        help="Output YAML path (default: configs/splits/mix_true_gt_v2.yaml).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=TEST_FRAC,
        help="Fraction of validation files to hold out as test (default: 0.1).",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=TEST_SEED,
        help="Seed for deterministic test split (default: 42).",
    )
    args = parser.parse_args()

    h36m_train, h36m_val, h36m_test = h36m_splits(args.test_frac, args.test_seed)
    aist_train, aist_val, aist_test = aist_splits(args.test_frac, args.test_seed)
    (
        shelf_train,
        shelf_val,
        shelf_test,
        campus_train,
        campus_val,
        campus_test,
    ) = shelf_campus_splits(args.test_frac, args.test_seed)

    train_files = (
        h36m_train
        + aist_train
        + shelf_train
        + campus_train
    )
    val_files = (
        h36m_val
        + aist_val
        + shelf_val
        + campus_val
    )
    test_files = (
        h36m_test
        + aist_test
        + shelf_test
        + campus_test
    )

    train_names = (
        _label_for(h36m_train, "h36m")
        + _label_for(aist_train, "aist")
        + _label_for(shelf_train, "shelf")
        + _label_for(campus_train, "campus")
    )
    val_names = (
        _label_for(h36m_val, "h36m")
        + _label_for(aist_val, "aist")
        + _label_for(shelf_val, "shelf")
        + _label_for(campus_val, "campus")
    )
    test_names = (
        _label_for(h36m_test, "h36m")
        + _label_for(aist_test, "aist")
        + _label_for(shelf_test, "shelf")
        + _label_for(campus_test, "campus")
    )

    domain_weights = _compute_domain_weights(train_names)

    manifest = {
        "name": "H36M True GT + AIST++ + Shelf/Campus mixed training (v2 balanced)",
        "description": (
            "Cross-dataset mix using non-circular H36M true-GT, AIST++ canonical "
            "multi-view clips, and detected Shelf/Campus. H36M follows the standard "
            "S1,S5-S8->S9/S11 protocol. AIST++ is split by genre to avoid choreography "
            "overlap. A deterministic fraction of each domain's validation pool is "
            "held out as a test split; single-file val pools are retained in val. "
            "Shelf and Campus are treated as separate domains."
        ),
        "domain_balancing_weights": domain_weights,
        "train_paths": _relative(train_files),
        "train_names": train_names,
        "val_paths": _relative(val_files),
        "val_names": val_names,
        "test_paths": _relative(test_files),
        "test_names": test_names,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# Mixed-dataset training manifest (H36M true-GT + AIST++ + Shelf/Campus).\n")
        f.write("# Generated by scripts/generate_mix_true_gt_v2_balanced.py; do not edit by hand.\n")
        f.write("#\n")
        f.write("# Improvements over the original v2 split:\n")
        f.write("#   - Shelf and Campus are labelled as separate domains ('shelf', 'campus').\n")
        f.write("#   - A deterministic 10% of each domain's validation pool is held out as test.\n")
        f.write("#   - Domain-balancing weights (inverse frequency, mean=1.0) are recorded.\n")
        f.write("#\n")
        f.write("# H36M true-GT: non-circular mocap world coordinates from data/h36m_true_gt/.\n")
        f.write("#   Train subjects: S1, S5, S6, S7, S8\n")
        f.write("#   Val subjects:   S9, S11 (minus the test hold-out)\n")
        f.write("#   Test hold-out:  10% of the val subjects (deterministic)\n")
        f.write("# AIST++: canonical multi-view .npz under data/webbridge/aistpp_canonical/.\n")
        f.write("#   Train genres: gBR, gHO, gJB, gJS, gKR, gLH\n")
        f.write("#   Val genres:   gLO, gMH, gPO, gWA\n")
        f.write("#   Test hold-out: 10% of val-genre files (deterministic)\n")
        f.write("# Shelf/Campus: detected 2D + true 3D under data/webbridge/shelf_campus_detected/.\n")
        f.write("#   Train files: *_train_detected_m.npz\n")
        f.write("#   Val files:   *_val_detected_m.npz\n")
        f.write("#   Test hold-out: 10% of the val files when the pool is large enough;\n")
        f.write("#                  single-file val pools are kept in val so val stays non-empty.\n")
        f.write("\n")
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"[mix_true_gt_v2_balanced] wrote {args.output}")
    print(
        f"  train samples: {len(train_files)} "
        f"(h36m={len(h36m_train)}, aist={len(aist_train)}, shelf={len(shelf_train)}, campus={len(campus_train)})"
    )
    print(
        f"  val   samples: {len(val_files)} "
        f"(h36m={len(h36m_val)}, aist={len(aist_val)}, shelf={len(shelf_val)}, campus={len(campus_val)})"
    )
    print(
        f"  test  samples: {len(test_files)} "
        f"(h36m={len(h36m_test)}, aist={len(aist_test)}, shelf={len(shelf_test)}, campus={len(campus_test)})"
    )
    print(f"  domain balancing weights: {domain_weights}")


if __name__ == "__main__":
    main()
