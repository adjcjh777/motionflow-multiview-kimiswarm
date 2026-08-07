"""Load train/val/test file lists from a YAML split manifest."""

from pathlib import Path
from typing import Dict, List

import yaml


def load_split_manifest(path: str) -> Dict[str, List[str]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return {
        "train": data.get("train", []),
        "val": data.get("val", []),
        "test": data.get("test", []),
    }


def load_multi_dataset_manifest(manifest_paths: List[str]) -> Dict[str, List[str]]:
    """Load multiple manifests and concatenate train/val/test file lists.

    Args:
        manifest_paths: List of YAML manifest paths.

    Returns:
        {'train': [...], 'val': [...], 'test': [...]} with concatenated lists.
    """
    combined: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for path in manifest_paths:
        split = load_split_manifest(path)
        for key in combined:
            combined[key].extend(split[key])
    return combined


def resolve_paths(project_root: str, paths: List[str]) -> List[str]:
    """Resolve relative paths in a manifest against the project root."""
    root = Path(project_root)
    return [str(root / p) for p in paths]


if __name__ == "__main__":
    import json

    manifest_path = "configs/splits/mpiinf3dhp_train_val_test.yaml"
    split = load_split_manifest(manifest_path)
    project_root = str(Path(manifest_path).resolve().parent.parent.parent)
    for key in split:
        split[key] = resolve_paths(project_root, split[key])
        missing = [p for p in split[key] if not Path(p).exists()]
        print(f"{key}: {len(split[key])} files, {len(missing)} missing")
        if missing:
            print("  missing:", missing[:3])
    print(json.dumps({k: len(v) for k, v in split.items()}, indent=2))
