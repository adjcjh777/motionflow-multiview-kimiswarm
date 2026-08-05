"""Reusable benchmark protocol for training/evaluation and multi-seed reporting.

The :class:`BenchmarkProtocol` class provides a model-agnostic harness for
reproducible evaluation.  It wraps:

* metric computation via :func:`motionflow_mv.eval.metrics.compute_all_metrics`,
* optional multi-seed training via external scripts,
* and manifest logging so that every run can be audited.

Typical usage::

    cfg = BenchmarkConfig(dataset="mpiinf3dhp", split="test")
    protocol = BenchmarkProtocol(cfg)

    # single evaluation
    report = protocol.evaluate_model(model, dataloader, device="cpu")

    # multi-seed training (dry-run or real)
    manifest = protocol.run_multi_seed(
        script="experiments/train.py",
        base_args="--epochs 20 --batch_size 8",
        seeds=[42, 43, 44],
        out_dir="outputs/my_model/seeds",
        base_name="my_model",
        dry_run=False,
    )
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from .metrics import compute_all_metrics, summarize_metrics


ScalarOrArray = Union[float, np.ndarray]


@dataclass
class BenchmarkConfig:
    """Configuration shared across all benchmark phases.

    Attributes:
        dataset: Dataset name, e.g. ``"mpiinf3dhp"`` or ``"h36m"``.
        split: ``"train"``, ``"val"`` or ``"test"``.
        clip_len: Number of frames per temporal clip.
        stride: Stride between consecutive clips / windows.
        root_joint: Index of the root (pelvis) joint used for root-relative metrics.
        unit_scale: Scale factor to convert model units to millimeters.  Defaults to
            ``1000.0`` because most models output meters while the metrics expect mm.
        seed: Default random seed for reproducibility.
    """

    dataset: str
    split: str
    clip_len: int = 13
    stride: int = 1
    root_joint: int = 0
    unit_scale: float = 1000.0
    seed: int = 42


class BenchmarkProtocol:
    """Encapsulates training, evaluation and multi-seed reproducibility.

    Args:
        cfg: Benchmark configuration.
    """

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.frames_evaluated: List[int] = []
        self.last_report: Optional[Dict[str, ScalarOrArray]] = None

    @staticmethod
    def set_seed(seed: int) -> None:
        """Pin random seeds for Python, NumPy and PyTorch."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: Union[str, torch.device],
        parents: Optional[np.ndarray] = None,
    ) -> Dict[str, ScalarOrArray]:
        """Evaluate a model on *dataloader* and return a metrics report.

        The model is put into evaluation mode and no gradients are computed.
        Temporal outputs are flattened to per-frame predictions before metric
        computation.  Inputs and outputs are scaled by ``self.cfg.unit_scale``.

        Args:
            model: PyTorch model with signature ``model(x, K=K, R=R, t=t)``.
            dataloader: DataLoader yielding ``(x, y, K, R, t)`` tuples.
            device: Device to run inference on.
            parents: Optional skeleton parent array for bone-length error.

        Returns:
            Dictionary returned by :func:`compute_all_metrics` plus the keys
            ``root_joint`` and ``unit_scale`` for bookkeeping.
        """
        model.to(device)
        model.eval()

        all_preds: List[np.ndarray] = []
        all_gts: List[np.ndarray] = []

        with torch.no_grad():
            for batch in dataloader:
                xb, yb, K, R, t = batch
                xb = xb.to(device)
                yb = yb.to(device)
                K = K.to(device)
                R = R.to(device)
                t = t.to(device)

                out = model(xb, K=K, R=R, t=t)
                # Models may return extra outputs; keep the primary prediction.
                if isinstance(out, (tuple, list)):
                    pred = out[0]
                else:
                    pred = out

                all_preds.append(pred.cpu().numpy())
                all_gts.append(yb.cpu().numpy())

        pred = np.concatenate(all_preds, axis=0)  # (N, T, J, 3)
        gt = np.concatenate(all_gts, axis=0)  # (N, T, J, 3)

        # Flatten temporal dimension so all frames participate in per-frame metrics.
        pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
        gt = gt.reshape(-1, gt.shape[-2], gt.shape[-1])

        pred_mm = pred * self.cfg.unit_scale
        gt_mm = gt * self.cfg.unit_scale

        report = compute_all_metrics(pred_mm, gt_mm, parents=parents)
        report["root_joint"] = self.cfg.root_joint
        report["unit_scale"] = self.cfg.unit_scale

        self.last_report = report
        return report

    def run(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: Union[str, torch.device],
        out_dir: Union[str, Path],
        parents: Optional[np.ndarray] = None,
    ) -> Dict[str, ScalarOrArray]:
        """Evaluate and persist a ``results.json`` report.

        Args:
            model, dataloader, device, parents: forwarded to :meth:`evaluate_model`.
            out_dir: Directory where ``results.json`` (and a summary ``results.txt``)
                will be written.

        Returns:
            Metrics report dictionary.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        report = self.evaluate_model(model, dataloader, device, parents=parents)

        # Convert numpy arrays to lists for JSON serialization.
        serializable: Dict[str, Any] = {}
        for k, v in report.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = v

        manifest = {
            "config": {
                "dataset": self.cfg.dataset,
                "split": self.cfg.split,
                "clip_len": self.cfg.clip_len,
                "stride": self.cfg.stride,
                "root_joint": self.cfg.root_joint,
                "unit_scale": self.cfg.unit_scale,
                "seed": self.cfg.seed,
            },
            "metrics": serializable,
        }

        manifest_path = out_dir / "results.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        summary_path = out_dir / "results.txt"
        with open(summary_path, "w") as f:
            f.write(summarize_metrics(report))
            f.write("\n")

        return report

    def train(
        self,
        script: Union[str, Path],
        base_args: Union[str, List[str]],
        seed: int,
        output: Union[str, Path],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run a training script with a fixed seed and output checkpoint path.

        Args:
            script: Path to the training script.
            base_args: Extra arguments passed to the script.  May be a string or a
                list of strings.
            seed: Random seed for this run.
            output: Path where the checkpoint is expected to be written.
            dry_run: If ``True``, only build and return the command without running.

        Returns:
            Manifest entry dict with ``path``, ``seed``, ``status`` and ``command``.
        """
        script = Path(script)
        output = Path(output)

        if isinstance(base_args, str):
            args_list = base_args.split() if base_args.strip() else []
        else:
            args_list = list(base_args)

        command = [
            sys.executable,
            str(script),
            *args_list,
            "--seed",
            str(seed),
            "--output",
            str(output),
        ]

        entry = {
            "path": str(output),
            "seed": seed,
            "status": "dry_run" if dry_run else "pending",
            "command": command,
        }

        if dry_run:
            return entry

        result = subprocess.run(command, capture_output=False, text=False)
        entry["status"] = "completed" if result.returncode == 0 else "failed"
        entry["returncode"] = result.returncode
        return entry

    def run_multi_seed(
        self,
        script: Union[str, Path],
        base_args: Union[str, List[str]],
        seeds: Iterable[int],
        out_dir: Union[str, Path],
        base_name: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Train with multiple seeds and write a manifest.

        Args:
            script: Training script path.
            base_args: Shared arguments for every seed run.
            seeds: Sequence of seeds.
            out_dir: Directory for per-seed checkpoints and manifest.
            base_name: Prefix for checkpoint file names.
            dry_run: If ``True``, do not execute any training command.

        Returns:
            Manifest dictionary containing ``script``, ``base_args``, ``seeds`` and
            per-seed checkpoint entries under ``checkpoints``.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(base_args, str):
            base_args_str = base_args
        else:
            base_args_str = " ".join(str(a) for a in base_args)

        manifest: Dict[str, Any] = {
            "script": str(script),
            "base_args": base_args_str,
            "seeds": list(seeds),
            "checkpoints": {},
        }

        for seed in seeds:
            checkpoint = out_dir / f"{base_name}_seed{seed}.pth"
            entry = self.train(
                script=script,
                base_args=base_args,
                seed=seed,
                output=checkpoint,
                dry_run=dry_run,
            )
            manifest["checkpoints"][str(seed)] = entry

        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest


__all__ = [
    "BenchmarkConfig",
    "BenchmarkProtocol",
]
