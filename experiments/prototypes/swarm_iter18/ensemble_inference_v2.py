"""P09 Ensemble inference v2 for swarm_iter18.

This script improves on ``experiments/prototypes/eval_ensemble_checkpoints.py``
by adding **uncertainty-aware aggregation** across multiple checkpoints of the
Bayesian triangulation v2 family.  In addition to a plain uniform average, it
supports per-joint inverse-variance weighting using the predicted image-space
covariances, robust median aggregation, and trimmed-mean outlier rejection.

Example
-------
    python experiments/prototypes/swarm_iter18/ensemble_inference_v2.py \
        --model bayesian_tri_v2_pp \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --checkpoint outputs/bayesian_tri_v2_large_scale_mpiinf3dhp_seed0.pth \
        --checkpoint outputs/bayesian_tri_v2_large_scale_mpiinf3dhp_seed1.pth \
        --checkpoint outputs/bayesian_tri_v2_large_scale_mpiinf3dhp_seed2.pth \
        --clip_len 13 --d 128 --residual_hidden 256 --n_st_layers 3 \
        --val_stride 50 \
        --strategy inverse_variance \
        --output_json outputs/ensemble_v2_mpiinf3dhp.json \
        --output_npz outputs/ensemble_v2_mpiinf3dhp.npz

Notes
-----
- The aggregation is performed on the **3-D pose predictions** produced by each
  checkpoint.  When ``--strategy inverse_variance`` is used, the per-checkpoint
  predicted image-space covariance determinants are converted to per-joint
  precision weights.
- The script also computes an **epistemic uncertainty** estimate (variance
  across ensemble members) and an coarse **aleatoric uncertainty** estimate
  (mean predicted covariance magnitude).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics, summarize_metrics
from motionflow_mv.fusion.prototypes.ensemble_predictor import (
    MultiCheckpointEnsemble,
)
from experiments.eval_full_metrics import (
    MODEL_CLASSES,
    TemporalClipDataset,
    build_model,
    collate_fn,
)


class BayesianTriV2Ensemble(nn.Module):
    """Multi-checkpoint ensemble with uncertainty-aware aggregation.

    Parameters
    ----------
    build_fn:
        Callable returning a fresh model instance.  Invoked once per checkpoint.
    checkpoint_paths:
        Paths to checkpoints.  Order matters when ``weights`` is supplied.
    device:
        Target device for every sub-model.
    weights:
        Optional positive global weights for each checkpoint (used by
        ``uniform``/``trimmed_mean`` only).  Normalised to sum to one.
    strategy:
        One of ``uniform``, ``inverse_variance``, ``robust_median``,
        ``trimmed_mean``.
    return_covariance:
        Whether member models should return their predicted image-space
        covariances.  Required for ``inverse_variance``.
    trim_alpha:
        Fraction of extreme members to drop for ``trimmed_mean`` (default 0.1).
    """

    _VALID_STRATEGIES = {"uniform", "inverse_variance", "robust_median", "trimmed_mean"}

    def __init__(
        self,
        build_fn: Callable[[], nn.Module],
        checkpoint_paths: Sequence[str],
        device: Union[str, torch.device] = "cpu",
        weights: Optional[Sequence[float]] = None,
        strategy: str = "uniform",
        return_covariance: bool = True,
        trim_alpha: float = 0.1,
    ):
        super().__init__()
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose from {self._VALID_STRATEGIES}."
            )
        if strategy == "inverse_variance" and not return_covariance:
            raise ValueError("inverse_variance strategy requires return_covariance=True.")
        if not checkpoint_paths:
            raise ValueError("At least one checkpoint path is required.")

        self.device = torch.device(device)
        self.strategy = strategy
        self.return_covariance = return_covariance
        self.trim_alpha = trim_alpha

        models: List[nn.Module] = []
        for path in checkpoint_paths:
            model = build_fn().to(self.device)
            # Enable covariance output for uncertainty-aware strategies.
            if return_covariance and hasattr(model, "return_covariance"):
                model.return_covariance = True
            state = torch.load(path, map_location=self.device, weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"Warning: missing keys in {path}: {missing[:5]}")
            if unexpected:
                print(f"Warning: unexpected keys in {path} (ignored): {unexpected[:5]}")
            model.eval()
            models.append(model)

        self.models = nn.ModuleList(models)

        if weights is not None:
            weights_t = torch.as_tensor(
                list(weights), dtype=torch.float32, device=self.device
            )
            if len(weights_t) != len(self.models):
                raise ValueError("Number of weights must match number of checkpoints.")
            if weights_t.sum().item() <= 0:
                raise ValueError("Sum of ensemble weights must be positive.")
            self.register_buffer("weights", weights_t / weights_t.sum())
        else:
            self.register_buffer(
                "weights",
                torch.ones(len(self.models), device=self.device) / len(self.models),
            )

    def _extract_covariance(self, out: Tuple) -> Optional[torch.Tensor]:
        """Extract the 2x2 Cholesky factor ``L`` from a Bayesian tri v2 output tuple.

        The expected tuple layout (when ``return_covariance=True``) is:
        ``(pred_3d, weights, pp_delta, L, epi_loss)``.  If ``L`` is not present,
        ``None`` is returned.
        """
        if not self.return_covariance:
            return None
        # The covariance tensor has shape (..., 2, 2); other tuple elements are
        # scalar/loss or have fewer dims.
        for tensor in out:
            if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                if tensor.shape[-2:] == (2, 2):
                    return tensor
        return None

    def _precision_from_cholesky(self, L: torch.Tensor) -> torch.Tensor:
        """Compute per-joint precision from a batch of 2x2 Cholesky factors.

        Args:
            L: (B, T, V, J, 2, 2) lower-triangular Cholesky factor of the
               image-space covariance.

        Returns:
            precision: (B, T, J) mean per-joint precision across views.
        """
        # L[..., 0, 0] and L[..., 1, 1] are the diagonal entries.
        det = (L[..., 0, 0] * L[..., 1, 1]) ** 2  # (B, T, V, J)
        # Add a small epsilon to avoid division by zero.
        precision = 1.0 / (det.clamp(min=1e-8))
        # Aggregate precision across the view dimension.
        return precision.mean(dim=2)  # (B, T, J)

    def forward(
        self, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run each member and aggregate predictions.

        Returns
        -------
        ensemble_pred:
            Averaged 3-D pose tensor of shape ``(B, T, J, 3)``.
        epistemic_var:
            (optional) Per-joint variance across ensemble members, returned only
            when ``return_epistemic=True`` is passed in ``kwargs``.
        """
        return_epistemic = kwargs.pop("return_epistemic", False)

        preds = []
        precisions = [] if self.strategy == "inverse_variance" else None
        for model in self.models:
            with torch.no_grad():
                out = model(*args, **kwargs)
            if isinstance(out, (list, tuple)):
                pred = out[0]
                if self.strategy == "inverse_variance":
                    L = self._extract_covariance(out)
                    if L is None:
                        raise RuntimeError(
                            "inverse_variance requested but member model does not "
                            "return covariance. Ensure the model class supports "
                            "return_covariance=True."
                        )
                    precisions.append(self._precision_from_cholesky(L))
            else:
                pred = out
            preds.append(pred)

        stacked = torch.stack(preds, dim=0)  # (M, B, T, J, 3)

        if self.strategy == "uniform" or self.strategy == "robust_median":
            if self.strategy == "uniform":
                weights = self.weights.view(-1, *([1] * (stacked.dim() - 1)))
                ensemble = (stacked * weights).sum(dim=0)
            else:  # robust_median
                ensemble = stacked.median(dim=0).values
        elif self.strategy == "inverse_variance":
            # precision: (M, B, T, J)
            precision_stack = torch.stack(precisions, dim=0)  # (M, B, T, J)
            # Normalise across ensemble members.
            w = precision_stack / precision_stack.sum(dim=0, keepdim=True)
            w = w.unsqueeze(-1)  # (M, B, T, J, 1), broadcasts over xyz
            ensemble = (stacked * w).sum(dim=0)
        elif self.strategy == "trimmed_mean":
            # Drop the lowest and highest predictions per joint along the ensemble
            # dimension, then uniform average of the remainder.
            alpha = self.trim_alpha
            if alpha <= 0 or alpha >= 0.5:
                raise ValueError("trim_alpha must be in (0, 0.5).")
            M = stacked.size(0)
            drop = max(1, int(M * alpha)) if M > 2 else 0
            if drop > 0:
                # Sort along ensemble dimension by coordinate value.
                sorted_stack, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_stack[drop : M - drop]
            else:
                trimmed = stacked
            ensemble = trimmed.mean(dim=0)
        else:
            raise ValueError(f"Unhandled strategy {self.strategy!r}")

        if not return_epistemic:
            return ensemble

        # Epistemic uncertainty = variance across ensemble predictions.
        epistemic_var = stacked.var(dim=0)  # (B, T, J, 3)
        return ensemble, epistemic_var


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(MODEL_CLASSES), required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        action="append",
        required=True,
        help="Path to a model checkpoint; can be repeated.",
    )
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--n_view_layers", type=int, default=2)
    parser.add_argument("--n_view_groups", type=int, default=2)
    parser.add_argument("--n_joint_graph_layers", type=int, default=1)
    parser.add_argument("--no_skeleton_graph", action="store_true")
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--target_k", type=int, default=4)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--gt_scale", type=float, default=1.0)
    parser.add_argument("--camera_scale", type=float, default=1.0)
    parser.add_argument("--parents", type=str, default=None)
    parser.add_argument("--symmetry_pairs", type=str, default=None)
    parser.add_argument("--source_n_views", type=int, default=None)
    parser.add_argument(
        "--strategy",
        type=str,
        default="inverse_variance",
        choices=["uniform", "inverse_variance", "robust_median", "trimmed_mean"],
        help="Ensemble aggregation strategy.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-checkpoint weights (used by uniform/trimmed_mean).",
    )
    parser.add_argument(
        "--trim_alpha",
        type=float,
        default=0.1,
        help="Trim fraction for trimmed_mean strategy.",
    )
    parser.add_argument(
        "--return_covariance",
        action="store_true",
        default=True,
        help="Force member models to return covariance (default True).",
    )
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--output_npz",
        type=str,
        default=None,
        help="If provided, save predictions, GT, and uncertainties to an .npz.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a lightweight CPU smoke test instead of full evaluation.",
    )
    return parser.parse_args()


def run_smoke_test():
    """Lightweight CPU smoke test using random tiny checkpoints.

    Verifies that ``BayesianTriV2Ensemble`` can be instantiated, loaded, and run
    with each supported strategy.
    """
    import tempfile

    from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
        RayAttentionFusionModelBayesianTriV2,
    )

    device = torch.device("cpu")
    V, J, d = 4, 17, 16
    checkpoint_paths = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for seed in (0, 1):
            model = RayAttentionFusionModelBayesianTriV2(
                j=J,
                d=d,
                n_views=V,
                n_st_layers=1,
                residual_hidden=32,
                return_covariance=True,
            )
            model.eval()
            path = Path(tmpdir) / f"ckpt_{seed}.pth"
            torch.save(model.state_dict(), path)
            checkpoint_paths.append(str(path))

        def build_fn():
            return RayAttentionFusionModelBayesianTriV2(
                j=J,
                d=d,
                n_views=V,
                n_st_layers=1,
                residual_hidden=32,
                return_covariance=True,
            )

        B, T = 2, 3
        x = torch.rand(B, T, V, J, 3)
        K = torch.eye(3).unsqueeze(0).expand(V, -1, -1).float()
        R = torch.eye(3).unsqueeze(0).expand(V, -1, -1).float()
        t = torch.zeros(V, 3).float()

        for strategy in BayesianTriV2Ensemble._VALID_STRATEGIES:
            ensemble = BayesianTriV2Ensemble(
                build_fn,
                checkpoint_paths,
                device=device,
                strategy=strategy,
                return_covariance=True,
                trim_alpha=0.1,
            )
            pred, epistemic = ensemble(x, K=K, R=R, t=t, return_epistemic=True)
            assert pred.shape == (B, T, J, 3), pred.shape
            assert epistemic.shape == (B, T, J, 3), epistemic.shape
            print(f"Smoke test passed for strategy={strategy}")


def main():
    args = parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    dataset = TemporalClipDataset(
        args.dataset,
        args.clip_len,
        stride=args.val_stride,
        gt_scale=args.gt_scale,
        camera_scale=args.camera_scale,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    source_n_views = args.source_n_views if args.source_n_views is not None else n_views

    def build_fn():
        return build_model(args, source_n_views, j)

    ensemble = BayesianTriV2Ensemble(
        build_fn,
        args.checkpoint,
        device=device,
        weights=args.weights,
        strategy=args.strategy,
        return_covariance=args.return_covariance,
        trim_alpha=args.trim_alpha,
    )

    preds, gts, epistemic_vars = [], [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, epi_var = ensemble(xb, K=K, R=R, t=t, return_epistemic=True)
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())
            epistemic_vars.append(epi_var.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    epistemic_vars = np.concatenate(epistemic_vars, axis=0)

    preds_flat = preds.reshape(-1, preds.shape[-2], 3)
    gts_flat = gts.reshape(-1, gts.shape[-2], 3)

    parents_arr = None
    if args.parents:
        from experiments.eval_full_metrics import _load_list
        parents_arr = np.array(_load_list(args.parents, int), dtype=np.int64)

    report = compute_all_metrics(preds_flat * 1000.0, gts_flat * 1000.0, parents=parents_arr)
    print(summarize_metrics(report))
    print(f"MPJPE: {report['mpjpe']:.2f} mm")
    print(f"PA-MPJPE: {report['pa_mpjpe']:.2f} mm")
    print(f"PCK@50: {report['pck@50mm']:.4f}")
    print(f"PCK@100: {report['pck@100mm']:.4f}")
    print(f"PCK@150: {report['pck@150mm']:.4f}")
    print(f"PCK-AUC: {report['pck_auc']:.4f}")
    print(f"Mean epistemic std: {np.sqrt(epistemic_vars).mean():.4f} m")

    if args.output_json:
        serializable = {}
        for k, v in report.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = float(v)
        serializable["strategy"] = args.strategy
        serializable["n_members"] = len(args.checkpoint)
        serializable["mean_epistemic_std_m"] = float(np.sqrt(epistemic_vars).mean())
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved metrics to {out_path}")

    if args.output_npz:
        out_path = Path(args.output_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            pred_3d=preds,
            gt_3d=gts,
            epistemic_var=epistemic_vars,
            strategy=args.strategy,
        )
        print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    if "--smoke_test" in sys.argv:
        run_smoke_test()
    else:
        main()
