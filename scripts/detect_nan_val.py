"""Detect which submodule first emits NaN/Inf during a validation batch."""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import torch

# Allow importing the training script as a module.
EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

try:
    from train_omniview_fusion_v5_webbridge_multi import (
        build_datasets,
        build_model_from_args,
        webbridge_mixed_collate_fn,
    )
except Exception as exc:  # pragma: no cover
    print(f"Could not import training script: {exc}")
    raise


def load_args(config_path: str):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    args = argparse.Namespace(**cfg)
    # Some fields may be lists that were serialised as lists; keep them.
    return args


def main():
    import argparse as _argparse  # local import to avoid shadowing

    # Parse simple CLI.
    parser = _argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to .config.json")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--max_batches", type=int, default=1)
    parser.add_argument("--skip_modules", default="", help="Comma-separated module names to skip hooking")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    train_args = load_args(str(cfg_path))
    # Windows/WSL safety.
    if os.name == "nt":
        train_args.num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build datasets and val loader.
    _, val_dataset, n_views, n_joints = build_datasets(train_args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_args.batch_size,
        shuffle=False,
        collate_fn=webbridge_mixed_collate_fn,
        num_workers=0,
    )

    # Build model and load checkpoint.
    model = build_model_from_args(train_args, n_joints, n_views, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint {ckpt_path}")

    skip_set = set(name.strip() for name in args.skip_modules.split(",") if name.strip())

    # Register hooks that print the first NaN/Inf source.
    first_bad = {"found": False}

    def make_hook(name):
        def hook(module, input, output):
            if first_bad["found"]:
                return
            # Handle tuple/list outputs.
            outs = output if isinstance(output, (tuple, list)) else (output,)
            for i, out in enumerate(outs):
                if isinstance(out, torch.Tensor) and not torch.isfinite(out).all():
                    first_bad["found"] = True
                    has_nan = torch.isnan(out).any().item()
                    has_inf = torch.isinf(out).any().item()
                    print(
                        f"FIRST BAD: {name} output[{i}] "
                        f"shape={tuple(out.shape)} nan={has_nan} inf={has_inf}"
                    )
                    # Print summary of where the bad values are.
                    if out.numel() > 0:
                        bad_mask = ~torch.isfinite(out)
                        bad_idx = torch.nonzero(bad_mask.view(-1), as_tuple=False)[:5].flatten().tolist()
                        print(f"  sample bad flat indices: {bad_idx}")
                    break
        return hook

    handles = []
    for name, module in model.named_modules():
        if name in skip_set:
            continue
        handle = module.register_forward_hook(make_hook(name))
        handles.append(handle)

    print(f"Registered {len(handles)} hooks")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= args.max_batches:
                break
            print(f"\n--- Batch {batch_idx} ---")
            first_bad["found"] = False
            if len(batch) == 6:
                x, y, K, R, t, dataset_id = batch
                forward_kwargs = {"domain_id": dataset_id.to(device)}
            else:
                x, y, K, R, t = batch
                forward_kwargs = {}
            x = x.to(device)
            y = y.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)
            try:
                out = model(x, K=K, R=R, t=t, **forward_kwargs)
                pred = out[0]
                print(
                    f"pred finite={torch.isfinite(pred).all().item()} "
                    f"nan={torch.isnan(pred).any().item()} inf={torch.isinf(pred).any().item()}"
                )
            except Exception as exc:
                print(f"Forward raised exception: {exc}")
                import traceback
                traceback.print_exc()

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
