"""v31 domain-balanced sampling wrapper for the v5 WebBridge trainer.

This script monkey-patches ``build_webbridge_mixed_dataloaders`` so that
the training DataLoader uses :class:`DomainBalancedSampler`, then delegates
to the standard ``train_omniview_fusion_v5_webbridge_multi.py``.  No
existing source files are modified.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from motionflow_mv.data import webbridge_mixed_dataset
from motionflow_mv.data.domain_balanced_sampler import DomainBalancedSampler


_original_build_dataloaders = webbridge_mixed_dataset.build_webbridge_mixed_dataloaders


def build_webbridge_mixed_dataloaders_with_balanced(
    train_paths,
    train_names,
    val_paths,
    val_names,
    clip_len,
    batch_size,
    train_samples=200,
    val_stride=1,
    num_workers=0,
    return_view_mask=False,
):
    """Build mixed WebBridge loaders with domain-balanced training sampling."""
    train_dataset = webbridge_mixed_dataset.WebBridgeMixedDataset(
        train_paths,
        train_names,
        clip_len,
        n_samples=train_samples,
        return_view_mask=return_view_mask,
    )
    val_dataset = webbridge_mixed_dataset.WebBridgeMixedDataset(
        val_paths,
        val_names,
        clip_len,
        n_samples=None,
        stride=val_stride,
        return_view_mask=return_view_mask,
    )

    collate_fn = (
        webbridge_mixed_dataset.webbridge_mixed_collate_fn_with_mask
        if return_view_mask
        else webbridge_mixed_dataset.webbridge_mixed_collate_fn
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=DomainBalancedSampler(train_dataset, domain_names=train_names, seed=42),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )
    return train_loader, val_loader


# Patch the mixed-dataset loader *before* the base training script imports it.
webbridge_mixed_dataset.build_webbridge_mixed_dataloaders = build_webbridge_mixed_dataloaders_with_balanced

# Import the standard v5 trainer after patching.
import experiments.train_omniview_fusion_v5_webbridge_multi as base  # noqa: E402

if __name__ == "__main__":
    base.main()
