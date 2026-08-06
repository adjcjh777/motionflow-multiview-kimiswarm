"""Confidence-aware per-joint view dropout with resampling."""

from typing import Optional

import torch


def _flatten_view_joint(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])


def confidence_resample_view_dropout(
    x: torch.Tensor,
    dropout_rate: float,
    resample: bool = True,
    confidence_channel: int = -1,
    min_views: int = 1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
    if dropout_rate == 0.0:
        return x.clone()
    if x.dim() < 3:
        raise ValueError(f"Input must have at least 3 dimensions, got shape {x.shape}")

    c = confidence_channel if confidence_channel >= 0 else x.shape[-1] + confidence_channel
    if c < 0 or c >= x.shape[-1]:
        raise ValueError(f"confidence_channel {confidence_channel} out of bounds for last dim {x.shape[-1]}")

    x = x.clone()
    V = x.shape[-3]
    if V < min_views:
        raise ValueError(f"min_views ({min_views}) cannot exceed number of views ({V})")

    flat = _flatten_view_joint(x)
    N = flat.shape[0]
    conf = flat[..., c].clamp(min=0.0)

    sum_conf = conf.sum(dim=1, keepdim=True).clamp(min=1e-6)
    target_kept = max(min_views, int(round(V * (1.0 - dropout_rate))))
    p_keep = (conf / sum_conf) * target_kept
    p_keep = p_keep.clamp(min=0.0, max=1.0)

    keep_mask = torch.rand(N, V, conf.shape[-1], generator=generator, device=flat.device) < p_keep
    keep_mask = keep_mask.to(flat.device)

    if min_views > 0:
        _, top_view_indices = conf.topk(min_views, dim=1)
        batch_idx = torch.arange(N, device=flat.device)[:, None, None]
        joint_idx = torch.arange(conf.shape[-1], device=flat.device)[None, None, :]
        keep_mask[batch_idx, top_view_indices, joint_idx] = True

    if not resample:
        flat[..., c][~keep_mask] = 0.0
        return x

    conf_kept = conf * keep_mask.float()
    conf_sum = conf_kept.sum(dim=1, keepdim=True).clamp(min=1e-6)
    probs = conf_kept / conf_sum

    dropped = ~keep_mask
    if not dropped.any():
        return x

    probs_nj_v = probs.permute(0, 2, 1).reshape(-1, V)
    samples = torch.multinomial(
        probs_nj_v,
        num_samples=V,
        replacement=True,
        generator=generator,
    )
    samples = samples.view(N, conf.shape[-1], V).permute(0, 2, 1)

    samples_expanded = samples.unsqueeze(-1).expand(-1, -1, -1, flat.shape[-1])
    gathered = torch.gather(flat, dim=1, index=samples_expanded)

    flat[dropped] = gathered[dropped]
    return x


class ConfidenceResampleDropout:
    def __init__(
        self,
        dropout_rate: float = 0.2,
        resample: bool = True,
        min_views: int = 1,
        seed: Optional[int] = None,
    ):
        self.dropout_rate = dropout_rate
        self.resample = resample
        self.min_views = min_views
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return confidence_resample_view_dropout(
            x,
            dropout_rate=self.dropout_rate,
            resample=self.resample,
            min_views=self.min_views,
            generator=self.generator,
        )

    def state_dict(self) -> dict:
        return {
            "dropout_rate": self.dropout_rate,
            "resample": self.resample,
            "min_views": self.min_views,
            "generator_state": self.generator.get_state().tolist(),
        }

    def load_state_dict(self, state: dict):
        self.dropout_rate = state["dropout_rate"]
        self.resample = state["resample"]
        self.min_views = state["min_views"]
        self.generator.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))
