"""Domain-balanced sampler for WebBridge mixed datasets.

A typical :class:`torch.utils.data.ConcatDataset` with uniform shuffling
samples domains in proportion to their number of clips.  This sampler
rebalances training so that each domain contributes equally to every
epoch, resampling smaller domains with replacement when necessary.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import Sampler


class DomainBalancedSampler(Sampler):
    """Balanced sampler over a :class:`ConcatDataset` of WebBridge domains.

    The sampler assumes the underlying dataset is a
    :class:`torch.utils.data.ConcatDataset` whose inner datasets expose a
    ``dataset_name`` attribute (e.g. ``"h36m"``, ``"mpi"``).  Each epoch is
    built by round-robin sampling across domains, so every batch contains
    an approximately equal mix of domains.  Domains with fewer samples are
    resampled with replacement to match the largest domain.

    Parameters
    ----------
    concat_dataset:
        A ``ConcatDataset`` whose inner datasets each expose a
        ``dataset_name`` attribute.
    domain_names:
        Optional parallel sequence of domain names, one per inner dataset.
        If ``None``, names are inferred from ``concat_dataset.datasets``.
    seed:
        Random seed for reproducible shuffling.
    resample:
        If ``True`` (default), domains with fewer samples are resampled
        with replacement so that every epoch contains an equal number of
        samples from each domain.  The epoch length is therefore
        ``num_domains * max_domain_size``.  If ``False``, the epoch stops
        when the smallest domain is exhausted.
    """

    def __init__(
        self,
        concat_dataset: ConcatDataset,
        domain_names: Optional[Sequence[str]] = None,
        seed: int = 42,
        resample: bool = True,
    ):
        if not hasattr(concat_dataset, "datasets"):
            raise ValueError(
                "DomainBalancedSampler expects a torch ConcatDataset with a "
                "'datasets' attribute."
            )

        if domain_names is None:
            domain_names = [
                getattr(ds, "dataset_name", f"domain_{i}")
                for i, ds in enumerate(concat_dataset.datasets)
            ]

        self.concat_dataset = concat_dataset
        self.domain_names = list(domain_names)
        self.rng = np.random.RandomState(seed)
        self.resample = resample

        # Build per-domain list of global sample indices.
        self.domain_to_indices: dict[str, List[int]] = defaultdict(list)
        for d_idx, ds in enumerate(concat_dataset.datasets):
            name = self.domain_names[d_idx]
            start = 0 if d_idx == 0 else concat_dataset.cumulative_sizes[d_idx - 1]
            end = concat_dataset.cumulative_sizes[d_idx]
            self.domain_to_indices[name].extend(range(start, end))

        if not self.domain_to_indices:
            raise ValueError("DomainBalancedSampler found no domains.")

        self.domains = sorted(self.domain_to_indices.keys())
        self.max_domain_size = max(len(v) for v in self.domain_to_indices.values())

        # Epoch length: equal number of samples per domain.
        self._epoch_length = self.max_domain_size * len(self.domains)

    def __len__(self) -> int:
        return self._epoch_length

    def __iter__(self) -> Iterator[int]:
        # Shuffle each domain independently.
        per_domain: dict[str, List[int]] = {}
        for name in self.domains:
            indices = np.array(self.domain_to_indices[name], dtype=np.int64)
            self.rng.shuffle(indices)
            per_domain[name] = indices.tolist()

        domain_positions = {name: 0 for name in self.domains}

        for _ in range(self.max_domain_size):
            for name in self.domains:
                pos = domain_positions[name]
                if pos >= len(per_domain[name]):
                    if not self.resample:
                        continue
                    # Resample this domain with replacement and reset pointer.
                    per_domain[name] = self.rng.choice(
                        self.domain_to_indices[name],
                        size=len(self.domain_to_indices[name]),
                        replace=True,
                    ).tolist()
                    domain_positions[name] = 0
                    pos = 0
                yield per_domain[name][pos]
                domain_positions[name] = pos + 1
