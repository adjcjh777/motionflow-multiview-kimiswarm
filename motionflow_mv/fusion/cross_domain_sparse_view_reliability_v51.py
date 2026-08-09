"""v51 Cross-Domain Sparse-View Reliability (CDSVR).

A lightweight, identity-at-init cross-attention block that makes the v50
Self-Evolution Feedback Head domain-aware.  It consumes per-view reliability
``r ∈ R^V`` and per-joint log-variance ``σ ∈ R^J`` together with a domain
embedding, and outputs a domain-conditioned reliability offset
``Δr ∈ R^V`` and a per-joint uncertainty rescale ``α ∈ R^J``.

The update is applied as ``r' = r + Δr`` and ``σ' = σ / α``.  At init,
``Δr = 0`` and ``α = 1``, so enabling the module does not perturb the
already-trained v50 SEFH baseline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CrossDomainSparseViewReliabilityV51(nn.Module):
    """Domain-conditioned refinement of sparse-view reliability and uncertainty.

    Parameters
    ----------
    n_views:
        Number of camera views ``V``.
    n_joints:
        Number of joints ``J``.
    hidden:
        Hidden dimension of the cross-attention block.
    num_heads:
        Number of attention heads.
    dropout:
        Dropout probability inside the cross-attention layers.
    offset_min:
        Minimum positive floor for the refined reliability weight
        ``w' = sigmoid(r' / τ)`` used in the auxiliary loss.
    use_domain_label:
        If ``True`` and ``domain_id`` is passed, embed it with a learned
        embedding.  Otherwise expect an externally provided ``domain_emb``.
    uncertainty_temperature:
        Temperature for the refined reliability sigmoid in the auxiliary loss.
    identity_init_gate:
        If ``True``, the final projection layers are zero-initialized so the
        module is identity at startup.
    num_domains:
        Number of distinct domain labels (used when ``domain_id`` is supplied).
    """

    def __init__(
        self,
        n_views: int = 4,
        n_joints: int = 17,
        hidden: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        offset_min: float = 0.05,
        use_domain_label: bool = True,
        uncertainty_temperature: float = 1.0,
        identity_init_gate: bool = True,
        num_domains: int = 6,
    ) -> None:
        super().__init__()
        self.n_views = n_views
        self.n_joints = n_joints
        self.hidden = hidden
        self.num_heads = num_heads
        self.offset_min = offset_min
        self.use_domain_label = use_domain_label
        self.uncertainty_temperature = uncertainty_temperature
        self.identity_init_gate = identity_init_gate

        # Domain embedding (learned) used when an external embedding is not given.
        self.num_domains = num_domains
        self.domain_embed = nn.Embedding(num_domains, hidden)

        # Learned positional embeddings for views and joints.
        self.view_pos_embed = nn.Parameter(torch.randn(n_views, hidden) * 0.02)
        self.joint_pos_embed = nn.Parameter(torch.randn(n_joints, hidden) * 0.02)

        # Input projections: scalar reliability / log-variance -> hidden.
        self.view_in_proj = nn.Linear(1, hidden)
        self.joint_in_proj = nn.Linear(1, hidden)

        # Two-layer cross-attention block.
        self.cross_attn_layer1 = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_layer2 = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        # Output heads.
        self.reliability_offset_head = nn.Linear(hidden, 1)
        self.uncertainty_scale_head = nn.Linear(hidden, 1)

        # Identity-at-init: zero-initialize final layers.
        if identity_init_gate:
            nn.init.zeros_(self.reliability_offset_head.weight)
            nn.init.zeros_(self.reliability_offset_head.bias)
            nn.init.zeros_(self.uncertainty_scale_head.weight)
            nn.init.zeros_(self.uncertainty_scale_head.bias)

    def _get_domain_emb(
        self,
        domain_emb: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Return a ``(B, hidden)`` domain embedding vector."""
        if domain_emb is not None:
            return domain_emb
        if domain_id is not None:
            return self.domain_embed(domain_id.long().view(-1))
        # No domain info: use a learned fallback (zero-initialized, trainable).
        return self.domain_embed(torch.zeros(batch_size, dtype=torch.long, device=self.domain_embed.weight.device))

    def forward(
        self,
        reliability: torch.Tensor,
        log_var: torch.Tensor,
        domain_emb: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine reliability and uncertainty given domain information.

        Parameters
        ----------
        reliability:
            Per-view reliability ``(B, V)`` from v50 SEFH.
        log_var:
            Per-joint log-variance ``(B, J)`` from v50 SEFH.
        domain_emb:
            Optional external domain embedding ``(B, d)``.
        domain_id:
            Optional integer domain label ``(B,)``.

        Returns
        -------
        reliability_offset:
            ``(B, V)`` additive offset, clamped to ``[-2, 2]``.
        uncertainty_scale:
            ``(B, J)`` positive multiplicative rescale, initialized near 1.
        """
        B = reliability.shape[0]
        V = self.n_views
        J = self.n_joints
        device = reliability.device

        dom_emb = self._get_domain_emb(domain_emb, domain_id)  # (B, hidden)

        # Build view tokens: (B, V, hidden)
        rel_in = reliability.unsqueeze(-1)  # (B, V, 1)
        view_tokens = self.view_in_proj(rel_in) + self.view_pos_embed.unsqueeze(0)
        view_tokens = view_tokens + dom_emb.unsqueeze(1)

        # Build joint tokens: (B, J, hidden)
        log_var_in = log_var.unsqueeze(-1)  # (B, J, 1)
        joint_tokens = self.joint_in_proj(log_var_in) + self.joint_pos_embed.unsqueeze(0)
        joint_tokens = joint_tokens + dom_emb.unsqueeze(1)

        # Layer 1: views attend to joints.
        view_tokens2, _ = self.cross_attn_layer1(
            query=view_tokens, key=joint_tokens, value=joint_tokens
        )
        view_tokens2 = self.norm1(view_tokens2 + view_tokens)

        # Layer 2: joints attend to views (using updated view tokens).
        joint_tokens2, _ = self.cross_attn_layer2(
            query=joint_tokens, key=view_tokens2, value=view_tokens2
        )
        joint_tokens2 = self.norm2(joint_tokens2 + joint_tokens)

        # Predict reliability offset per view.
        reliability_offset = self.reliability_offset_head(view_tokens2).squeeze(-1)  # (B, V)
        reliability_offset = reliability_offset.clamp(-2.0, 2.0)

        # Predict uncertainty rescale per joint (positive, identity-at-init ≈ 1).
        uncertainty_scale = self.uncertainty_scale_head(joint_tokens2).squeeze(-1)  # (B, J)
        uncertainty_scale = torch.exp(uncertainty_scale).clamp(min=1e-3)

        return reliability_offset, uncertainty_scale


if __name__ == "__main__":
    B, V, J = 4, 4, 17
    reliability = torch.rand(B, V)
    log_var = torch.randn(B, J)
    domain_id = torch.randint(0, 6, (B,))

    module = CrossDomainSparseViewReliabilityV51(n_views=V, n_joints=J)
    offset, scale = module(reliability, log_var, domain_id=domain_id)
    assert offset.shape == (B, V)
    assert scale.shape == (B, J)
    assert (scale >= 1e-3).all()
    assert offset.abs().max().item() <= 2.0 + 1e-6
    print("CrossDomainSparseViewReliabilityV51 CPU smoke test passed")
