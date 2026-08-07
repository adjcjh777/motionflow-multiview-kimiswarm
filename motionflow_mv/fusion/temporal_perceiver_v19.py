"""Temporal Perceiver — compress long video clips to a latent set and decode per-frame 3D poses.

This module is designed as a drop-in temporal refinement stage after an
initial per-frame pose estimator (e.g. the v5 omniview fusion model).  It
treats a sequence of per-frame joint features as a variable-length token
stream, compresses it with a small Perceiver encoder to a fixed-size latent
set, then decodes per-frame/per-joint 3D pose residuals with a Perceiver
decoder.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TemporalPerceiverRefiner(nn.Module):
    """Perceiver-based temporal refiner for long video clips.

    The model receives a sequence of per-frame joint feature tokens, compresses
    the whole clip into a small set of latent tokens, and decodes per-frame
    per-joint 3D pose corrections.

    Parameters
    ----------
    j:
        Number of joints per skeleton.
    in_dim:
        Dimension of the input feature for each joint in each frame.
    d:
        Hidden dimension used inside the perceiver.
    n_latents:
        Number of latent tokens the encoder compresses the clip to.
    n_layers:
        Number of perceiver encoder/decoder layers.
    n_heads:
        Number of attention heads.
    dropout:
        Dropout probability for attention and feed-forward layers.
    max_temporal_len:
        Maximum temporal length supported by the learned positional embeddings.
    """

    def __init__(
        self,
        j: int = 17,
        in_dim: int = 3,
        d: int = 64,
        n_latents: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
        max_temporal_len: int = 256,
    ):
        super().__init__()
        self.j = j
        self.d = d
        self.n_latents = n_latents
        self.n_layers = n_layers

        # Project raw input tokens (e.g. per-joint 3D pose or fused features) to
        # the perceiver dimension.
        self.input_proj = nn.Linear(in_dim, d)

        # Positional encodings.  Time and joint embeddings are added to each token
        # so the model can distinguish temporal order and joint identity.
        self.time_pos_embed = nn.Parameter(torch.randn(max_temporal_len, d) * 0.02)
        self.joint_pos_embed = nn.Parameter(torch.randn(j, d) * 0.02)

        # Encoder: a fixed set of latents summarises the whole temporal clip.
        self.latents = nn.Parameter(torch.randn(n_latents, d) * 0.02)

        self.encoder_self_attn = nn.ModuleList()
        self.encoder_cross_attn = nn.ModuleList()
        self.encoder_ffn = nn.ModuleList()
        self.encoder_norm_self = nn.ModuleList()
        self.encoder_norm_cross = nn.ModuleList()
        self.encoder_norm_ffn = nn.ModuleList()

        # Optional self-attention over the input tokens inside each encoder layer,
        # mirroring a standard Perceiver block.
        self.input_self_attn = nn.ModuleList()
        self.input_norm_self = nn.ModuleList()

        for _ in range(n_layers):
            self.encoder_self_attn.append(
                nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
            )
            self.encoder_cross_attn.append(
                nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
            )
            self.input_self_attn.append(
                nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
            )
            self.encoder_ffn.append(
                nn.Sequential(
                    nn.Linear(d, d * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d * 4, d),
                )
            )
            self.encoder_norm_self.append(nn.LayerNorm(d))
            self.encoder_norm_cross.append(nn.LayerNorm(d))
            self.encoder_norm_ffn.append(nn.LayerNorm(d))
            self.input_norm_self.append(nn.LayerNorm(d))

        # Decoder: per-frame, per-joint queries cross-attend to the latent set.
        self.decoder_query = nn.Parameter(
            torch.randn(max_temporal_len, j, d) * 0.02
        )
        self.decoder_cross_attn = nn.ModuleList()
        self.decoder_ffn = nn.ModuleList()
        self.decoder_norm_cross = nn.ModuleList()
        self.decoder_norm_ffn = nn.ModuleList()
        for _ in range(n_layers):
            self.decoder_cross_attn.append(
                nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
            )
            self.decoder_ffn.append(
                nn.Sequential(
                    nn.Linear(d, d * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d * 4, d),
                )
            )
            self.decoder_norm_cross.append(nn.LayerNorm(d))
            self.decoder_norm_ffn.append(nn.LayerNorm(d))

        # Final residual pose head.
        self.pose_head = nn.Linear(d, 3)

    def _add_positional_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Add time + joint positional embeddings to input tokens.

        x: (B, T, J, d)
        """
        B, T, J, d = x.shape
        time_emb = self.time_pos_embed[:T].view(1, T, 1, d)
        joint_emb = self.joint_pos_embed[:J].view(1, 1, J, d)
        return x + time_emb + joint_emb

    def forward(
        self,
        x: torch.Tensor,
        baseline_3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            ``(B, T, J, in_dim)`` input token sequence, e.g. per-joint 3D pose
            estimates or fused per-joint features.
        baseline_3d:
            Optional ``(B, T, J, 3)`` baseline pose to which the predicted
            residual is added.  If omitted, the module returns the raw 3D
            residual.

        Returns
        -------
            ``(B, T, J, 3)`` refined per-frame 3D poses.
        """
        if x.dim() != 4:
            raise ValueError(f"Input must be (B, T, J, in_dim), got {x.shape}")

        B, T, J, _ = x.shape

        # Project and add positional embeddings.
        x = self.input_proj(x)  # (B, T, J, d)
        x = self._add_positional_embeddings(x)

        # Flatten spatial/temporal tokens: (B, T*J, d).
        x_flat = x.view(B, T * J, self.d)

        # Initialise latents for the batch.
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        # Encoder: latents summarise the full video clip.
        for i in range(self.n_layers):
            # 1. Latent self-attention.
            self_out, _ = self.encoder_self_attn[i](latents, latents, latents)
            latents = self.encoder_norm_self[i](latents + self_out)

            # 2. Optional self-attention over input tokens.
            input_self_out, _ = self.input_self_attn[i](x_flat, x_flat, x_flat)
            x_flat = self.input_norm_self[i](x_flat + input_self_out)

            # 3. Cross-attention: latents query the input token set.
            cross_out, _ = self.encoder_cross_attn[i](latents, x_flat, x_flat)
            latents = self.encoder_norm_cross[i](latents + cross_out)

            # 4. Feed-forward on latents.
            ffn_out = self.encoder_ffn[i](latents)
            latents = self.encoder_norm_ffn[i](latents + ffn_out)

        # Decoder: per-frame, per-joint queries reconstruct per-frame poses.
        queries = (
            self.decoder_query[:T, :J, :].unsqueeze(0).expand(B, -1, -1, -1)
        )  # (B, T, J, d)
        # Add time + joint positional embeddings to queries.
        queries = queries + self._add_positional_embeddings(queries)
        queries = queries.view(B, T * J, self.d)

        for i in range(self.n_layers):
            cross_out, _ = self.decoder_cross_attn[i](queries, latents, latents)
            queries = self.decoder_norm_cross[i](queries + cross_out)
            ffn_out = self.decoder_ffn[i](queries)
            queries = self.decoder_norm_ffn[i](queries + ffn_out)

        # Reshape and predict residual 3D pose.
        queries = queries.view(B, T, J, self.d)
        residual = self.pose_head(queries)  # (B, T, J, 3)

        if baseline_3d is not None:
            if baseline_3d.shape != (B, T, J, 3):
                raise ValueError(
                    f"baseline_3d shape {baseline_3d.shape} incompatible with "
                    f"output shape {(B, T, J, 3)}"
                )
            return baseline_3d + residual

        return residual


if __name__ == "__main__":
    B, T, J = 2, 24, 17
    model = TemporalPerceiverRefiner(j=J, in_dim=3, d=64, n_latents=32, n_layers=2)
    x = torch.randn(B, T, J, 3)
    baseline = torch.randn(B, T, J, 3)
    out = model(x, baseline)
    assert out.shape == (B, T, J, 3)

    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("TemporalPerceiverRefiner CPU smoke test passed")
