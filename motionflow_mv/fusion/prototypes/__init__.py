"""Prototypes for future fusion modules.

Modules here are experimental and should not be imported by production code
without explicit review.
"""

from .cross_view_graph_attention import (
    CrossViewGraphAttention,
    CrossViewGraphAttentionLayer,
)

__all__ = [
    "CrossViewGraphAttention",
    "CrossViewGraphAttentionLayer",
]
