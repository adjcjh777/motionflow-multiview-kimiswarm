"""Anchor model with a dataset-conditional canonical-skeleton residual refiner.

Subclasses ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``
and replaces its dense residual MLP with a graph-based
``CanonicalSkeletonResidualRefiner`` that carries a per-dataset embedding and
a shared skeleton graph.  The forward signature adds an optional
``dataset_ids`` argument so the model can be trained on the mixed MPI +
Human3.6M + AIST++ loader.
"""

import torch

from .canonical_skeleton_residual_refiner import CanonicalSkeletonResidualRefiner
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCanonicalSkeleton(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view temporal residual model with a canonical skeleton prior.

    Parameters
    ----------
    num_datasets:
        Number of datasets in the mixed training set (default 3).
    dataset_embed_dim:
        Dimension of the learnable per-dataset embedding (default 16).
    graph_num_layers:
        Number of graph message-passing layers in the residual refiner
        (default 2).
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining arguments.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
        return_visibility: bool = False,
        return_raw: bool = False,
        num_datasets: int = 3,
        dataset_embed_dim: int = 16,
        graph_num_layers: int = 2,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_st_layers=n_st_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            principal_point_hidden=principal_point_hidden,
            principal_point_max_offset=principal_point_max_offset,
            focal_max_scale=focal_max_scale,
            return_pp_delta=return_pp_delta,
            return_visibility=return_visibility,
            return_raw=return_raw,
        )
        self.num_datasets = num_datasets
        self.dataset_embed_dim = dataset_embed_dim
        self.graph_num_layers = graph_num_layers

        # Replace the dense residual MLP with the graph-based canonical
        # skeleton refiner.  The parent residual_mlp is left untouched so that
        # the inherited __init__ still allocates it, but this model uses
        # self.residual_refiner instead.
        self.residual_refiner = CanonicalSkeletonResidualRefiner(
            j=j,
            in_dim=d + 3,
            residual_hidden=residual_hidden,
            num_datasets=num_datasets,
            dataset_embed_dim=dataset_embed_dim,
            graph_num_layers=graph_num_layers,
        )

    def forward(
        self,
        x,
        cameras=None,
        K=None,
        R=None,
        t=None,
        dataset_ids=None,
    ):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from .ray_attention_temporal_crossview_model import _cameras_to_tensors

            K, R, t = _cameras_to_tensors(cameras, device)

        # Prepare per-sample camera tensors and flatten time into batch.
        if K.dim() == 3:
            K = K.unsqueeze(0).expand(B * T, -1, -1, -1)
            R = R.unsqueeze(0).expand(B * T, -1, -1, -1)
            t = t.unsqueeze(0).expand(B * T, -1, -1)
        elif K.dim() == 4:
            K = K.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            R = R.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            t = t.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, V, 3)
        else:
            raise ValueError("K must have shape (V, 3, 3) or (B, V, 3, 3)")

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        # Principal-point / intrinsic correction before ray embedding.
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Per-frame v3 features (uses corrected intrinsics).
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Spatio-temporal (time + view) attention.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        # (B, J, T, V, d) -> (B*J, T*V, d)
        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Optional visibility-aware weighting (base returns 1).
        visibility = self._visibility_multiplier(feat, confidences)  # (B*T, V, J)

        # Per-frame weight prediction and triangulation with corrected intrinsics.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * visibility  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        from .ray_attention_model import _triangulate_weighted_dlt

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Canonical-skeleton residual refinement.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)  # (B*T, J, d+3)

        if dataset_ids is not None:
            dataset_ids_exp = dataset_ids.unsqueeze(1).expand(B, T).reshape(-1)
        else:
            dataset_ids_exp = None
        delta = self.residual_refiner(residual_input, dataset_ids_exp)  # (B*T, J, 3)
        pred_3d = pred_3d_raw + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        if self.return_visibility:
            visibility = visibility.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            if self.return_visibility:
                visibility = visibility.squeeze(1)

        if self.return_pp_delta:
            out = [pred_3d, weights, pp_delta]
            if self.correct_focal:
                out.insert(3, focal_scale)
            if self.return_raw:
                out.append(pred_3d_raw.view(B, T, J, 3))
            return tuple(out)
        if self.return_visibility:
            return pred_3d, weights, visibility
        if self.return_raw:
            return pred_3d, weights, pred_3d_raw.view(B, T, J, 3)
        return pred_3d, weights


if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, V, J = 2, 5, 4, 17
    x = torch.randn(B, T, V, J, 3)
    # Make confidences positive.
    x[..., 2] = torch.rand(B, T, V, J)
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)
    K[:, :, 0, 0] = 800.0
    K[:, :, 1, 1] = 800.0
    K[:, :, 0, 2] = 320.0
    K[:, :, 1, 2] = 240.0
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)
    t = torch.randn(B, V, 3)
    dataset_ids = torch.randint(0, 3, (B,))

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCanonicalSkeleton(
        j=J, d=32, n_views=V, residual_hidden=64, principal_point_max_offset=10.0
    )
    pred, w = model(x, K=K, R=R, t=t, dataset_ids=dataset_ids)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    loss = pred.mean()
    loss.backward()
    print("model forward/backward smoke test passed")
