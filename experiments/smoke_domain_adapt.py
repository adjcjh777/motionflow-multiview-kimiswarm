"""CPU smoke test for the GRL + FiLM domain-adaptation wrapper.

Builds a tiny synthetic multi-view clip and exercises
``motionflow_mv.models.domain_adaptation_wrapper.DomainAdaptationWrapper`` in
three modes:

1. Forward with source domain labels and domain logits.
2. Forward without domain labels (inference on unlabeled target).
3. Backward pass to verify gradients reach both the backbone and the domain
   / FiLM heads.

Usage
-----
    python experiments/smoke_domain_adapt.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.models.domain_adaptation_wrapper import DomainAdaptationWrapper


def make_inputs(batch_size: int = 2, clip_len: int = 5, n_views: int = 4, n_joints: int = 17):
    """Return random (x, K, R, t) and source/target domain labels."""
    x = torch.rand(batch_size, clip_len, n_views, n_joints, 3)
    x[..., 2] = torch.rand_like(x[..., 2])  # confidence in [0, 1]

    K = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_views, 1, 1)
    K[:, :, 0, 0] = 800.0
    K[:, :, 1, 1] = 800.0
    K[:, :, 0, 2] = 320.0
    K[:, :, 1, 2] = 240.0

    # Simple circular rig.
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_views, 1, 1)
    t = torch.zeros(batch_size, n_views, 3)
    for i in range(n_views):
        theta = 2.0 * 3.14159265 * i / n_views
        t[:, i, :] = torch.tensor([3.0 * torch.cos(torch.tensor(theta)),
                                   3.0 * torch.sin(torch.tensor(theta)),
                                   0.0])

    domain_labels = torch.randint(0, 2, (batch_size,))
    return x, K, R, t, domain_labels


def main():
    print("Instantiating DomainAdaptationWrapper on CPU...")
    model = DomainAdaptationWrapper(
        j=17,
        d=32,
        n_views=4,
        n_heads=2,
        n_joint_layers=1,
        n_st_layers=1,
        max_temporal_len=64,
        residual_hidden=32,
        use_domain_classifier=True,
        use_domain_film=True,
        grl_lambda=1.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    x, K, R, t, domain_labels = make_inputs(batch_size=2, clip_len=5, n_views=4, n_joints=17)

    # 1. Forward with domain labels and GRL logits.
    print("\n--- Mode 1: forward with domain labels ---")
    pred, weights, dlogits = model(
        x,
        K=K,
        R=R,
        t=t,
        domain_labels=domain_labels,
        return_domain_logits=True,
    )
    print(f"pred_3d shape: {pred.shape}")
    print(f"weights shape: {weights.shape}")
    print(f"domain_logits shape: {dlogits.shape}")
    assert pred.shape == (2, 5, 17, 3)
    assert weights.shape == (2, 5, 4, 17)
    assert dlogits.shape == (2 * 5, 2)

    # 2. Backward pass.
    print("\n--- Mode 2: backward pass ---")
    target = domain_labels.unsqueeze(1).expand(2, 5).reshape(-1)
    domain_loss = F.cross_entropy(dlogits, target)
    pose_loss = pred.mean()
    loss = pose_loss + 0.1 * domain_loss
    loss.backward()
    grads = [p.grad is not None for p in model.parameters()]
    print(f"Parameters with gradients: {sum(grads)}/{len(grads)}")
    assert any(grads), "No gradients computed"

    # 3. Forward without domain labels (unlabeled target inference).
    print("\n--- Mode 3: forward without domain labels ---")
    with torch.no_grad():
        pred2, weights2 = model(x, K=K, R=R, t=t)
    print(f"pred_3d shape: {pred2.shape}")
    print(f"weights shape: {weights2.shape}")
    assert pred2.shape == (2, 5, 17, 3)
    assert weights2.shape == (2, 5, 4, 17)

    # 4. Optional ablation: disable domain classifier / FiLM.
    print("\n--- Mode 4: wrapper without domain classifier / FiLM ---")
    model_ablate = DomainAdaptationWrapper(
        j=17,
        d=32,
        n_views=4,
        n_heads=2,
        n_joint_layers=1,
        n_st_layers=1,
        max_temporal_len=64,
        residual_hidden=32,
        use_domain_classifier=False,
        use_domain_film=False,
    )
    with torch.no_grad():
        pred3, weights3 = model_ablate(x, K=K, R=R, t=t)
    assert pred3.shape == (2, 5, 17, 3)
    assert weights3.shape == (2, 5, 4, 17)
    print(f"pred_3d shape: {pred3.shape}")
    print(f"weights shape: {weights3.shape}")

    print("\nCPU smoke test passed.")


if __name__ == "__main__":
    main()
