"""Gradient steps on the AlphaGo Zero loss."""

from __future__ import annotations

import numpy as np
import torch

from model.net import Connect4Net, alphazero_loss


def train_steps(
    net: Connect4Net,
    buffer,
    steps: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    generator: torch.Generator | None = None,
) -> dict:
    """`steps` Adam steps on uniform batches drawn from the buffer.

    Returns the mean policy / value / total loss over the steps.  L2
    regularisation is Adam's `weight_decay`, i.e. the `1e-4 * ||theta||^2` term
    of the AlphaGo Zero loss.
    """
    X, P, Z = buffer.tensors()
    n = len(X)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    g = generator or torch.Generator().manual_seed(0)
    lp_sum = lv_sum = lt_sum = 0.0
    bs = min(batch_size, n)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        opt.zero_grad()
        logits, value = net(X[idx])
        loss, lp, lv = alphazero_loss(logits, value, P[idx], Z[idx])
        loss.backward()
        opt.step()
        lt_sum += float(loss.detach())
        lp_sum += float(lp.detach())
        lv_sum += float(lv.detach())
    net.eval()
    return {
        "loss_total": lt_sum / steps,
        "loss_policy": lp_sum / steps,
        "loss_value": lv_sum / steps,
        "n_train_samples": n,
    }
