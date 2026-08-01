"""Two-headed residual network, AlphaGo Zero style (course, "Golois" section).

The course offers "completely connected as a starting point.  Option:
convolutional network, residual network"; we take the residual option:

    input (3, 6, 7)
      -> Conv3x3(C) + BN + ReLU
      -> B x ResidualBlock(C)
      -> policy head : Conv1x1(2) + BN + ReLU -> flatten -> Linear(7)   [logits]
      -> value  head : Conv1x1(1) + BN + ReLU -> flatten -> Linear(32)
                       -> ReLU -> Linear(1) -> tanh

With C=64, B=3 that is about 225 k parameters and a CPU forward pass well under
a millisecond, which is what makes batching unnecessary here: the priors and
value of a state are cached in its transposition table entry, so each distinct
position is evaluated once per search.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from game.connect4 import COLS, ROWS
from model.encode import PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class Connect4Net(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * ROWS * COLS, COLS)
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(ROWS * COLS, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy logits (B,7), value (B,1) in [-1,1] POV player to move)."""
        x = self.tower(self.stem(x))
        p = self.policy_fc(self.policy_conv(x).flatten(1))
        v = self.value_fc(self.value_conv(x).flatten(1))
        return p, v

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def alphazero_loss(
    logits: torch.Tensor,
    value: torch.Tensor,
    pi: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AlphaGo Zero loss:  L = CE(policy, pi) + MSE(value, z)  (+ L2 via Adam).

    Cross-entropy with a *soft* target: -sum_a pi(a) log p(a).  The logits are
    NOT masked during training -- the target pi is already 0 on illegal moves,
    so the loss pushes those logits down on its own.  Masking is applied at
    inference (`model.encode.masked_softmax`).
    """
    logp = F.log_softmax(logits, dim=1)
    loss_p = -(pi * logp).sum(dim=1).mean()
    loss_v = F.mse_loss(value.reshape(-1), z.reshape(-1))
    return loss_p + loss_v, loss_p, loss_v


def save_net(net: Connect4Net, path: str, **extra) -> None:
    torch.save(
        {
            "channels": net.channels,
            "blocks": net.blocks,
            "state_dict": net.state_dict(),
            **extra,
        },
        path,
    )


def load_net(path: str) -> Connect4Net:
    """Load a checkpoint in eval mode (BatchNorm must not use batch statistics)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = Connect4Net(channels=ckpt["channels"], blocks=ckpt["blocks"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def new_net(channels: int = 64, blocks: int = 3) -> Connect4Net:
    net = Connect4Net(channels=channels, blocks=blocks)
    net.eval()
    return net
