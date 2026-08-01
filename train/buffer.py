"""Replay buffer: the samples of the last `buffer_iters` iterations, capped."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Keeps one bucket of samples per iteration and samples uniformly.

    AlphaZero trains on a sliding window of recent self-play games; keeping the
    window per-iteration (rather than per-sample) makes "the last 8 iterations"
    exactly what it says.
    """

    def __init__(self, max_iters: int = 8, max_samples: int = 60_000) -> None:
        self.buckets: deque[list] = deque(maxlen=max_iters)
        self.max_samples = max_samples

    def add_iteration(self, samples: list) -> None:
        self.buckets.append(list(samples))
        while self.n_samples > self.max_samples and len(self.buckets) > 1:
            self.buckets.popleft()

    @property
    def n_samples(self) -> int:
        return sum(len(b) for b in self.buckets)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The whole buffer as three tensors (X, pi, z)."""
        flat = [s for b in self.buckets for s in b]
        if not flat:
            raise ValueError("empty replay buffer")
        if len(flat) > self.max_samples:
            flat = flat[-self.max_samples :]
        X = torch.from_numpy(np.stack([s[0] for s in flat]))
        P = torch.from_numpy(np.stack([s[1] for s in flat]))
        Z = torch.tensor([float(s[2]) for s in flat], dtype=torch.float32)
        return X, P, Z

    def __len__(self) -> int:
        return self.n_samples
