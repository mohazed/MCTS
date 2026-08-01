"""Self-play generation of (planes, pi, z) samples, with root parallelisation.

One game:
  1. PUCT with `sims` simulations and Dirichlet noise at the root
  2. pi = visits / visits.sum(), computed AFTER the noise: the target is the
     visit distribution actually produced
  3. store (planes, pi, turn)
  4. play a move sampled from pi while moves < temperature_moves, then argmax
     of the visits (temperature 1 then 0)
  5. at the end of the game, z = to_pov(score, turn) for every stored state
  6. add the mirrored sample (mirror(planes), mirror(pi), z)

Parallelisation is at the *root*, the simplest scheme of the course's
"Parallelization of MCTS" slide: each process plays whole games on its own and
returns its samples.  `torch.set_num_threads(1)` in every worker, otherwise the
processes fight over the cores and it is slower than sequential.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from game.connect4 import COLS, Board
from model.encode import encode, mirror_pi, mirror_planes, to_pov
from model.net import load_net
from search.puct import BestMovePUCT

Sample = tuple[np.ndarray, np.ndarray, float]  # (3,6,7) f32, (7,) f32, f32


@dataclass
class SelfPlayConfig:
    sims: int = 100
    c_puct: float = 1.0
    temperature_moves: int = 8
    dirichlet_alpha: float = 1.0
    dirichlet_eps: float = 0.25
    augment_symmetry: bool = True

    @classmethod
    def from_cfg(cls, cfg: dict) -> "SelfPlayConfig":
        sp = cfg.get("selfplay", {})
        return cls(
            sims=sp.get("sims", 100),
            c_puct=cfg.get("search", {}).get("c_puct", 1.0),
            temperature_moves=sp.get("temperature_moves", 8),
            dirichlet_alpha=sp.get("dirichlet_alpha", 1.0),
            dirichlet_eps=sp.get("dirichlet_eps", 0.25),
            augment_symmetry=sp.get("augment_symmetry", True),
        )


def play_game(net, cfg: SelfPlayConfig, rng: np.random.Generator) -> list[Sample]:
    """One self-play game; returns its training samples."""
    b = Board()
    states: list[tuple[np.ndarray, np.ndarray, int]] = []
    while not b.terminal():
        _, visits, _, _, _ = BestMovePUCT(
            b,
            net,
            cfg.sims,
            cfg.c_puct,
            training=True,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_eps=cfg.dirichlet_eps,
            rng=rng,
        )
        total = visits.sum()
        if total == 0:  # sims == 0, degenerate config
            legal = b.legalMoves()
            visits = np.zeros(COLS)
            visits[legal] = 1.0
            total = visits.sum()
        pi = (visits / total).astype(np.float32)
        states.append((encode(b), pi, b.turn))
        if b.moves < cfg.temperature_moves:  # temperature 1
            # renormalise in float64: np.random.choice rejects a float32 sum
            p64 = pi.astype(np.float64)
            col = int(rng.choice(COLS, p=p64 / p64.sum()))
        else:  # temperature 0
            col = int(np.argmax(visits))
        b.play(col)

    score = b.score()
    samples: list[Sample] = []
    for planes, pi, turn in states:
        z = np.float32(to_pov(score, turn))
        samples.append((planes, pi, z))
        if cfg.augment_symmetry:
            samples.append((mirror_planes(planes), mirror_pi(pi), z))
    return samples


def _init_worker() -> None:
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _worker(args):
    ckpt_path, cfg, seed, n_games = args
    torch.set_num_threads(1)
    net = load_net(ckpt_path)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    out: list[Sample] = []
    for _ in range(n_games):
        out.extend(play_game(net, cfg, rng))
    return out


def generate(
    ckpt_path: str,
    cfg: SelfPlayConfig,
    n_games: int,
    workers: int = 1,
    seed: int = 0,
    pool: "mp.pool.Pool | None" = None,
) -> list[Sample]:
    """Play `n_games` self-play games, spread over `workers` processes."""
    if workers <= 1 and pool is None:
        return _worker((ckpt_path, cfg, seed, n_games))
    per = [n_games // workers] * workers
    for i in range(n_games % workers):
        per[i] += 1
    jobs = [
        (ckpt_path, cfg, seed + 7919 * i, k) for i, k in enumerate(per) if k > 0
    ]
    if pool is not None:
        chunks = pool.map(_worker, jobs)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker) as p:
            chunks = p.map(_worker, jobs)
    return [s for chunk in chunks for s in chunk]


def make_pool(workers: int):
    """A reusable spawn pool (workers are expensive to start on macOS)."""
    ctx = mp.get_context("spawn")
    return ctx.Pool(workers, initializer=_init_worker)
