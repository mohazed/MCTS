"""Match runner with balanced openings, Wilson intervals and Elo.

Why balanced openings: Connect 4 is a solved game and the FIRST PLAYER WINS with
perfect play.  Letting one agent always move first would bias every number in
the report.  The protocol is therefore: the 7 possible first
moves, each played twice with the colours swapped, repeated `k` times -> 14k
games.  The forced first move is played for the side to move, whichever agent
that is.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from game.connect4 import COLS, RED, YELLOW, Board
from search.base import build_agent

Z95 = 1.959963984540054


def wilson(successes: float, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    `successes` may be fractional: draws count as half a win, which is the usual
    convention for game results.  The Wilson interval is preferred to the normal
    approximation because it stays inside [0, 1] and behaves at p = 0 or 1.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def elo_diff(p: float) -> float:
    """Elo difference implied by a score rate p:  d = -400 log10(1/p - 1)."""
    eps = 1e-9
    p = min(1.0 - eps, max(eps, p))
    return -400.0 * math.log10(1.0 / p - 1.0)


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    ci_low: float
    ci_high: float
    elo: float
    seconds: float
    ms_per_move_a: float = 0.0
    ms_per_move_b: float = 0.0
    openings_depth: int = 1
    tt_hit_rate_a: float = 0.0
    tt_hit_rate_b: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.name_a} vs {self.name_b}: {self.score:.1%} "
            f"[{self.ci_low:.1%}, {self.ci_high:.1%}]  "
            f"({self.wins}W/{self.draws}D/{self.losses}L on {self.games} games, "
            f"Elo {self.elo:+.0f})"
        )


def play_game(
    yellow, red, opening, seed: int, collect: dict | None = None
) -> float:
    """Play one game after a forced opening; returns score in [0,1] POV YELLOW."""
    random.seed(seed)
    b = Board()
    for col in (opening if isinstance(opening, (list, tuple)) else [opening]):
        b.play(col)
    while not b.terminal():
        agent = yellow if b.turn == YELLOW else red
        col, info = agent.choose_move(b)
        if col not in b.legalMoves():
            raise RuntimeError(f"{agent.name} returned illegal move {col}")
        if collect is not None:
            side = "a" if agent is collect["agent_a"] else "b"
            collect[f"ms_{side}"] += info["time_ms"]
            collect[f"n_{side}"] += 1
            collect[f"hits_{side}"] += info.get("tt_hits", 0)
            collect[f"look_{side}"] += info.get("tt_lookups", 0)
        b.play(col)
    return b.score()


def balanced_schedule(k: int = 2, depth: int = 1) -> list[tuple[tuple, bool]]:
    """(opening moves, a_is_yellow) pairs: 7^depth openings x 2 colours x k.

    `depth = 1` is the standard protocol: the 7 possible first moves, each
    played twice with the colours swapped, repeated k times -> 14k games.

    `depth = 2` uses the 49 two-move openings -> 98k games.  It exists because of
    a real problem with depth 1: when BOTH agents are deterministic (PUCT with
    training=False against alpha-beta, say), the k repeats of an opening replay
    the SAME game, so 14k games contain only 14 distinct ones and the Wilson
    interval computed on 14k is far too narrow.  Deterministic pairings must
    therefore widen the opening book rather than repeat it.
    """
    openings = [(c,) for c in range(COLS)]
    for _ in range(depth - 1):
        openings = [o + (c,) for o in openings for c in range(COLS)]
    sched = []
    for _ in range(k):
        for o in openings:
            sched.append((o, True))
            sched.append((o, False))
    return sched


def _worker(args):
    spec_a, spec_b, opening, a_yellow, seed = args
    agent_a = build_agent(spec_a)
    agent_b = build_agent(spec_b)
    collect = {
        "agent_a": agent_a,
        "ms_a": 0.0, "ms_b": 0.0, "n_a": 0, "n_b": 0,
        "hits_a": 0, "hits_b": 0, "look_a": 0, "look_b": 0,
    }
    y, r = (agent_a, agent_b) if a_yellow else (agent_b, agent_a)
    s = play_game(y, r, opening, seed, collect)
    sa = s if a_yellow else 1.0 - s
    collect.pop("agent_a")
    return sa, collect


def play_match(
    agent_a=None,
    agent_b=None,
    k: int = 2,
    seed: int = 20260731,
    spec_a: dict | None = None,
    spec_b: dict | None = None,
    workers: int = 1,
    name_a: str | None = None,
    name_b: str | None = None,
    pool=None,
    openings_depth: int = 1,
) -> MatchResult:
    """Balanced-opening match.  `score` counts a draw as half a win.

    Either pass instantiated agents (sequential) or picklable specs (which also
    enables `workers > 1`, each worker rebuilding its own agents).
    """
    sched = balanced_schedule(k, openings_depth)
    t0 = time.perf_counter()
    acc = {
        "ms_a": 0.0, "ms_b": 0.0, "n_a": 0, "n_b": 0,
        "hits_a": 0, "hits_b": 0, "look_a": 0, "look_b": 0,
    }
    scores: list[float] = []

    if workers > 1 or pool is not None:
        if spec_a is None or spec_b is None:
            raise ValueError("parallel matches require spec_a and spec_b")
        jobs = [
            (spec_a, spec_b, opening, a_yellow, seed + 1000 * i)
            for i, (opening, a_yellow) in enumerate(sched)
        ]
        if pool is not None:
            results = pool.map(_worker, jobs)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_init_worker) as p:
                results = p.map(_worker, jobs)
        for sa, coll in results:
            scores.append(sa)
            for key in acc:
                acc[key] += coll[key]
        name_a = name_a or spec_a.get("name") or spec_a["kind"]
        name_b = name_b or spec_b.get("name") or spec_b["kind"]
    else:
        if agent_a is None:
            agent_a = build_agent(spec_a)
        if agent_b is None:
            agent_b = build_agent(spec_b)
        collect = {"agent_a": agent_a, **acc}
        for i, (opening, a_yellow) in enumerate(sched):
            y, r = (agent_a, agent_b) if a_yellow else (agent_b, agent_a)
            s = play_game(y, r, opening, seed + 1000 * i, collect)
            scores.append(s if a_yellow else 1.0 - s)
        collect.pop("agent_a")
        acc = collect
        name_a = name_a or agent_a.name
        name_b = name_b or agent_b.name

    n = len(scores)
    wins = sum(1 for s in scores if s == 1.0)
    draws = sum(1 for s in scores if s == 0.5)
    losses = n - wins - draws
    total = wins + 0.5 * draws
    p = total / n
    lo, hi = wilson(total, n)
    return MatchResult(
        name_a=name_a,
        name_b=name_b,
        games=n,
        wins=wins,
        draws=draws,
        losses=losses,
        score=p,
        ci_low=lo,
        ci_high=hi,
        elo=elo_diff(p),
        seconds=time.perf_counter() - t0,
        openings_depth=openings_depth,
        ms_per_move_a=acc["ms_a"] / max(1, acc["n_a"]),
        ms_per_move_b=acc["ms_b"] / max(1, acc["n_b"]),
        tt_hit_rate_a=acc["hits_a"] / max(1, acc["look_a"]),
        tt_hit_rate_b=acc["hits_b"] / max(1, acc["look_b"]),
    )


def _init_worker() -> None:
    """Workers must not each grab every core for torch."""
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
