"""Endgame test set: positions solved exactly by alpha-beta.

200 non-terminal positions with at least 20 stones, deduplicated
by `board.h`, solved without depth limit.  The metric is the AGREEMENT RATE with
perfect play -- the fraction of positions where the agent picks one of the
optimal moves.

It is far more readable than a win rate: it does not saturate, it is noise-free
(the ground truth is exact), and it moves visibly from one iteration to the next.
"""

from __future__ import annotations

import json
import os
import random
import time

import numpy as np

from eval.baselines import AlphaBetaPlayer
from game.connect4 import RED, YELLOW, Board, from_moves
from model.encode import encode

DEFAULT_PATH = "report/results/testset.json"


def _random_position(rng: random.Random, min_stones: int) -> list[int] | None:
    b = Board()
    moves = []
    while b.moves < min_stones:
        if b.terminal():
            return None
        m = rng.choice(b.legalMoves())
        moves.append(m)
        b.play(m)
    return None if b.terminal() else moves


def build_testset(
    n: int = 200,
    min_stones: int = 20,
    seed: int = 20260731,
    path: str | None = DEFAULT_PATH,
    verbose: bool = True,
) -> list[dict]:
    """Generate and solve the test set (slow: minutes).  Cached as JSON."""
    rng = random.Random(seed)
    solver = AlphaBetaPlayer(depth=None)
    out: list[dict] = []
    seen: set[int] = set()
    t0 = time.time()
    while len(out) < n:
        moves = _random_position(rng, min_stones)
        if moves is None:
            continue
        b = from_moves(moves)
        if b.h in seen:
            continue
        seen.add(b.h)
        value, best = solver.analyse(b, depth=None)
        out.append(
            {
                "moves": moves,
                "optimal": sorted(int(m) for m in best),
                "value_yellow": float(value),
                "turn": int(b.turn),
                "stones": int(b.moves),
            }
        )
        if verbose and len(out) % 25 == 0:
            print(f"    testset {len(out)}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"min_stones": min_stones, "seed": seed, "positions": out}, f)
    return out


def load_testset(path: str = DEFAULT_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)["positions"]


def agreement(agent, positions: list[dict]) -> float:
    """Fraction of positions where `agent` plays one of the optimal moves."""
    if not positions:
        return float("nan")
    ok = 0
    for p in positions:
        b = from_moves(p["moves"])
        col, _ = agent.choose_move(b)
        ok += int(col in p["optimal"])
    return ok / len(positions)


def network_agreement(net, positions: list[dict]) -> float:
    """Same, for the raw policy head (argmax, no search).  Batched."""
    if not positions:
        return float("nan")
    from model.encode import evaluate_batch

    boards = [from_moves(p["moves"]) for p in positions]
    priors, _ = evaluate_batch(net, boards)
    return float(
        np.mean([int(int(np.argmax(priors[i])) in p["optimal"])
                 for i, p in enumerate(positions)])
    )


# --------------------------------------------------------------------------
# fixed sanity positions, used to watch the value head during training
# --------------------------------------------------------------------------
def winning_positions(n: int = 50, seed: int = 1234) -> list[Board]:
    """Positions where the player to move has an immediate win (value = +1)."""
    rng = random.Random(seed)
    out: list[Board] = []
    seen: set[int] = set()
    guard = 0
    while len(out) < n and guard < 200_000:
        guard += 1
        b = Board()
        while not b.terminal():
            me = b.turn
            win = None
            for m in b.legalMoves():
                b.play(m)
                if b.winner == me:
                    win = m
                b.unplay()
                if win is not None:
                    break
            if win is not None:
                if b.h not in seen:
                    seen.add(b.h)
                    out.append(b.copy())
                break
            b.play(rng.choice(b.legalMoves()))
    return out


def mean_value(net, boards: list[Board]) -> float:
    from model.encode import evaluate_batch

    if not boards:
        return float("nan")
    _, values = evaluate_batch(net, boards)
    return float(values.mean())
