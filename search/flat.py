"""Flat Monte Carlo and UCB at the root -- direct transcriptions of the course.

These are the two first rungs of the ladder the course climbs:
random playouts -> equal budget per move (flat) -> bandit allocation at the root
(UCB) -> a tree of bandits (UCT).  They are the floor of the results table.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np

from game.connect4 import COLS, RED, Board
from search.base import empty_info


def flat(board: Board, n: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Flat Monte Carlo: split `n` playouts equally between the legal moves.

    Course's `flat(board, n)`.  Returns (best column, visits, mean scores),
    the scores being seen from the player to move.
    """
    moves = board.legalMoves()
    visits = np.zeros(COLS, dtype=np.float64)
    sums = np.zeros(COLS, dtype=np.float64)
    per_move = max(1, n // len(moves))
    black = board.turn == RED
    for m in moves:
        for _ in range(per_move):
            b = board.copy()
            b.play(m)
            r = b.playout()
            if black:
                r = 1.0 - r
            sums[m] += r
            visits[m] += 1.0
    means = np.full(COLS, -1.0)
    np.divide(sums, visits, out=means, where=visits > 0)
    best = max(moves, key=lambda m: means[m])
    return best, visits, means


def UCB(board: Board, n: int, c: float = 0.4) -> tuple[int, np.ndarray, np.ndarray]:
    """UCB at the root: allocate the `n` playouts with the UCB1 formula.

    Course's `UCB(board, n)`.  The move finally returned is the most played one
    (the course's `BestMove`), not the one with the best mean: that is the
    standard, more robust choice and the one UCT also uses.
    """
    moves = board.legalMoves()
    visits = np.zeros(COLS, dtype=np.float64)
    sums = np.zeros(COLS, dtype=np.float64)
    black = board.turn == RED
    for i in range(n):
        best_score = -1e99
        best_move = moves[0]
        for m in moves:
            if visits[m] == 0:
                score = 1e99
            else:
                score = sums[m] / visits[m] + c * math.sqrt(math.log(i + 1) / visits[m])
            if score > best_score:
                best_score = score
                best_move = m
        b = board.copy()
        b.play(best_move)
        r = b.playout()
        if black:
            r = 1.0 - r
        sums[best_move] += r
        visits[best_move] += 1.0
    means = np.full(COLS, -1.0)
    np.divide(sums, visits, out=means, where=visits > 0)
    best = max(moves, key=lambda m: (visits[m], means[m]))
    return best, visits, means


class FlatPlayer:
    def __init__(self, playouts: int = 200, name: str | None = None) -> None:
        self.playouts = playouts
        self.name = name or f"flat-{playouts}"

    def choose_move(self, board: Board) -> tuple[int, dict]:
        t0 = time.perf_counter()
        col, visits, means = flat(board, self.playouts)
        return col, empty_info(
            visits=visits,
            value=float(means[col]) * 2.0 - 1.0,
            time_ms=(time.perf_counter() - t0) * 1e3,
        )


class UCBPlayer:
    def __init__(
        self, playouts: int = 200, c: float = 0.4, name: str | None = None
    ) -> None:
        self.playouts = playouts
        self.c = c
        self.name = name or f"ucb-{playouts}"

    def choose_move(self, board: Board) -> tuple[int, dict]:
        t0 = time.perf_counter()
        col, visits, means = UCB(board, self.playouts, self.c)
        return col, empty_info(
            visits=visits,
            value=float(means[col]) * 2.0 - 1.0,
            time_ms=(time.perf_counter() - t0) * 1e3,
        )
