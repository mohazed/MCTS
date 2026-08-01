"""Recursive UCT with a transposition table -- the course's `UCT(board)`.

Transcription of the course snippet, with two purely notational changes:
  * the entry is a small object instead of a list (`t.n` for `t[0]`, etc.);
  * per-move arrays are indexed by column instead of by rank in `legalMoves()`,
    since in Connect 4 the move *is* the column.

Everything else is the course verbatim, in particular:
  * scores stay in [0, 1] from YELLOW's point of view and are flipped with
    `if board.turn == RED: Q = 1 - Q`;
  * an unvisited move gets an infinite UCB value, so every move is tried once;
  * a state absent from the table is added and evaluated with a single random
    playout ("expansion + simulation" in one step);
  * `BestMoveUCT` clears the table at every root move (`Table = {}`).
"""

from __future__ import annotations

import math
import time

import numpy as np

from game.connect4 import COLS, RED, Board
from search.base import empty_info
from search.tt import TranspositionTable

INF = 1e9
DEFAULT_C = 0.4  # course's exploration constant, calibrated for scores in [0,1]


def UCT(board: Board, table: TranspositionTable, c: float = DEFAULT_C) -> float:
    """One UCT simulation from `board` (mutated in place).  Returns the score
    of the playout, in [0, 1] from YELLOW's point of view."""
    if board.terminal():
        return board.score()
    t = table.look(board)
    if t is not None:
        best_value = -INF
        best = -1
        moves = board.legalMoves()
        log_n = math.log(t.n) if t.n > 0 else 0.0
        black = board.turn == RED
        nplayouts = t.nplayouts
        nwins = t.nwins
        for m in moves:
            ni = nplayouts[m]
            if ni == 0:
                val = INF
            else:
                Q = nwins[m] / ni
                if black:
                    Q = 1.0 - Q
                val = Q + c * math.sqrt(log_n / ni)
            if val > best_value:
                best_value = val
                best = m
        board.play(best)
        res = UCT(board, table, c)
        t.n += 1
        nplayouts[best] += 1.0
        nwins[best] += res
        return res
    table.add(board)
    return board.playout()


def BestMoveUCT(
    board: Board, n: int, c: float = DEFAULT_C
) -> tuple[int, np.ndarray, TranspositionTable]:
    """`n` UCT simulations from `board`; returns the most visited move."""
    table = TranspositionTable()
    for _ in range(n):
        b = board.copy()
        UCT(b, table, c)
    t = table.look(board)
    moves = board.legalMoves()
    if t is None:  # n == 0
        return moves[0], np.zeros(COLS), table
    visits = t.nplayouts
    best = max(moves, key=lambda m: visits[m])
    return best, visits, table


class UCTPlayer:
    def __init__(
        self, playouts: int = 200, c: float = DEFAULT_C, name: str | None = None
    ) -> None:
        self.playouts = playouts
        self.c = c
        self.name = name or f"uct-{playouts}"

    def choose_move(self, board: Board) -> tuple[int, dict]:
        t0 = time.perf_counter()
        col, visits, table = BestMoveUCT(board, self.playouts, self.c)
        t = table.look(board)
        value = None
        if t is not None and t.nplayouts[col] > 0:
            q = t.nwins[col] / t.nplayouts[col]
            if board.turn == RED:
                q = 1.0 - q
            value = 2.0 * q - 1.0
        return col, empty_info(
            visits=visits.copy(),
            value=value,
            time_ms=(time.perf_counter() - t0) * 1e3,
            tt_hits=table.hits,
            tt_lookups=table.lookups,
        )
