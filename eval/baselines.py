"""Non-Monte-Carlo opponents: random play, alpha-beta, and the network alone.

The alpha-beta player follows the course's "Solving Games" section: negamax
with alpha-beta cuts, a transposition table keyed on the Zobrist hash storing
(depth, value, bound flag, best move), and center-first move ordering.

With no depth limit it *solves* the position; that is how the endgame test set
of eval/testset.py gets its ground truth.
"""

from __future__ import annotations

import random
import time

import numpy as np

from game.connect4 import ALL_LINES, COLS, EMPTY, NCELLS, RED, YELLOW, Board
from search.base import empty_info

# --- heuristic evaluation --------------------------------------------------
LINE_IDX = np.array(ALL_LINES, dtype=np.int64)  # (69, 4)
CENTER_COL = COLS // 2
WIN = 10_000  # a win is worth more than any heuristic score
MAX_HEURISTIC = 2_000

EXACT, LOWER, UPPER = 0, 1, 2


def heuristic(board: Board) -> float:
    """Positional score in (-2000, 2000), from YELLOW's point of view.

    Sum over the 69 alignments of: an open three is worth 10, an open two is
    worth 1 ("open" = the alignment contains no enemy stone), plus a bonus for
    stones in the central column.
    """
    arr = np.asarray(board.cells, dtype=np.int8)[LINE_IDX]  # (69, 4)
    y = (arr == YELLOW).sum(axis=1)
    r = (arr == RED).sum(axis=1)
    open_y = r == 0
    open_r = y == 0
    score = (
        10 * int(np.count_nonzero(open_y & (y == 3)))
        + int(np.count_nonzero(open_y & (y == 2)))
        - 10 * int(np.count_nonzero(open_r & (r == 3)))
        - int(np.count_nonzero(open_r & (r == 2)))
    )
    center = board.cells[CENTER_COL :: COLS]
    score += 3 * (center.count(YELLOW) - center.count(RED))
    return float(score)


def terminal_value(board: Board) -> float:
    """Exact value of a finished game, POV YELLOW; quicker wins score higher."""
    if board.winner == YELLOW:
        return WIN - board.moves
    if board.winner == RED:
        return -WIN + board.moves
    return 0.0


class AlphaBetaPlayer:
    """Minimax + alpha-beta on the [-WIN, WIN] scale, POV YELLOW.

    `depth=None` searches to the end of the game (exact solver).
    """

    def __init__(self, depth: int | None = 4, name: str | None = None) -> None:
        self.depth = depth
        self.name = name or (
            f"alphabeta-{depth}" if depth is not None else "alphabeta-solver"
        )
        self.table: dict[int, tuple] = {}
        self.nodes = 0
        self.lookups = 0
        self.hits = 0

    # --- search --------------------------------------------------------
    def _search(self, board: Board, depth: int | None, alpha: float, beta: float) -> float:
        self.nodes += 1
        if board.terminal():
            return terminal_value(board)
        if depth is not None and depth == 0:
            return heuristic(board)

        alpha0, beta0 = alpha, beta
        self.lookups += 1
        hit = self.table.get(board.h)
        tt_move = -1
        if hit is not None:
            self.hits += 1
            h_depth, h_value, h_flag, h_move = hit
            # `None` depth (solved to the end) is deeper than any finite depth
            deep_enough = h_depth is None or (depth is not None and h_depth >= depth)
            if deep_enough:
                if h_flag == EXACT:
                    return h_value
                if h_flag == LOWER:
                    alpha = max(alpha, h_value)
                else:
                    beta = min(beta, h_value)
                if alpha >= beta:
                    return h_value
            tt_move = h_move

        moves = board.legalMoves()  # already center-first
        if tt_move >= 0 and tt_move in moves:
            moves = [tt_move] + [m for m in moves if m != tt_move]

        maximizing = board.turn == YELLOW
        child_depth = None if depth is None else depth - 1
        best_move = moves[0]
        if maximizing:
            value = -float("inf")
            for m in moves:
                board.play(m)
                v = self._search(board, child_depth, alpha, beta)
                board.unplay()
                if v > value:
                    value, best_move = v, m
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            for m in moves:
                board.play(m)
                v = self._search(board, child_depth, alpha, beta)
                board.unplay()
                if v < value:
                    value, best_move = v, m
                beta = min(beta, value)
                if alpha >= beta:
                    break

        flag = EXACT
        if value <= alpha0:
            flag = UPPER
        elif value >= beta0:
            flag = LOWER
        # keep whichever entry was searched deeper; depth None == solved exactly,
        # i.e. deeper than any finite depth, so it must never be overwritten
        prev = self.table.get(board.h)
        if prev is None or depth is None or (
            prev[0] is not None and depth >= prev[0]
        ):
            self.table[board.h] = (depth, value, flag, best_move)
        return value

    def analyse(self, board: Board, depth: int | None = ...) -> tuple[float, list[int]]:
        """Return (value POV YELLOW, list of every optimal move)."""
        if depth is ...:
            depth = self.depth
        self.table = {}
        self.nodes = 0
        self.lookups = 0
        self.hits = 0
        b = board.copy()
        maximizing = b.turn == YELLOW
        child_depth = None if depth is None else depth - 1
        values: dict[int, float] = {}
        alpha, beta = -float("inf"), float("inf")
        for m in b.legalMoves():
            b.play(m)
            v = self._search(b, child_depth, alpha, beta)
            b.unplay()
            values[m] = v
            # no cut at the root: we want the value of every move
        best = max(values.values()) if maximizing else min(values.values())
        return best, [m for m, v in values.items() if v == best]

    def choose_move(self, board: Board) -> tuple[int, dict]:
        t0 = time.perf_counter()
        value, best = self.analyse(board)
        col = best[0]  # legalMoves is center-first, so this prefers the centre
        visits = np.zeros(COLS, dtype=np.float64)
        visits[best] = 1.0
        pov = value if board.turn == YELLOW else -value
        return col, empty_info(
            visits=visits,
            value=float(np.clip(pov / MAX_HEURISTIC, -1.0, 1.0)),
            time_ms=(time.perf_counter() - t0) * 1e3,
            tt_hits=self.hits,
            tt_lookups=self.lookups,
        )


class RandomPlayer:
    """Uniform random legal move.  The floor of every comparison."""

    def __init__(self, seed: int | None = None, name: str | None = None) -> None:
        self.rng = random.Random(seed) if seed is not None else random
        self.name = name or "random"

    def choose_move(self, board: Board) -> tuple[int, dict]:
        t0 = time.perf_counter()
        moves = board.legalMoves()
        col = self.rng.choice(moves)
        visits = np.zeros(COLS, dtype=np.float64)
        visits[moves] = 1.0
        return col, empty_info(visits=visits, time_ms=(time.perf_counter() - t0) * 1e3)


class NetworkOnlyPlayer:
    """argmax of the policy head, ZERO simulation.  This is criterion C4:
    it is the only agent whose strength comes entirely from the network."""

    def __init__(self, net, name: str | None = None) -> None:
        self.net = net
        self.name = name or "network-only"

    def choose_move(self, board: Board) -> tuple[int, dict]:
        from model.encode import evaluate

        t0 = time.perf_counter()
        prior, value = evaluate(self.net, board)
        col = int(np.argmax(prior))
        return col, empty_info(
            visits=prior.copy(),
            priors=prior.copy(),
            value=float(value),
            time_ms=(time.perf_counter() - t0) * 1e3,
        )
