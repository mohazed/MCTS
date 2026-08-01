"""AMAF, RAVE and GRAVE -- transcriptions of the course's snippets.

AMAF ("all moves as first") credits every move played anywhere in a playout, as
if it had been played first.  RAVE blends the AMAF mean into the UCT score with
a weight beta that decays as the real statistics accumulate.  GRAVE (Cazenave,
IJCAI 2015) fixes RAVE's weakness at low counts: the AMAF statistics used at a
node are those of the closest ANCESTOR with at least `ref` playouts, so a node
visited twice already benefits from thousands of AMAF samples.

`ref = 0` makes the node itself always the reference, i.e. plain RAVE -- which
is the non-regression test the course suggests.

A note on the move codes.  We deliberately do NOT code a move by
its column: there are only 7 columns and a single playout plays nearly all of
them, so every code would be credited in nearly every playout and the AMAF
statistics would carry no signal.  We code the FILLED CELL plus the PLAYER,
`code = player_index * 42 + row * 7 + col`, i.e. 84 codes -- the analogue of the
course's `Move.code` for Breakthrough.
"""

from __future__ import annotations

import math
import time

import numpy as np

from game.connect4 import AMAF_CODES, COLS, RED, Board
from search.base import empty_info
from search.tt import Entry, TranspositionTable

INF = 1e9
DEFAULT_REF = 50
DEFAULT_BIAS = 1e-5


def updateAMAF(t: Entry, played: list[int], res: float) -> None:
    """Course's `updateAMAF`: credit the first occurrence of each code.

    Only the first occurrence counts, otherwise a cell played late would be
    credited with the same weight as the move actually chosen at this node.
    """
    namaf = t.namaf
    wamaf = t.wamaf
    for i in range(len(played)):
        code = played[i]
        seen = False
        for j in range(i):
            if played[j] == code:
                seen = True
                break
        if not seen:
            namaf[code] += 1.0
            wamaf[code] += res


def RAVE(
    board: Board,
    table: TranspositionTable,
    played: list[int],
    bias: float = DEFAULT_BIAS,
) -> float:
    """Course's `RAVE(board, played)`; scores in [0,1] POV YELLOW."""
    if board.terminal():
        return board.score()
    t = table.look(board)
    if t is None:
        table.add(board)
        return board.playoutAMAF(played)

    best_value = -INF
    best = -1
    moves = board.legalMoves()
    black = board.turn == RED
    for m in moves:
        code = board.move_code(m)
        val = INF
        ni = t.nplayouts[m]
        if ni > 0:
            Q = t.nwins[m] / ni
            if black:
                Q = 1.0 - Q
            nam = t.namaf[code]
            if nam > 0:
                AMAF = t.wamaf[code] / nam
                if black:
                    AMAF = 1.0 - AMAF
                beta = nam / (ni + nam + bias * ni * nam)
                val = (1.0 - beta) * Q + beta * AMAF
            else:
                val = Q
        if val > best_value:
            best_value = val
            best = m
    played.append(board.move_code(best))
    board.play(best)
    res = RAVE(board, table, played, bias)
    t.n += 1
    t.nplayouts[best] += 1.0
    t.nwins[best] += res
    updateAMAF(t, played, res)
    return res


def GRAVE(
    board: Board,
    table: TranspositionTable,
    played: list[int],
    tref: Entry,
    ref: int = DEFAULT_REF,
    bias: float = DEFAULT_BIAS,
) -> float:
    """Course's `GRAVE(board, played, tref)`.

    `tref` is the entry of the closest ancestor with at least `ref` playouts;
    its AMAF statistics are the ones used here.  With `ref = 0` every node is
    its own reference and GRAVE degenerates into RAVE.
    """
    if board.terminal():
        return board.score()
    t = table.look(board)
    if t is None:
        table.add(board)
        return board.playoutAMAF(played)

    if t.n > ref:
        tref = t
    best_value = -INF
    best = -1
    moves = board.legalMoves()
    black = board.turn == RED
    namaf_ref = tref.namaf
    wamaf_ref = tref.wamaf
    for m in moves:
        code = board.move_code(m)
        val = INF
        ni = t.nplayouts[m]
        if ni > 0:
            Q = t.nwins[m] / ni
            if black:
                Q = 1.0 - Q
            nam = namaf_ref[code]
            if nam > 0:
                AMAF = wamaf_ref[code] / nam
                if black:
                    AMAF = 1.0 - AMAF
                beta = nam / (ni + nam + bias * ni * nam)
                val = (1.0 - beta) * Q + beta * AMAF
            else:
                val = Q
        if val > best_value:
            best_value = val
            best = m
    played.append(board.move_code(best))
    board.play(best)
    res = GRAVE(board, table, played, tref, ref, bias)
    t.n += 1
    t.nplayouts[best] += 1.0
    t.nwins[best] += res
    updateAMAF(t, played, res)
    return res


def BestMoveGRAVE(
    board: Board,
    n: int,
    ref: int = DEFAULT_REF,
    bias: float = DEFAULT_BIAS,
) -> tuple[int, np.ndarray, TranspositionTable]:
    table = TranspositionTable(amaf=True)
    root = table.add(board)
    for _ in range(n):
        b = board.copy()
        played: list[int] = []
        GRAVE(b, table, played, root, ref, bias)
    moves = board.legalMoves()
    visits = root.nplayouts
    best = max(moves, key=lambda m: visits[m])
    return best, visits, table


def BestMoveRAVE(
    board: Board, n: int, bias: float = DEFAULT_BIAS
) -> tuple[int, np.ndarray, TranspositionTable]:
    table = TranspositionTable(amaf=True)
    table.add(board)
    for _ in range(n):
        b = board.copy()
        played: list[int] = []
        RAVE(b, table, played, bias)
    root = table.look(board)
    moves = board.legalMoves()
    visits = root.nplayouts
    best = max(moves, key=lambda m: visits[m])
    return best, visits, table


class GRAVEPlayer:
    """`ref = 0` gives RAVE, `ref = 50` the course's GRAVE."""

    def __init__(
        self,
        playouts: int = 200,
        ref: int = DEFAULT_REF,
        bias: float = DEFAULT_BIAS,
        name: str | None = None,
    ) -> None:
        self.playouts = playouts
        self.ref = ref
        self.bias = bias
        self.name = name or (
            f"{'rave' if ref == 0 else 'grave'}-{playouts}"
        )

    def choose_move(self, board: Board) -> tuple[int, dict]:
        t0 = time.perf_counter()
        col, visits, table = BestMoveGRAVE(board, self.playouts, self.ref, self.bias)
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
