"""Transposition table: a dict keyed on the Zobrist hash, course style.

The course writes `Table = {}` and stores one list per state:

    t[0] = total number of playouts through the state
    t[1] = playouts per move
    t[2] = wins per move
    t[3] = AMAF playouts per move code   (RAVE / GRAVE)
    t[4] = AMAF wins per move code

`Entry` below is exactly that layout with names instead of indices, plus the two
fields PUCT needs: the policy `prior` and the network `value` of the state
(which the course also caches in the entry).

The table is rebuilt at every root move, as in the course's `BestMoveUCT`.
Connect 4 is rich in transpositions -- any permutation of the same set of column
drops reaches the same position -- so the hit rate is worth reporting (E9).
"""

from __future__ import annotations

import numpy as np

from game.connect4 import AMAF_CODES, COLS, Board

NMOVES = COLS  # a move is a column, so move index == column index


class Entry:
    """Statistics of one state.  Arrays are indexed by column, not by rank in
    `legalMoves()`; for Connect 4 the column *is* the move."""

    __slots__ = ("n", "nplayouts", "nwins", "namaf", "wamaf", "prior", "value")

    def __init__(self, amaf: bool = False) -> None:
        self.n: float = 0.0
        self.nplayouts = np.zeros(NMOVES, dtype=np.float64)
        self.nwins = np.zeros(NMOVES, dtype=np.float64)
        self.namaf = np.zeros(AMAF_CODES, dtype=np.float64) if amaf else None
        self.wamaf = np.zeros(AMAF_CODES, dtype=np.float64) if amaf else None
        self.prior: np.ndarray | None = None
        self.value: float | None = None

    def __repr__(self) -> str:
        return f"<Entry n={self.n:.0f} visits={self.nplayouts.astype(int).tolist()}>"


class TranspositionTable:
    """`dict[int, Entry]` plus hit statistics."""

    __slots__ = ("table", "lookups", "hits", "amaf")

    def __init__(self, amaf: bool = False) -> None:
        self.table: dict[int, Entry] = {}
        self.lookups = 0
        self.hits = 0
        self.amaf = amaf

    def look(self, board: Board) -> Entry | None:
        self.lookups += 1
        e = self.table.get(board.h)
        if e is not None:
            self.hits += 1
        return e

    def add(self, board: Board) -> Entry:
        e = Entry(amaf=self.amaf)
        self.table[board.h] = e
        return e

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return (
            f"<TT {len(self.table)} states, "
            f"{self.hits}/{self.lookups} hits ({self.hit_rate:.1%})>"
        )
