"""Connect 4 (Puissance 4) engine, 6 rows x 7 columns.

Written in the style of the course's Breakthrough `Board` class: `legalMoves`,
`play`, `terminal`, `score`, `playout`.

Conventions (frozen interface, see README.md):
  - row 0 is the BOTTOM row, row 5 the top one.
  - cell index = row * COLS + col, in [0, 42).
  - YELLOW plays first.
  - `score()` returns a value in [0, 1] from YELLOW's point of view:
    1.0 if YELLOW wins, 0.0 if RED wins, 0.5 for a draw.  This is exactly the
    course's convention (`1.0 if White wins, 0.0 else, 0.5 draw`), so the
    course's UCT / RAVE / GRAVE / PUCT snippets can be transcribed unchanged,
    including the `if board.turn == Black: Q = 1 - Q` idiom.

Storage note: the canonical storage is a flat Python list of 42 ints
(`cells`) plus a list of 7 column heights.  `board` is a read-only derived
property returning the (6, 7) int8 numpy array of the specification.  Flat
lists are used because Flat MC / UCT / GRAVE run millions of random playouts
and pure-Python list indexing is several times faster than numpy scalar
indexing here.  This is *not* a bitboard: the representation is still one
integer per cell, exactly as in the course's snippets.
"""

from __future__ import annotations

import random

import numpy as np

ROWS, COLS = 6, 7
NCELLS = ROWS * COLS  # 42
EMPTY, YELLOW, RED = 0, 1, 2  # YELLOW plays first

# AMAF / RAVE move codes: the *filled cell* plus the *player*, 2 * 42 = 84 codes.
# Coding by column would give only 7 codes, which is statistically useless: a
# single playout plays almost every column, so every code would be credited in
# almost every playout.  This mirrors the course's `Move.code` for Breakthrough
# (`5 * (Dy * x1 + y1) + direction`, duplicated per colour).
AMAF_CODES = 2 * NCELLS  # 84

# Center-first move ordering, as recommended by the course for alpha-beta and
# reused everywhere so that every agent breaks ties the same way.
MOVE_ORDER = (3, 2, 4, 1, 5, 0, 6)

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _build_lines() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """For each cell, the list of 4-in-a-row lines going through it."""
    lines: list[tuple[int, ...]] = []
    for row in range(ROWS):
        for col in range(COLS):
            for dr, dc in DIRECTIONS:
                end_r, end_c = row + 3 * dr, col + 3 * dc
                if 0 <= end_r < ROWS and 0 <= end_c < COLS:
                    lines.append(
                        tuple((row + k * dr) * COLS + (col + k * dc) for k in range(4))
                    )
    through: list[list[tuple[int, ...]]] = [[] for _ in range(NCELLS)]
    for line in lines:
        for idx in line:
            through[idx].append(line)
    return tuple(tuple(x) for x in through)


ALL_LINES = tuple(
    line
    for row in range(ROWS)
    for col in range(COLS)
    for dr, dc in DIRECTIONS
    if 0 <= row + 3 * dr < ROWS and 0 <= col + 3 * dc < COLS
    for line in (tuple((row + k * dr) * COLS + (col + k * dc) for k in range(4)),)
)
LINES_THROUGH = _build_lines()

# --- Zobrist hashing -------------------------------------------------------
# The course asks "how many random numbers do we need?".  Here: one 64 bit
# number per (colour, cell) pair plus one for the side to move, i.e.
# 2 * 42 + 1 = 85.  (For chess the course counts 789.)
# The numbers are drawn from a *fixed* seed so that hashes are identical across
# processes -- self-play uses multiprocessing and workers must agree.
_ZOBRIST_SEED = 20260731


def _build_zobrist() -> tuple[list[list[int]], int]:
    rng = random.Random(_ZOBRIST_SEED)
    table = [[rng.getrandbits(64) for _ in range(NCELLS)] for _ in range(2)]
    turn_key = rng.getrandbits(64)
    return table, turn_key


ZOBRIST, ZOBRIST_TURN = _build_zobrist()


def move_code(row: int, col: int, player: int) -> int:
    """AMAF code of "player drops a stone in cell (row, col)"; in [0, 84)."""
    return (player - 1) * NCELLS + row * COLS + col


class Board:
    """A Connect 4 position.

    Attributes:
        cells:   list[int] of length 42, canonical storage (index row*7+col).
        heights: list[int] of length 7, number of stones already in each column.
        turn:    YELLOW or RED, the player to move.
        moves:   number of stones played so far.
        h:       Zobrist hash of the position (side to move included).
        last:    (row, col) of the last stone played, or None.
        winner:  YELLOW, RED or None; maintained incrementally by `play`.
    """

    __slots__ = ("cells", "heights", "turn", "moves", "h", "last", "winner", "_undo")

    def __init__(self) -> None:
        self.cells = [EMPTY] * NCELLS
        self.heights = [0] * COLS
        self.turn = YELLOW
        self.moves = 0
        self.h = 0
        self.last: tuple[int, int] | None = None
        self.winner: int | None = None
        self._undo: list[tuple[int, int, int | None, tuple[int, int] | None]] = []

    # --- derived views -----------------------------------------------------

    @property
    def board(self) -> np.ndarray:
        """(6, 7) int8 view of the position, row 0 = bottom.  Read-only."""
        return np.asarray(self.cells, dtype=np.int8).reshape(ROWS, COLS)

    def key(self) -> tuple:
        """Exact, collision-free identity of the position (for tests)."""
        return (tuple(self.cells), self.turn)

    # --- core interface ----------------------------------------------------

    def legalMoves(self) -> list[int]:
        """Playable columns, center-first.  Never returns a full column."""
        heights = self.heights
        return [c for c in MOVE_ORDER if heights[c] < ROWS]

    def move_code(self, col: int) -> int:
        """AMAF code of playing `col` in the current position."""
        return move_code(self.heights[col], col, self.turn)

    def play(self, col: int) -> None:
        """Drop a stone of the player to move in column `col`."""
        row = self.heights[col]
        if row >= ROWS:
            raise ValueError(f"column {col} is full")
        player = self.turn
        idx = row * COLS + col
        self.cells[idx] = player
        self.heights[col] = row + 1
        self._undo.append((col, row, self.winner, self.last))
        self.last = (row, col)
        self.moves += 1
        self.h ^= ZOBRIST[player - 1][idx] ^ ZOBRIST_TURN
        self.turn = RED if player == YELLOW else YELLOW
        if self._wins_at(idx, player):
            self.winner = player

    def unplay(self) -> None:
        """Undo the last `play` (make / unmake, used by alpha-beta).

        Only moves made by `play` can be undone: `playout` clears the stack, so
        calling `unplay` after a playout raises IndexError rather than quietly
        corrupting the position.
        """
        col, row, winner, last = self._undo.pop()
        idx = row * COLS + col
        player = self.cells[idx]
        self.cells[idx] = EMPTY
        self.heights[col] = row
        self.moves -= 1
        self.h ^= ZOBRIST[player - 1][idx] ^ ZOBRIST_TURN
        self.turn = player
        self.winner = winner
        self.last = last

    def terminal(self) -> bool:
        return self.winner is not None or self.moves == NCELLS

    def score(self) -> float:
        """Result in [0, 1] from YELLOW's point of view (course convention)."""
        if self.winner == YELLOW:
            return 1.0
        if self.winner == RED:
            return 0.0
        return 0.5

    def playout(self) -> float:
        """Uniform random playout *in place*; returns `score()`.

        Course style: the caller is responsible for working on a copy.
        """
        cells = self.cells
        heights = self.heights
        randrange = random.randrange
        # a playout does not record undo information: drop the stack so that a
        # later unplay() fails loudly instead of silently corrupting the board
        self._undo.clear()
        while self.winner is None and self.moves < NCELLS:
            legal = [c for c in MOVE_ORDER if heights[c] < ROWS]
            col = legal[randrange(len(legal))]
            row = heights[col]
            player = self.turn
            idx = row * COLS + col
            cells[idx] = player
            heights[col] = row + 1
            self.moves += 1
            self.h ^= ZOBRIST[player - 1][idx] ^ ZOBRIST_TURN
            self.turn = RED if player == YELLOW else YELLOW
            self.last = (row, col)
            if self._wins_at(idx, player):
                self.winner = player
        return self.score()

    def playoutAMAF(self, played: list[int]) -> float:
        """Random playout appending the AMAF code of every move to `played`.

        Transcription of the course's `playoutAMAF(board, played)`.
        """
        cells = self.cells
        heights = self.heights
        randrange = random.randrange
        self._undo.clear()  # see playout(): no undo information is recorded
        while self.winner is None and self.moves < NCELLS:
            legal = [c for c in MOVE_ORDER if heights[c] < ROWS]
            col = legal[randrange(len(legal))]
            row = heights[col]
            player = self.turn
            played.append((player - 1) * NCELLS + row * COLS + col)
            idx = row * COLS + col
            cells[idx] = player
            heights[col] = row + 1
            self.moves += 1
            self.h ^= ZOBRIST[player - 1][idx] ^ ZOBRIST_TURN
            self.turn = RED if player == YELLOW else YELLOW
            self.last = (row, col)
            if self._wins_at(idx, player):
                self.winner = player
        return self.score()

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.cells = self.cells[:]
        b.heights = self.heights[:]
        b.turn = self.turn
        b.moves = self.moves
        b.h = self.h
        b.last = self.last
        b.winner = self.winner
        b._undo = []
        return b

    # --- symmetry ----------------------------------------------------------

    def mirror(self) -> "Board":
        """Left/right mirror of the position (the only symmetry of Connect 4)."""
        b = Board.__new__(Board)
        cells = self.cells
        b.cells = [
            cells[row * COLS + (COLS - 1 - col)]
            for row in range(ROWS)
            for col in range(COLS)
        ]
        b.heights = self.heights[::-1]
        b.turn = self.turn
        b.moves = self.moves
        b.last = None if self.last is None else (self.last[0], COLS - 1 - self.last[1])
        b.winner = self.winner
        b._undo = []
        b.h = b.compute_hash()
        return b

    # --- hashing -----------------------------------------------------------

    def compute_hash(self) -> int:
        """Recompute the Zobrist hash from scratch (used to test incrementality)."""
        h = 0
        for idx, player in enumerate(self.cells):
            if player != EMPTY:
                h ^= ZOBRIST[player - 1][idx]
        if self.turn == RED:
            h ^= ZOBRIST_TURN
        return h

    # --- helpers -----------------------------------------------------------

    def _wins_at(self, idx: int, player: int) -> bool:
        """Incremental win check: only the lines through the last stone."""
        cells = self.cells
        for a, b, c, d in LINES_THROUGH[idx]:
            if (
                cells[a] == player
                and cells[b] == player
                and cells[c] == player
                and cells[d] == player
            ):
                return True
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return (
            self.cells == other.cells
            and self.heights == other.heights
            and self.turn == other.turn
            and self.moves == other.moves
            and self.h == other.h
            and self.last == other.last
            and self.winner == other.winner
        )

    def __hash__(self) -> int:
        return self.h

    def __str__(self) -> str:
        symbols = {EMPTY: ".", YELLOW: "Y", RED: "R"}
        rows = [
            " ".join(symbols[self.cells[r * COLS + c]] for c in range(COLS))
            for r in range(ROWS - 1, -1, -1)
        ]
        rows.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(rows)

    def __repr__(self) -> str:
        return f"<Board moves={self.moves} turn={self.turn} h={self.h:#x}>"


def from_moves(cols: list[int] | str) -> Board:
    """Build a position by replaying a sequence of columns (helper for tests)."""
    if isinstance(cols, str):
        cols = [int(ch) for ch in cols if ch.isdigit()]
    b = Board()
    for c in cols:
        b.play(c)
    return b
