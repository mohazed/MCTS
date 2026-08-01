"""Step 1 validation: the Connect 4 engine.

The engine detects wins *incrementally*, around the last stone only.  Every test
here cross-checks that against an INDEPENDENT full-board scanner written from
scratch below, so a bug in the incremental check cannot hide behind itself.
"""

import random

import numpy as np
import pytest

from game.connect4 import (
    AMAF_CODES,
    COLS,
    EMPTY,
    NCELLS,
    RED,
    ROWS,
    YELLOW,
    Board,
    from_moves,
    move_code,
)


# --------------------------------------------------------------------------
# Independent reference implementation (full board scan, no incrementality)
# --------------------------------------------------------------------------
def scan_wins(cells) -> list[tuple[int, tuple[int, ...]]]:
    """Every 4-in-a-row on the board.  Deliberately naive and exhaustive."""
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if not (0 <= r + 3 * dr < ROWS and 0 <= c + 3 * dc < COLS):
                    continue
                line = tuple((r + k * dr) * COLS + (c + k * dc) for k in range(4))
                p = cells[line[0]]
                if p != EMPTY and all(cells[i] == p for i in line):
                    out.append((p, line))
    return out


# --------------------------------------------------------------------------
# 40 hand-checked winning alignments: 10 horizontal, 10 vertical,
# 10 up-right diagonal, 10 anti-diagonal.  Both players, both side edges,
# bottom row and top row are all covered.
# Each entry is (name, move sequence, winning cells, expected winner).
# --------------------------------------------------------------------------
WIN_CASES = [
    ("H row0 cols0-3 Y", [0, 6, 1, 4, 3, 4, 2], (0, 1, 2, 3), YELLOW),
    ("H row0 cols3-6 Y", [5, 1, 4, 2, 3, 1, 6], (3, 4, 5, 6), YELLOW),
    ("H row0 cols0-3 R", [4, 2, 5, 0, 6, 1, 6, 3], (0, 1, 2, 3), RED),
    ("H row0 cols3-6 R", [0, 5, 2, 6, 2, 4, 2, 3], (3, 4, 5, 6), RED),
    ("H row1 cols1-4 Y", [3, 2, 3, 4, 4, 0, 2, 1, 1], (8, 9, 10, 11), YELLOW),
    ("H row2 cols0-3 R", [0, 0, 2, 1, 1, 1, 3, 0, 3, 3, 2, 2], (14, 15, 16, 17), RED),
    ("H row3 cols3-6 Y",
     [5, 3, 5, 6, 6, 5, 6, 3, 5, 4, 3, 4, 3, 1, 6, 4, 4], (24, 25, 26, 27), YELLOW),
    ("H row4 cols2-5 R",
     [3, 5, 5, 4, 4, 4, 4, 5, 3, 4, 3, 3, 2, 2, 5, 5, 2, 3, 2, 2],
     (30, 31, 32, 33), RED),
    ("H row5 cols0-3 Y",
     [1, 5, 1, 3, 3, 1, 1, 2, 3, 3, 3, 0, 0, 0, 3, 2, 0, 0, 0, 1, 2, 2, 1, 2, 2],
     (35, 36, 37, 38), YELLOW),
    ("H row5 cols3-6 R",
     [4, 4, 4, 5, 5, 4, 3, 6, 4, 4, 6, 5, 5, 3, 6, 5, 3, 5, 6, 6, 3, 6, 3, 3],
     (38, 39, 40, 41), RED),
    ("V col0 rows0-3 Y", [0, 6, 0, 2, 0, 2, 0], (0, 7, 14, 21), YELLOW),
    ("V col6 rows0-3 R", [2, 6, 4, 6, 5, 6, 5, 6], (6, 13, 20, 27), RED),
    ("V col3 rows0-3 Y", [3, 2, 3, 5, 3, 4, 3], (3, 10, 17, 24), YELLOW),
    ("V col0 rows2-5 R", [0, 2, 0, 0, 4, 0, 1, 0, 4, 0], (14, 21, 28, 35), RED),
    ("V col6 rows2-5 Y", [6, 6, 6, 4, 6, 3, 6, 2, 6], (20, 27, 34, 41), YELLOW),
    ("V col1 rows1-4 Y", [5, 1, 1, 2, 1, 3, 1, 3, 1], (8, 15, 22, 29), YELLOW),
    ("V col2 rows1-4 R", [2, 2, 0, 2, 1, 2, 0, 2], (9, 16, 23, 30), RED),
    ("V col4 rows2-5 Y", [4, 4, 4, 1, 4, 3, 4, 5, 4], (18, 25, 32, 39), YELLOW),
    ("V col5 rows0-3 R", [0, 5, 3, 5, 1, 5, 1, 5], (5, 12, 19, 26), RED),
    ("V col3 rows2-5 R", [3, 1, 3, 3, 5, 3, 1, 3, 1, 3], (17, 24, 31, 38), RED),
    ("D/ (0,0)-(3,3) Y", [0, 3, 6, 2, 2, 1, 1, 3, 2, 3, 3], (0, 8, 16, 24), YELLOW),
    ("D/ (0,3)-(3,6) R", [4, 3, 6, 4, 6, 5, 6, 6, 5, 5], (3, 11, 19, 27), RED),
    ("D/ (0,1)-(3,4) Y", [1, 3, 3, 0, 3, 4, 2, 4, 2, 4, 4], (1, 9, 17, 25), YELLOW),
    ("D/ (0,2)-(3,5) R", [3, 2, 4, 3, 4, 5, 5, 4, 5, 5], (2, 10, 18, 26), RED),
    ("D/ (1,0)-(4,3) R",
     [2, 3, 2, 1, 3, 0, 1, 1, 2, 0, 3, 2, 3, 3], (7, 15, 23, 31), RED),
    ("D/ (2,0)-(5,3) Y",
     [1, 5, 3, 1, 2, 0, 3, 1, 0, 2, 0, 2, 1, 3, 3, 2, 2, 3, 3],
     (14, 22, 30, 38), YELLOW),
    ("D/ (2,3)-(5,6) R",
     [5, 4, 3, 3, 5, 3, 5, 6, 4, 5, 4, 4, 6, 5, 6, 6, 6, 6],
     (17, 25, 33, 41), RED),
    ("D/ (1,2)-(4,5) Y",
     [3, 1, 3, 5, 3, 4, 2, 5, 5, 5, 2, 4, 5, 4, 4], (9, 17, 25, 33), YELLOW),
    ("D/ (2,1)-(5,4) R",
     [1, 3, 2, 4, 1, 4, 2, 1, 4, 4, 3, 4, 2, 4, 3, 2, 3, 3],
     (15, 23, 31, 39), RED),
    ("D/ (1,3)-(4,6) Y",
     [5, 3, 6, 4, 4, 6, 4, 1, 5, 6, 3, 5, 5, 6, 6], (10, 18, 26, 34), YELLOW),
    ("D\\ (3,0)-(0,3) Y", [1, 0, 0, 0, 0, 5, 3, 1, 1, 2, 2], (3, 9, 15, 21), YELLOW),
    ("D\\ (3,3)-(0,6) R", [3, 4, 3, 6, 3, 3, 4, 4, 5, 5], (6, 12, 18, 24), RED),
    ("D\\ (3,1)-(0,4) R", [3, 1, 2, 4, 2, 3, 1, 2, 1, 1], (4, 10, 16, 22), RED),
    ("D\\ (3,2)-(0,5) Y", [2, 4, 5, 1, 3, 2, 4, 3, 3, 2, 2], (5, 11, 17, 23), YELLOW),
    ("D\\ (4,0)-(1,3) R",
     [3, 0, 1, 3, 2, 2, 1, 2, 1, 1, 0, 0, 0, 0], (10, 16, 22, 28), RED),
    ("D\\ (5,0)-(2,3) Y",
     [1, 4, 3, 2, 2, 2, 2, 1, 3, 0, 0, 0, 3, 0, 1, 0, 0, 1, 1],
     (17, 23, 29, 35), YELLOW),
    ("D\\ (5,3)-(2,6) R",
     [3, 3, 5, 4, 4, 4, 3, 4, 6, 4, 3, 3, 6, 3, 5, 6, 5, 5],
     (20, 26, 32, 38), RED),
    ("D\\ (4,2)-(1,5) Y",
     [2, 3, 2, 2, 4, 2, 4, 3, 2, 3, 3, 5, 5, 0, 4], (12, 18, 24, 30), YELLOW),
    ("D\\ (5,1)-(2,4) R",
     [1, 3, 1, 3, 4, 1, 3, 4, 1, 2, 1, 1, 2, 4, 2, 3, 2, 2],
     (18, 24, 30, 36), RED),
    ("D\\ (4,3)-(1,6) Y",
     [4, 4, 6, 5, 6, 3, 3, 2, 3, 5, 5, 4, 4, 3, 3], (13, 19, 25, 31), YELLOW),
]

# A random game that ends 42-0 with no alignment.
DRAW_MOVES = [3, 0, 5, 4, 3, 4, 2, 6, 4, 1, 1, 0, 6, 4, 1, 2, 3, 6, 5, 2, 6, 1,
              6, 4, 2, 3, 4, 6, 1, 3, 2, 2, 5, 1, 3, 5, 5, 0, 0, 0, 5, 0]


def test_forty_alignments_cover_all_four_directions():
    """Sanity check on the test data itself, not on the engine."""
    assert len(WIN_CASES) == 40
    dirs = {"H": 0, "V": 0, "D/": 0, "D\\": 0}
    for name, _, _, _ in WIN_CASES:
        dirs[name.split()[0]] += 1
    assert dirs == {"H": 10, "V": 10, "D/": 10, "D\\": 10}
    # top row and both side edges are represented
    all_cells = {c for _, _, line, _ in WIN_CASES for c in line}
    assert any(c // COLS == ROWS - 1 for c in all_cells), "top row not covered"
    assert any(c % COLS == 0 for c in all_cells), "left edge not covered"
    assert any(c % COLS == COLS - 1 for c in all_cells), "right edge not covered"
    assert {p for _, _, _, p in WIN_CASES} == {YELLOW, RED}


@pytest.mark.parametrize("name,seq,line,winner", WIN_CASES, ids=[c[0] for c in WIN_CASES])
def test_win_detection(name, seq, line, winner):
    b = Board()
    for k, col in enumerate(seq):
        assert not b.terminal(), f"{name}: game ended early at move {k}"
        b.play(col)
        if k < len(seq) - 1:
            assert b.winner is None, f"{name}: premature win at move {k}"
    # the incremental detector agrees with the exhaustive scan
    assert b.winner == winner, name
    assert b.terminal()
    found = scan_wins(b.cells)
    assert len(found) == 1, f"{name}: expected exactly one alignment, got {found}"
    assert found[0][0] == winner
    assert set(found[0][1]) == set(line), name
    # score convention: [0, 1] from YELLOW's point of view
    assert b.score() == (1.0 if winner == YELLOW else 0.0)


def test_draw_at_42_stones():
    b = from_moves(DRAW_MOVES)
    assert b.moves == NCELLS == 42
    assert b.winner is None
    assert scan_wins(b.cells) == []
    assert b.terminal()
    assert b.score() == 0.5
    assert b.legalMoves() == []


def test_full_column_is_rejected():
    # Y R Y R Y R stacked in column 3: nobody aligns four vertically.
    b = from_moves([3] * ROWS)
    assert b.heights[3] == ROWS
    assert b.winner is None
    assert 3 not in b.legalMoves()
    with pytest.raises(ValueError):
        b.play(3)
    # every other column is still playable
    for c in range(COLS):
        if c != 3:
            b.copy().play(c)


def test_legal_moves_never_returns_a_full_column():
    rng = random.Random(7)
    for _ in range(2000):
        b = Board()
        while not b.terminal():
            legal = b.legalMoves()
            assert legal, "no legal move in a non-terminal position"
            for c in legal:
                assert b.heights[c] < ROWS
            assert len(set(legal)) == len(legal)
            b.play(rng.choice(legal))
        # once full, every column is rejected
        if b.moves == NCELLS:
            assert b.legalMoves() == []


def test_legal_moves_are_center_first():
    b = Board()
    assert b.legalMoves() == [3, 2, 4, 1, 5, 0, 6]


def test_random_selfplay_5000_games():
    """5 000 random games: no exception, no illegal move, coherent results."""
    rng = random.Random(2026)
    results = {1.0: 0, 0.0: 0, 0.5: 0}
    for _ in range(5000):
        b = Board()
        while not b.terminal():
            legal = b.legalMoves()
            col = rng.choice(legal)
            assert b.heights[col] < ROWS
            b.play(col)
        assert 0 < b.moves <= NCELLS
        # incremental winner == exhaustive scan
        found = scan_wins(b.cells)
        if b.winner is None:
            assert found == []
            assert b.moves == NCELLS
        else:
            assert found and all(p == b.winner for p, _ in found)
        results[b.score()] += 1
    assert sum(results.values()) == 5000
    # YELLOW moves first, so it must win noticeably more often than RED
    assert results[1.0] > results[0.0]


def test_zobrist_incrementality():
    """`h` maintained by XOR == `h` recomputed from scratch, at every ply."""
    rng = random.Random(31337)
    for _ in range(2000):
        b = Board()
        assert b.h == b.compute_hash() == 0
        while not b.terminal():
            b.play(rng.choice(b.legalMoves()))
            assert b.h == b.compute_hash()


def test_zobrist_unplay_restores_hash():
    rng = random.Random(99)
    for _ in range(500):
        b = Board()
        hashes = [b.h]
        while not b.terminal():
            b.play(rng.choice(b.legalMoves()))
            hashes.append(b.h)
        while b.moves:
            b.unplay()
            hashes.pop()
            assert b.h == hashes[-1] == b.compute_hash()
        assert b == Board()


def test_zobrist_no_collision_over_200k_positions():
    """0 collisions expected: 200 000 positions in a 2^64 space."""
    rng = random.Random(4242)
    seen: dict[int, tuple] = {}
    collisions = 0
    while len(seen) < 200_000:
        b = Board()
        while True:
            key = b.key()
            prev = seen.get(b.h)
            if prev is None:
                seen[b.h] = key
            elif prev != key:
                collisions += 1
            if b.terminal():
                break
            b.play(rng.choice(b.legalMoves()))
    assert collisions == 0, f"{collisions} Zobrist collisions"
    assert len(seen) >= 200_000


def test_zobrist_distinguishes_side_to_move():
    """Two boards with the same stones but a different player to move differ."""
    a = from_moves([0, 1, 2])  # RED to move
    c = from_moves([0, 1, 2])
    assert a.h == c.h
    assert a.turn == RED
    # same cells reached with an extra pair of moves is a different position
    d = from_moves([0, 1, 2, 5])
    assert d.h != a.h


def test_zobrist_transpositions_collide_on_purpose():
    """Different move orders reaching the same position must share a hash."""
    a = from_moves([0, 1, 2, 3])
    b = from_moves([2, 3, 0, 1])
    assert a.cells == b.cells and a.turn == b.turn
    assert a.h == b.h


def test_mirror_involution():
    rng = random.Random(5150)
    for _ in range(3000):
        b = Board()
        depth = rng.randrange(0, 20)
        for _ in range(depth):
            if b.terminal():
                break
            b.play(rng.choice(b.legalMoves()))
        m = b.mirror()
        assert m.mirror() == b
        assert m.h == m.compute_hash()
        assert m.moves == b.moves and m.turn == b.turn and m.winner == b.winner
        np.testing.assert_array_equal(m.board, b.board[:, ::-1])


def test_mirror_of_symmetric_position_is_itself():
    b = from_moves([3, 3, 3])
    assert b.mirror() == b


def test_board_property_shape_and_dtype():
    b = from_moves([3, 3, 2])
    arr = b.board
    assert arr.shape == (ROWS, COLS)
    assert arr.dtype == np.int8
    assert arr[0, 3] == YELLOW and arr[1, 3] == RED and arr[0, 2] == YELLOW
    assert arr[5, 0] == EMPTY


def test_copy_is_independent():
    b = from_moves([3, 2, 3])
    c = b.copy()
    assert c == b
    c.play(0)
    assert c != b
    assert b.moves == 3 and c.moves == 4


def test_move_codes_are_84_and_unique():
    codes = {move_code(r, c, p)
             for r in range(ROWS) for c in range(COLS) for p in (YELLOW, RED)}
    assert len(codes) == AMAF_CODES == 84
    assert min(codes) == 0 and max(codes) == 83


def test_board_move_code_matches_the_move_actually_played():
    rng = random.Random(11)
    for _ in range(500):
        b = Board()
        while not b.terminal():
            col = rng.choice(b.legalMoves())
            code = b.move_code(col)
            player, row = b.turn, b.heights[col]
            b.play(col)
            assert code == move_code(row, col, player)
            assert b.last == (row, col)


def test_playout_amaf_collects_every_move_of_the_playout():
    """`playoutAMAF` appends exactly one valid code per move it plays."""
    rng = random.Random(3)
    for _ in range(500):
        b = Board()
        for _ in range(rng.randrange(0, 12)):
            if b.terminal():
                break
            b.play(rng.choice(b.legalMoves()))
        start = b.moves
        played: list[int] = []
        s = b.playoutAMAF(played)
        assert b.terminal()
        assert len(played) == b.moves - start
        assert all(0 <= c < AMAF_CODES for c in played)
        assert len(set(played)) == len(played), "a cell was filled twice"
        assert s == b.score() and s in (0.0, 0.5, 1.0)
        # the codes describe cells that really are occupied by that player
        for code in played:
            player, cell = divmod(code, NCELLS)
            assert b.cells[cell] == player + 1


def test_playout_reaches_a_terminal_position():
    for _ in range(2000):
        b = Board()
        s = b.playout()
        assert b.terminal()
        assert s == b.score()


def test_unplay_after_a_playout_raises_instead_of_corrupting():
    """`playout` records no undo information, so `unplay` must fail loudly."""
    b = from_moves([3, 3, 2])
    b.play(4)
    b.unplay()          # a real move can be undone
    assert b.h == b.compute_hash()
    b.playout()
    with pytest.raises(IndexError):
        b.unplay()
    # a playout leaves a coherent position, it is only unplay that is forbidden
    assert b.h == b.compute_hash()
    assert b.terminal()


def test_playout_amaf_also_clears_the_undo_stack():
    b = from_moves([3, 2])
    b.playoutAMAF([])
    with pytest.raises(IndexError):
        b.unplay()
    assert b.h == b.compute_hash()
