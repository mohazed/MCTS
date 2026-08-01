"""Board <-> tensor encoding, sign conventions, and mirror symmetry.

THE SIGN CONVENTIONS ARE THE NUMBER ONE SOURCE OF BUGS IN THIS PROJECT, so
they live here, in one place, and are tested unitarily (tests/test_model.py):

  * `Board.score()` is in [0, 1] from YELLOW's point of view (course convention)
  * the VALUE HEAD outputs in [-1, 1] from the POINT OF VIEW OF THE PLAYER TO MOVE
  * `to_pov(score, turn)` is the ONLY bridge between the two, used everywhere

Encoding (following the course's "rotate the board for white so
that moves are always forward"): 3 planes of 6x7 float32, from the point of view
of the player to move -- own stones, opponent stones, constant 1.0.  The network
therefore never needs to know *which* colour it is playing.
"""

from __future__ import annotations

import numpy as np
import torch

from game.connect4 import COLS, EMPTY, RED, ROWS, YELLOW, Board

PLANES = 3


def to_pov(score: float, turn: int) -> float:
    """Convert a [0,1] score POV YELLOW into a [-1,1] value POV `turn`.

    to_pov(1.0, YELLOW) = +1   YELLOW won and YELLOW was to move  -> good
    to_pov(1.0, RED)    = -1   YELLOW won and RED was to move     -> bad
    to_pov(0.5, *)      =  0   draw
    """
    if turn == YELLOW:
        return 2.0 * score - 1.0
    return 1.0 - 2.0 * score


def from_pov(value: float, turn: int) -> float:
    """Inverse of `to_pov`: [-1,1] POV `turn` -> [0,1] POV YELLOW."""
    if turn == YELLOW:
        return (value + 1.0) / 2.0
    return (1.0 - value) / 2.0


def encode(board: Board) -> np.ndarray:
    """(3, 6, 7) float32 planes from the point of view of the player to move."""
    cells = np.asarray(board.cells, dtype=np.int8).reshape(ROWS, COLS)
    me = board.turn
    opp = RED if me == YELLOW else YELLOW
    planes = np.empty((PLANES, ROWS, COLS), dtype=np.float32)
    planes[0] = cells == me
    planes[1] = cells == opp
    planes[2] = 1.0
    return planes


def mirror_planes(planes: np.ndarray) -> np.ndarray:
    """Left/right mirror of the encoded position (last axis = columns)."""
    return np.ascontiguousarray(planes[..., ::-1])


def mirror_pi(pi: np.ndarray) -> np.ndarray:
    """Left/right mirror of a policy vector.  MUST be applied together with
    `mirror_planes` -- forgetting it teaches the network a mirrored policy."""
    return np.ascontiguousarray(pi[..., ::-1])


def masked_softmax(logits: np.ndarray, legal: list[int]) -> np.ndarray:
    """Softmax restricted to `legal`; illegal columns get exactly 0.0."""
    out = np.zeros(COLS, dtype=np.float64)
    if not legal:
        return out
    sub = logits[legal].astype(np.float64)
    sub = sub - sub.max()
    e = np.exp(sub)
    out[legal] = e / e.sum()
    return out


@torch.no_grad()
def evaluate(net, board: Board) -> tuple[np.ndarray, float]:
    """Network evaluation of one position.

    Returns (prior over the 7 columns with 0 on illegal ones, value in [-1,1]
    from the point of view of the player to move).
    """
    x = torch.from_numpy(encode(board)).unsqueeze(0)
    logits, value = net(x)
    prior = masked_softmax(logits[0].numpy(), board.legalMoves())
    return prior, float(value.item())


@torch.no_grad()
def evaluate_batch(net, boards: list[Board]) -> tuple[np.ndarray, np.ndarray]:
    """Same as `evaluate` for several positions (used by the test set only)."""
    x = torch.from_numpy(np.stack([encode(b) for b in boards]))
    logits, values = net(x)
    logits = logits.numpy()
    priors = np.stack(
        [masked_softmax(logits[i], b.legalMoves()) for i, b in enumerate(boards)]
    )
    return priors, values.numpy().reshape(-1)
