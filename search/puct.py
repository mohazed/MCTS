"""PUCT: MCTS guided by the policy + value network (AlphaGo Zero / AlphaZero).

Same recursive shape as the course's `UCT`, with three changes -- and only
those three, which are exactly what the course says AlphaGo Zero adds:

  1. the selection formula becomes
         val = Q + c_puct * prior[m] * sqrt(t.n) / (1 + t.nplayouts[m])
  2. a leaf is NOT evaluated by a random playout but by the value head; the
     policy head gives the priors of its children.  There is no rollout at all.
  3. at the root, and ONLY during self-play, the priors are mixed with
     Dirichlet noise:  P <- 0.75 P + 0.25 Dir(alpha).

Conventions.  Everything stays in the course's [0, 1] / POV-YELLOW world:
`t.value` stores the network value already converted with `from_pov`, so the
usual `if board.turn == RED: Q = 1 - Q` applies unchanged, and the value that
comes back up the recursion is a [0, 1] score exactly like a playout result.

`Q` of an unvisited move is `t.value`, the network value of the node -- the
course's `Q = t[4]`.
"""

from __future__ import annotations

import math
import time

import numpy as np

from game.connect4 import COLS, RED, Board
from model.encode import evaluate, from_pov
from search.base import empty_info
from search.tt import Entry, TranspositionTable

DEFAULT_C_PUCT = 1.0
DIRICHLET_ALPHA = 1.0
DIRICHLET_EPS = 0.25


def expand(board: Board, table: TranspositionTable, net) -> tuple[Entry, float]:
    """Create the entry of `board`, fill in priors and value from the network."""
    t = table.add(board)
    prior, value = evaluate(net, board)
    t.prior = prior
    t.value = from_pov(value, board.turn)  # [0,1] POV YELLOW
    return t, t.value


def add_dirichlet_noise(
    prior: np.ndarray,
    legal: list[int],
    alpha: float = DIRICHLET_ALPHA,
    eps: float = DIRICHLET_EPS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """P <- (1-eps) P + eps Dir(alpha), on the legal moves only."""
    rng = rng or np.random.default_rng()
    noise = rng.dirichlet([alpha] * len(legal))
    out = prior.copy()
    out[legal] = (1.0 - eps) * prior[legal] + eps * noise
    s = out.sum()
    if s > 0:
        out /= s
    return out


def PUCT(
    board: Board, table: TranspositionTable, net, c_puct: float = DEFAULT_C_PUCT
) -> float:
    """One PUCT simulation from `board` (mutated).  Returns [0,1] POV YELLOW."""
    if board.terminal():
        return board.score()
    t = table.look(board)
    if t is None:
        _, value = expand(board, table, net)
        return value
    moves = board.legalMoves()
    sqrt_n = math.sqrt(t.n)
    black = board.turn == RED
    prior = t.prior
    nplayouts = t.nplayouts
    nwins = t.nwins
    best_value = -1e9
    best = moves[0]
    for m in moves:
        ni = nplayouts[m]
        Q = (nwins[m] / ni) if ni > 0 else t.value
        if black:
            Q = 1.0 - Q
        val = Q + c_puct * prior[m] * sqrt_n / (1.0 + ni)
        if val > best_value:
            best_value = val
            best = m
    board.play(best)
    res = PUCT(board, table, net, c_puct)
    t.n += 1
    nplayouts[best] += 1.0
    nwins[best] += res
    return res


def BestMovePUCT(
    board: Board,
    net,
    sims: int,
    c_puct: float = DEFAULT_C_PUCT,
    training: bool = False,
    dirichlet_alpha: float = DIRICHLET_ALPHA,
    dirichlet_eps: float = DIRICHLET_EPS,
    rng: np.random.Generator | None = None,
) -> tuple[int, np.ndarray, np.ndarray, float, TranspositionTable]:
    """`sims` PUCT simulations; returns (best move, visits, raw priors, value, table).

    `visits` is the raw visit count of each column: that is what self-play turns
    into the training target pi, and what the web front draws over the priors.
    The Dirichlet noise is applied to the priors *used by the search*, so the
    visit distribution -- and therefore pi -- already reflects it, as required.
    The priors returned for display are the noiseless ones.
    """
    table = TranspositionTable()
    t, _ = expand(board, table, net)
    raw_prior = t.prior.copy()
    if training:
        t.prior = add_dirichlet_noise(
            t.prior, board.legalMoves(), dirichlet_alpha, dirichlet_eps, rng
        )
    for _ in range(sims):
        b = board.copy()
        PUCT(b, table, net, c_puct)
    visits = t.nplayouts.copy()
    moves = board.legalMoves()
    if visits.sum() == 0:  # sims == 0
        best = int(max(moves, key=lambda m: raw_prior[m]))
    else:
        best = int(max(moves, key=lambda m: visits[m]))
    # value of the root, back in [-1, 1] from the point of view of the mover
    value = 2.0 * t.value - 1.0
    if board.turn == RED:
        value = -value
    return best, visits, raw_prior, value, table


class PUCTPlayer:
    def __init__(
        self,
        net,
        sims: int = 100,
        c_puct: float = DEFAULT_C_PUCT,
        training: bool = False,
        name: str | None = None,
    ) -> None:
        self.net = net
        self.sims = sims
        self.c_puct = c_puct
        self.training = training
        self.name = name or f"puct-{sims}"

    def choose_move(self, board: Board) -> tuple[int, dict]:
        t0 = time.perf_counter()
        col, visits, prior, value, table = BestMovePUCT(
            board, self.net, self.sims, self.c_puct, training=self.training
        )
        return col, empty_info(
            visits=visits,
            priors=prior,
            value=value,
            time_ms=(time.perf_counter() - t0) * 1e3,
            tt_hits=table.hits,
            tt_lookups=table.lookups,
        )
