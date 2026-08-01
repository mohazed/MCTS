"""Step 2/3/6/9 validation: every search algorithm behaves like a search.

The three properties tested on all of them:
  * they never return an illegal move;
  * at 200 playouts they take a win that is available immediately;
  * at 200 playouts they block a loss that is otherwise immediate.

Plus the course's own non-regression test: GRAVE with `ref = 0` IS RAVE.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from eval.baselines import AlphaBetaPlayer, RandomPlayer
from game.connect4 import COLS, RED, ROWS, YELLOW, Board, from_moves
from model.net import new_net
from search.flat import FlatPlayer, UCBPlayer, UCB, flat
from search.grave import BestMoveGRAVE, BestMoveRAVE, GRAVEPlayer, updateAMAF
from search.puct import BestMovePUCT, PUCTPlayer
from search.tt import Entry, TranspositionTable
from search.uct import BestMoveUCT, UCTPlayer

torch.set_num_threads(1)
PLAYOUTS = 200
# Legality does not depend on the budget, so the 200-position sweep uses a
# small one; the tactical tests below use the full 200 playouts.
PLAYOUTS_LEGAL = 25


# --------------------------------------------------------------------------
# position generators
# --------------------------------------------------------------------------
def immediate_wins(b: Board) -> list[int]:
    me = b.turn
    out = []
    for m in b.legalMoves():
        b.play(m)
        if b.winner == me:
            out.append(m)
        b.unplay()
    return out


def random_positions(n: int, seed: int = 0) -> list[Board]:
    """Non-terminal positions of varied depth."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = Board()
        for _ in range(rng.randrange(0, 34)):
            if b.terminal():
                break
            b.play(rng.choice(b.legalMoves()))
        if not b.terminal():
            out.append(b)
    return out


def win_positions(n: int, seed: int = 1) -> list[tuple[Board, list[int]]]:
    """Positions where the player to move has an immediate win."""
    rng = random.Random(seed)
    out = []
    guard = 0
    while len(out) < n and guard < 100_000:
        guard += 1
        b = Board()
        while not b.terminal():
            w = immediate_wins(b)
            if w:
                out.append((b.copy(), w))
                break
            b.play(rng.choice(b.legalMoves()))
    return out


def opponent_threats(b: Board) -> list[int]:
    """Columns where the OPPONENT -- who is *not* to move -- would win at once.

    Implemented as a null move: flip the side to move on a throwaway copy and
    ask the same question.  (The copy's Zobrist hash is then inconsistent, which
    does not matter, nothing hashes it.)
    """
    c = b.copy()
    c.turn = RED if c.turn == YELLOW else YELLOW
    return immediate_wins(c)


def block_positions(n: int, seed: int = 2) -> list[tuple[Board, int]]:
    """Positions with exactly one move that does not lose on the spot.

    Conditions: the player to move has no immediate win; the opponent has
    exactly ONE column c where they would win immediately; and after we play c
    the opponent has no immediate win left.  Playing anything other than c
    leaves the threat standing, so c is unambiguously the only move, and any
    correct search must find it.
    """
    rng = random.Random(seed)
    out: list[tuple[Board, int]] = []
    guard = 0
    while len(out) < n and guard < 200_000:
        guard += 1
        b = Board()
        while not b.terminal():
            if immediate_wins(b):
                break
            threats = opponent_threats(b)
            if len(threats) == 1:
                c = threats[0]
                after = b.copy()
                after.play(c)
                if not after.terminal() and not immediate_wins(after):
                    out.append((b.copy(), c))
                    break
            b.play(rng.choice(b.legalMoves()))
    return out


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return new_net(channels=16, blocks=1)


@pytest.fixture(scope="module")
def agents(net):
    return [
        FlatPlayer(PLAYOUTS_LEGAL),
        UCBPlayer(PLAYOUTS_LEGAL),
        UCTPlayer(PLAYOUTS_LEGAL),
        GRAVEPlayer(PLAYOUTS_LEGAL, ref=50),
        GRAVEPlayer(PLAYOUTS_LEGAL, ref=0),
        PUCTPlayer(net, sims=PLAYOUTS_LEGAL),
    ]


def make_agent(name: str, net, budget: int = PLAYOUTS):
    return {
        "flat": lambda: FlatPlayer(budget),
        "ucb": lambda: UCBPlayer(budget),
        "uct": lambda: UCTPlayer(budget),
        "grave": lambda: GRAVEPlayer(budget, ref=50),
        "rave": lambda: GRAVEPlayer(budget, ref=0),
        "puct": lambda: PUCTPlayer(net, sims=budget),
    }[name]()


@pytest.fixture(scope="module")
def wins30():
    return win_positions(30, seed=1)


@pytest.fixture(scope="module")
def blocks30():
    return block_positions(30, seed=2)


# --------------------------------------------------------------------------
# legality
# --------------------------------------------------------------------------
def test_every_agent_returns_a_legal_move_on_200_random_positions(agents):
    positions = random_positions(200, seed=17)
    assert len(positions) == 200
    for agent in agents:
        for b in positions:
            before = b.copy()
            col, info = agent.choose_move(b)
            assert col in b.legalMoves(), f"{agent.name} played illegal {col}"
            assert b == before, f"{agent.name} mutated the board it was given"
            assert info["visits"].shape == (COLS,)
            assert info["time_ms"] >= 0.0


def test_visits_are_zero_on_illegal_columns(agents):
    for b in random_positions(30, seed=23):
        legal = set(b.legalMoves())
        for agent in agents:
            _, info = agent.choose_move(b)
            for c in range(COLS):
                if c not in legal:
                    assert info["visits"][c] == 0.0, agent.name


# --------------------------------------------------------------------------
# tactics
# --------------------------------------------------------------------------
def test_position_generators_are_sound(wins30, blocks30):
    assert len(wins30) == 30
    for b, w in wins30:
        assert w and all(m in b.legalMoves() for m in w)
        for m in w:
            c = b.copy()
            mover = c.turn
            c.play(m)
            assert c.winner == mover
    assert len(blocks30) == 30
    for b, c in blocks30:
        assert not immediate_wins(b), "the player to move should have no win"
        for m in b.legalMoves():
            after = b.copy()
            after.play(m)
            lost = bool(immediate_wins(after))
            assert lost == (m != c), f"move {m} should {'not ' if m == c else ''}lose"


@pytest.mark.parametrize("name", ["flat", "ucb", "uct", "grave", "rave", "puct"])
def test_agent_takes_an_immediate_win(name, net, wins30):
    agent = make_agent(name, net)
    random.seed(0)
    ok = sum(agent.choose_move(b)[0] in w for b, w in wins30)
    assert ok == len(wins30), f"{agent.name} missed {len(wins30) - ok} wins"


# Blocking is where the algorithms genuinely differ.  Measured on these 30
# positions (immediate wins taken / immediate losses blocked, out of 30):
#
#            200 playouts   400        1000
#   flat        30/23      30/26      30/28
#   ucb         30/26      30/26      30/27
#   uct         30/29      30/29      30/30
#   grave       30/29      30/30      30/30
#   rave        30/30      30/30      30/30
#   puct        30/30      30/30      30/30
#
# Flat Monte Carlo and root UCB are genuinely unreliable at blocking, and this
# is NOT a bug: not blocking still scores about 0.5 in a random playout, because
# a random opponent frequently fails to punish, and with 200/7 = 28 playouts per
# move that difference drowns in the noise.  Only a tree search, which spends
# its budget where it matters, resolves it.  This is precisely the reason UCT
# beats flat Monte Carlo in E1.
BLOCK_MIN = {"flat": 21, "ucb": 23, "uct": 28, "grave": 28, "rave": 28, "puct": 28}


@pytest.mark.parametrize("name", ["flat", "ucb", "uct", "grave", "rave", "puct"])
def test_agent_blocks_an_immediate_loss(name, net, blocks30):
    agent = make_agent(name, net)
    random.seed(0)
    ok = sum(agent.choose_move(b)[0] == c for b, c in blocks30)
    assert ok >= BLOCK_MIN[name], (
        f"{agent.name} blocked only {ok}/{len(blocks30)}, expected "
        f">= {BLOCK_MIN[name]}"
    )


@pytest.mark.parametrize("name", ["uct", "grave", "rave", "puct"])
def test_tree_search_blocks_almost_perfectly(name, net, blocks30):
    """The tree searches must be essentially perfect on this trivial tactic."""
    agent = make_agent(name, net)
    random.seed(0)
    ok = sum(agent.choose_move(b)[0] == c for b, c in blocks30)
    assert ok >= len(blocks30) - 1, f"{agent.name} blocked only {ok}/30"


def test_tree_search_blocks_better_than_flat_monte_carlo(net, blocks30):
    """The ordering flat < UCB < UCT is the whole point of the course's ladder."""
    def rate(name):
        agent = make_agent(name, net)
        random.seed(0)
        return sum(agent.choose_move(b)[0] == c for b, c in blocks30)

    flat, ucb, uct = rate("flat"), rate("ucb"), rate("uct")
    assert flat < uct, f"flat {flat} should block worse than UCT {uct}"
    assert ucb <= uct, f"UCB {ucb} should not block better than UCT {uct}"


def test_more_playouts_fix_the_blocks_flat_monte_carlo_misses(blocks30):
    """UCT reaches 30/30 at 1000 playouts: it is a budget issue, not a bug."""
    agent = UCTPlayer(1000)
    random.seed(0)
    ok = sum(agent.choose_move(b)[0] == c for b, c in blocks30)
    assert ok == len(blocks30)


def test_alphabeta_is_tactically_perfect(wins30, blocks30):
    agent = AlphaBetaPlayer(4)
    for b, w in wins30:
        assert agent.choose_move(b)[0] in w
    for b, c in blocks30:
        assert agent.choose_move(b)[0] == c


# --------------------------------------------------------------------------
# GRAVE(ref=0) == RAVE  -- the course's non-regression test
# --------------------------------------------------------------------------
def test_grave_with_ref_zero_reproduces_rave_exactly():
    """Same visit distribution, on the same seed, on 25 positions."""
    positions = random_positions(25, seed=5)
    for i, b in enumerate(positions):
        random.seed(1000 + i)
        _, v_grave, _ = BestMoveGRAVE(b, 300, ref=0)
        random.seed(1000 + i)
        _, v_rave, _ = BestMoveRAVE(b, 300)
        np.testing.assert_array_equal(
            v_grave, v_rave, err_msg=f"GRAVE(ref=0) != RAVE on position {i}"
        )


def test_grave_with_positive_ref_differs_from_rave():
    """Sanity check on the previous test: ref=50 is NOT the same algorithm."""
    positions = random_positions(25, seed=5)
    diff = 0
    for i, b in enumerate(positions):
        random.seed(1000 + i)
        _, v50, _ = BestMoveGRAVE(b, 300, ref=50)
        random.seed(1000 + i)
        _, v0, _ = BestMoveGRAVE(b, 300, ref=0)
        diff += int(not np.array_equal(v50, v0))
    assert diff > 0, "ref has no effect at all -- GRAVE is not implemented"


def test_update_amaf_counts_only_the_first_occurrence():
    t = Entry(amaf=True)
    updateAMAF(t, [3, 7, 3, 7, 9], 1.0)
    assert t.namaf[3] == 1.0 and t.namaf[7] == 1.0 and t.namaf[9] == 1.0
    assert t.wamaf[3] == 1.0
    assert t.namaf.sum() == 3.0


def test_amaf_arrays_are_only_allocated_when_asked():
    assert Entry(amaf=False).namaf is None
    assert Entry(amaf=True).namaf.shape == (84,)


# --------------------------------------------------------------------------
# transposition table
# --------------------------------------------------------------------------
def test_transposition_table_records_hits():
    tt = TranspositionTable()
    b = Board()
    assert tt.look(b) is None
    tt.add(b)
    assert tt.look(b) is not None
    assert tt.lookups == 2 and tt.hits == 1
    assert tt.hit_rate == 0.5


def test_transpositions_share_one_table_entry():
    """The table is keyed on board.h, so two move orders reaching the same
    position share the same Entry.

    Note it takes FOUR moves to transpose in Connect 4: [3,2] and [2,3] give the
    two colours opposite cells, so they are different positions.
    """
    x, y = from_moves([0, 1, 2, 3]), from_moves([2, 3, 0, 1])
    assert x.cells == y.cells and x.turn == y.turn
    assert x.h == y.h
    tt = TranspositionTable()
    e = tt.add(x)
    assert tt.look(y) is e, "a transposition was not recognised"
    assert len(tt) == 1


def test_uct_reuses_transpositions():
    """During a real search most lookups hit an existing entry.

    (Table size is not the evidence: each simulation adds exactly one leaf, so
    len(table) == number of simulations regardless.  The hit rate is.)
    """
    b = Board()
    _, _, table = BestMoveUCT(b, 600)
    assert table.hit_rate > 0.5
    assert len(table) == 600


# --------------------------------------------------------------------------
# determinism / budget
# --------------------------------------------------------------------------
def test_the_whole_budget_reaches_the_root():
    b = Board()
    for n in (50, 200):
        _, visits, _ = BestMoveUCT(b, n)
        # the FIRST simulation only creates the root and plays one playout, so
        # it increments nothing: the root collects n - 1 visits (course's UCT).
        assert visits.sum() == n - 1
    # PUCT expands the root before the loop, so all n simulations descend
    _, visits, _, _, _ = BestMovePUCT(b, new_net(16, 1), 137)
    assert visits.sum() == 137


def test_flat_splits_its_budget_equally():
    b = Board()
    _, visits, _ = flat(b, 210)
    assert set(visits[visits > 0]) == {30.0}
    assert visits.sum() == 210


def test_ucb_spends_its_whole_budget():
    b = Board()
    _, visits, _ = UCB(b, 200)
    assert visits.sum() == 200


# --------------------------------------------------------------------------
# arena protocol
# --------------------------------------------------------------------------
def test_balanced_schedule_is_colour_balanced():
    from eval.arena import balanced_schedule

    sched = balanced_schedule(k=2, depth=1)
    assert len(sched) == 28
    assert sum(1 for _, ay in sched if ay) == 14, "colours must be balanced"
    assert len({o for o, _ in sched}) == 7
    deep = balanced_schedule(k=1, depth=2)
    assert len(deep) == 98
    assert len({o for o, _ in deep}) == 49
    assert sum(1 for _, ay in deep if ay) == 49


def test_repeating_an_opening_between_deterministic_agents_gives_the_same_game():
    """Why `openings_depth=2` exists.

    Two deterministic agents replay the identical game for a given (opening,
    colour), so the k repeats of the 14k-game protocol are duplicates and
    the Wilson interval computed on 14k would be far too narrow.
    """
    from eval.arena import balanced_schedule, play_game

    a, b = AlphaBetaPlayer(2), AlphaBetaPlayer(1)
    s1 = play_game(a, b, (3,), seed=1)
    s2 = play_game(a, b, (3,), seed=999)  # different seed, same game
    assert s1 == s2
    # widening the opening book really does give distinct positions
    deep = balanced_schedule(k=1, depth=2)
    assert len({o for o, _ in deep}) == 49


def test_wilson_interval_matches_known_values():
    from eval.arena import elo_diff, wilson

    lo, hi = wilson(0, 28)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0.10 < hi < 0.13
    lo, hi = wilson(14, 28)
    assert lo == pytest.approx(0.5 - (hi - 0.5), abs=1e-9), "must be symmetric at p=.5"
    lo, hi = wilson(98, 98)
    assert hi == pytest.approx(1.0) and lo > 0.95
    # the interval must shrink as the number of games grows
    assert (wilson(49, 98)[1] - wilson(49, 98)[0]) < (wilson(14, 28)[1] - wilson(14, 28)[0])
    assert elo_diff(0.5) == pytest.approx(0.0)
    assert elo_diff(0.75) == pytest.approx(190.8, abs=1.0)
    assert elo_diff(0.25) == pytest.approx(-190.8, abs=1.0)
