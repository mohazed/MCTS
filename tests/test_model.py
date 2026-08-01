"""Step 5 validation: encoding, sign conventions, symmetry, and the network.

The three checks at the bottom (overfit / value(empty) / value(win next move))
are the ones we identified up front as the number one risk: a flipped `z` makes
the network learn to LOSE, and every other metric still looks plausible.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from eval.baselines import AlphaBetaPlayer
from game.connect4 import COLS, RED, ROWS, YELLOW, Board, from_moves
from model.encode import (
    PLANES,
    encode,
    evaluate,
    from_pov,
    masked_softmax,
    mirror_pi,
    mirror_planes,
    to_pov,
)
from model.net import Connect4Net, alphazero_loss, load_net, new_net, save_net

torch.set_num_threads(1)


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------
def test_encoding_shape_and_dtype():
    planes = encode(Board())
    assert planes.shape == (PLANES, ROWS, COLS)
    assert planes.dtype == np.float32
    assert planes[0].sum() == 0 and planes[1].sum() == 0
    assert np.all(planes[2] == 1.0)


def test_encoding_is_from_the_point_of_view_of_the_player_to_move():
    b = from_moves([3])  # YELLOW played col 3, RED to move
    assert b.turn == RED
    p = encode(b)
    assert p[0].sum() == 0.0, "RED (to move) has no stone yet"
    assert p[1][0, 3] == 1.0, "the YELLOW stone is on the opponent plane"
    b.play(2)  # RED plays col 2, YELLOW to move
    p = encode(b)
    assert p[0][0, 3] == 1.0 and p[1][0, 2] == 1.0
    assert p[0].sum() == 1.0 and p[1].sum() == 1.0


def test_encoding_planes_are_disjoint_and_cover_the_stones():
    rng = random.Random(1)
    for _ in range(500):
        b = Board()
        for _ in range(rng.randrange(0, 30)):
            if b.terminal():
                break
            b.play(rng.choice(b.legalMoves()))
        p = encode(b)
        assert np.all(p[0] * p[1] == 0.0)
        assert p[0].sum() + p[1].sum() == b.moves


# --------------------------------------------------------------------------
# sign conventions -- to_pov
# --------------------------------------------------------------------------
def test_to_pov_explicit_table():
    assert to_pov(1.0, YELLOW) == 1.0  # YELLOW won, YELLOW to move -> +1
    assert to_pov(1.0, RED) == -1.0  # YELLOW won, RED to move    -> -1
    assert to_pov(0.0, YELLOW) == -1.0
    assert to_pov(0.0, RED) == 1.0
    assert to_pov(0.5, YELLOW) == 0.0
    assert to_pov(0.5, RED) == 0.0


@pytest.mark.parametrize("turn", [YELLOW, RED])
@pytest.mark.parametrize("score", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_to_pov_round_trips_for_both_players(turn, score):
    assert from_pov(to_pov(score, turn), turn) == pytest.approx(score)


def test_to_pov_is_antisymmetric_in_the_player():
    for s in (0.0, 0.3, 0.5, 1.0):
        assert to_pov(s, YELLOW) == -to_pov(s, RED)


def test_to_pov_on_real_finished_games():
    """The winner's own states must be labelled +1, the loser's -1."""
    rng = random.Random(4)
    checked = 0
    for _ in range(300):
        b = Board()
        turns = []
        while not b.terminal():
            turns.append(b.turn)
            b.play(rng.choice(b.legalMoves()))
        s = b.score()
        if b.winner is None:
            continue
        for t in turns:
            z = to_pov(s, t)
            assert z == (1.0 if t == b.winner else -1.0)
        checked += 1
    assert checked > 100


# --------------------------------------------------------------------------
# symmetry -- must be applied to the planes AND to pi
# --------------------------------------------------------------------------
def test_mirror_planes_matches_mirroring_the_board():
    rng = random.Random(2)
    for _ in range(500):
        b = Board()
        for _ in range(rng.randrange(0, 25)):
            if b.terminal():
                break
            b.play(rng.choice(b.legalMoves()))
        np.testing.assert_array_equal(mirror_planes(encode(b)), encode(b.mirror()))


def test_mirror_pi_reverses_the_columns():
    pi = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05, 0.0])
    m = mirror_pi(pi)
    np.testing.assert_allclose(m, pi[::-1])
    np.testing.assert_allclose(mirror_pi(m), pi)
    assert m.sum() == pytest.approx(1.0)


def test_mirror_is_applied_to_both_planes_and_pi_consistently():
    """The augmented sample must describe the SAME game situation.

    Concretely: the best move of the mirrored position is the mirror of the
    best move of the original one.  If pi were left untouched, this fails.
    """
    b = from_moves([0, 1, 0])  # asymmetric position
    pi = np.zeros(COLS)
    pi[0] = 1.0  # "play column 0"
    planes_m, pi_m = mirror_planes(encode(b)), mirror_pi(pi)
    assert int(np.argmax(pi_m)) == COLS - 1
    np.testing.assert_array_equal(planes_m, encode(b.mirror()))
    # the mirrored board really does have the stones on the other side
    assert b.mirror().cells[COLS - 1] == b.cells[0]


# --------------------------------------------------------------------------
# masked softmax
# --------------------------------------------------------------------------
def test_masked_softmax_gives_zero_probability_to_full_columns():
    b = from_moves([3] * ROWS)  # column 3 is full
    legal = b.legalMoves()
    assert 3 not in legal
    logits = np.array([0.0, 0.0, 0.0, 99.0, 0.0, 0.0, 0.0])  # huge on the full column
    p = masked_softmax(logits, legal)
    assert p[3] == 0.0
    assert p.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(p[legal], 1.0 / len(legal))


def test_masked_softmax_on_random_positions():
    rng = random.Random(6)
    for _ in range(400):
        b = Board()
        for _ in range(rng.randrange(0, 40)):
            if b.terminal():
                break
            b.play(rng.choice(b.legalMoves()))
        if b.terminal():
            continue
        legal = b.legalMoves()
        p = masked_softmax(rng.random() * 10 * np.random.randn(COLS), legal)
        assert p.sum() == pytest.approx(1.0)
        assert np.all(p >= 0)
        for c in range(COLS):
            if c not in legal:
                assert p[c] == 0.0


def test_network_evaluate_respects_legality():
    net = new_net(channels=16, blocks=1)
    b = from_moves([3] * ROWS)
    prior, value = evaluate(net, b)
    assert prior[3] == 0.0
    assert prior.sum() == pytest.approx(1.0)
    assert -1.0 <= value <= 1.0


# --------------------------------------------------------------------------
# the network itself
# --------------------------------------------------------------------------
def test_net_parameter_count_and_shapes():
    net = new_net(channels=64, blocks=3)
    n = net.n_params()
    assert 150_000 < n < 350_000, n
    logits, value = net(torch.zeros(5, PLANES, ROWS, COLS))
    assert logits.shape == (5, COLS)
    assert value.shape == (5, 1)
    assert torch.all(value.abs() <= 1.0)


def test_save_and_load_round_trip(tmp_path):
    net = new_net(channels=16, blocks=2)
    x = torch.randn(3, PLANES, ROWS, COLS)
    p0, v0 = net(x)
    path = str(tmp_path / "n.pt")
    save_net(net, path, iteration=7)
    net2 = load_net(path)
    p1, v1 = net2(x)
    torch.testing.assert_close(p0, p1)
    torch.testing.assert_close(v0, v1)
    assert not net2.training, "a loaded net must be in eval mode (BatchNorm!)"


def make_random_samples(n: int, seed: int = 0):
    """n fixed (planes, pi, z) samples drawn from real, DISTINCT positions.

    Deduplication matters: two copies of the same position with two different
    random targets put an irreducible floor on the MSE, and the overfit test
    would then measure the dataset, not the network.
    """
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    X, P, Z = [], [], []
    seen: set[int] = set()
    while len(X) < n:
        b = Board()
        for _ in range(rng.randrange(0, 20)):
            if b.terminal():
                break
            b.play(rng.choice(b.legalMoves()))
        if b.terminal() or b.h in seen:
            continue
        seen.add(b.h)
        legal = b.legalMoves()
        pi = np.zeros(COLS, dtype=np.float32)
        w = nprng.random(len(legal)) + 0.05
        pi[legal] = w / w.sum()
        X.append(encode(b))
        P.append(pi)
        Z.append(nprng.uniform(-1, 1))
    return (
        torch.from_numpy(np.stack(X)),
        torch.from_numpy(np.stack(P)),
        torch.tensor(Z, dtype=torch.float32),
    )


def test_net_overfits_100_fixed_samples():
    """The network must be able to memorise 100 samples.

    Note on "loss -> 0": with a SOFT target the cross-entropy cannot go below
    the entropy H(pi) of the target itself.  The quantity that must go to zero
    is the excess over that floor, i.e. the KL divergence KL(pi || p_net).  We
    check KL -> 0 and MSE -> 0.
    """
    torch.manual_seed(0)
    X, P, Z = make_random_samples(100, seed=11)
    assert len({X[i].numpy().tobytes() for i in range(len(X))}) == 100
    entropy = float(-(P * torch.log(P.clamp_min(1e-12))).sum(1).mean())
    net = Connect4Net(channels=64, blocks=3)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=0.0)
    first = None
    for _ in range(600):
        opt.zero_grad()
        logits, value = net(X)
        loss, lp, lv = alphazero_loss(logits, value, P, Z)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
    kl = float(lp.detach()) - entropy
    assert float(lv.detach()) < 0.01, f"value MSE did not converge: {float(lv):.4f}"
    assert kl < 0.02, f"policy KL did not converge: {kl:.4f}"
    assert float(loss.detach()) < first


# --------------------------------------------------------------------------
# the decisive sign checks
# --------------------------------------------------------------------------
def immediate_wins(b: Board) -> list[int]:
    """Columns that win on the spot for the player to move."""
    me = b.turn
    out = []
    for m in b.legalMoves():
        b.play(m)
        if b.winner == me:
            out.append(m)
        b.unplay()
    return out


def is_lost_in_two(b: Board) -> bool:
    """True iff EVERY move of the player to move lets the opponent win at once.

    This is exact ground truth, not a heuristic: such a position is lost.
    """
    for m in b.legalMoves():
        b.play(m)
        lost = not b.terminal() and bool(immediate_wins(b))
        b.unplay()
        if not lost:
            return False
    return True


def make_sign_dataset(n_pos: int = 400, seed: int = 3):
    """Positions whose value is known EXACTLY, labelled through `to_pov`.

    win  : the player to move has an immediate win        -> z must be +1
    loss : every move of the player to move loses at once -> z must be -1

    Both labels are computed by `to_pov(final score, turn)`, so a sign bug
    anywhere in `to_pov` inverts them, the network dutifully learns the inverted
    mapping, and the assertions in the tests below fail.  That is the point.
    """
    rng = random.Random(seed)
    wins: list = []
    losses: list = []
    guard = 0
    while (len(wins) < n_pos or len(losses) < n_pos) and guard < 400_000:
        guard += 1
        b = Board()
        while not b.terminal():
            winning = immediate_wins(b)
            if winning:
                if len(wins) < n_pos:
                    w = winning[0]
                    p = b.copy()
                    p.play(w)
                    pi = np.zeros(COLS, dtype=np.float32)
                    pi[winning] = 1.0 / len(winning)
                    wins.append((encode(b), pi, to_pov(p.score(), b.turn), b.copy()))
                break
            if len(losses) < n_pos and is_lost_in_two(b):
                opp = RED if b.turn == YELLOW else YELLOW
                s = 1.0 if opp == YELLOW else 0.0
                pi = np.zeros(COLS, dtype=np.float32)
                pi[b.legalMoves()] = 1.0 / len(b.legalMoves())
                losses.append((encode(b), pi, to_pov(s, b.turn), b.copy()))
                break
            b.play(rng.choice(b.legalMoves()))
    return wins, losses


def test_sign_dataset_labels_are_verified_by_actually_playing_the_game_out():
    wins, losses = make_sign_dataset(150, seed=5)
    assert len(wins) >= 150 and len(losses) >= 150
    for _, pi, z, b in wins:
        assert z == 1.0, "a winning position was not labelled +1"
        mover = b.turn
        b2 = b.copy()
        b2.play(int(np.argmax(pi)))
        assert b2.winner == mover
        assert to_pov(b2.score(), mover) == 1.0
    rng = random.Random(0)
    for _, _, z, b in losses:
        assert z == -1.0, "a lost position was not labelled -1"
        mover = b.turn
        opp = RED if mover == YELLOW else YELLOW
        # whatever the player to move tries, the opponent wins immediately
        for m in b.legalMoves():
            b2 = b.copy()
            b2.play(m)
            assert not b2.terminal()
            w = immediate_wins(b2)
            assert w, "position was not actually lost"
            b2.play(w[0])
            assert b2.winner == opp
            assert to_pov(b2.score(), mover) == -1.0


def test_network_learns_the_right_value_sign():
    """The decisive anti-sign-bug test.

    Train on positions whose value is exact ground truth (+1 / -1), then check
    the network reproduces the SIGN, both on the positions it was trained on
    (memorisation: the labels really do say what we think they say) and on
    held-out ones (it is the concept, not the noise, that was learned).

    Thresholds are deliberately far from the observed values: with 800 training
    samples the held-out margin shrinks as the net overfits, so the test asserts
    a wide separation rather than a precise magnitude.
    """
    wins, losses = make_sign_dataset(500, seed=7)
    assert len(wins) >= 500 and len(losses) >= 500
    train_w, test_w = wins[:400], wins[400:]
    train_l, test_l = losses[:400], losses[400:]
    data = [(x, p, z) for x, p, z, _ in train_w + train_l]
    # a batch of empty / near-empty boards labelled by a real random-play
    # outcome, so that value(empty board) is pinned near 0
    # Board.playout() draws from the GLOBAL random module, so the global seed is
    # what makes these labels reproducible -- a local random.Random() would be
    # ignored and the test would depend on whatever ran before it.
    random.seed(9)
    for _ in range(200):
        b = Board()
        turn = b.turn
        p = b.copy()
        s = p.playout()
        pi = np.zeros(COLS, dtype=np.float32)
        pi[b.legalMoves()] = 1.0 / len(b.legalMoves())
        data.append((encode(b), pi, to_pov(s, turn)))
    X = torch.from_numpy(np.stack([d[0] for d in data]))
    P = torch.from_numpy(np.stack([d[1] for d in data]))
    Z = torch.tensor([d[2] for d in data], dtype=torch.float32)

    torch.manual_seed(1)
    net = Connect4Net(channels=32, blocks=2)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    for _ in range(1000):
        idx = torch.randint(0, len(X), (128,), generator=g)
        opt.zero_grad()
        logits, value = net(X[idx])
        loss, _, _ = alphazero_loss(logits, value, P[idx], Z[idx])
        loss.backward()
        opt.step()
    net.eval()

    def mean_value(samples):
        with torch.no_grad():
            x = torch.from_numpy(np.stack([s[0] for s in samples]))
            return net(x)[1].mean().item()

    tr_w, tr_l = mean_value(train_w), mean_value(train_l)
    te_w, te_l = mean_value(test_w), mean_value(test_l)
    with torch.no_grad():
        ve = net(torch.from_numpy(encode(Board())[None]))[1].item()

    # memorisation: the exact labels are reproduced, so the sign is right
    assert tr_w > 0.7, f"train value(win next move) = {tr_w:.3f} (SIGN BUG?)"
    assert tr_l < -0.7, f"train value(lost position) = {tr_l:.3f} (SIGN BUG?)"
    # generalisation: unseen positions land on the correct side, well separated
    assert te_w > 0.4, f"held-out value(win next move) = {te_w:.3f} (SIGN BUG?)"
    assert te_l < -0.4, f"held-out value(lost position) = {te_l:.3f} (SIGN BUG?)"
    assert te_w - te_l > 1.0, f"held-out separation only {te_w - te_l:.3f}"
    assert abs(ve) < 0.35, f"value(empty board) = {ve:.3f}, expected ~ 0"
