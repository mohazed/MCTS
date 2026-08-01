"""Step 6/7 validation: self-play samples and the training loop."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from game.connect4 import COLS, RED, ROWS, YELLOW, Board, from_moves
from model.encode import PLANES, encode, mirror_pi, mirror_planes, to_pov
from model.net import new_net, save_net
from search.puct import BestMovePUCT, add_dirichlet_noise
from train.buffer import ReplayBuffer
from train.pipeline import load_config, run_pipeline
from train.selfplay import SelfPlayConfig, generate, play_game
from train.train import train_steps

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def small_net():
    return new_net(channels=16, blocks=1)


# --------------------------------------------------------------------------
# sample format
# --------------------------------------------------------------------------
def test_selfplay_samples_have_the_frozen_format(small_net):
    cfg = SelfPlayConfig(sims=10, temperature_moves=4, augment_symmetry=False)
    rng = np.random.default_rng(0)
    samples = play_game(small_net, cfg, rng)
    assert samples
    for planes, pi, z in samples:
        assert planes.shape == (PLANES, ROWS, COLS) and planes.dtype == np.float32
        assert pi.shape == (COLS,) and pi.dtype == np.float32
        assert pi.sum() == pytest.approx(1.0, abs=1e-5), "pi must sum to 1"
        assert np.all(pi >= 0.0)
        assert float(z) in (-1.0, 0.0, 1.0), f"z must be in {{-1,0,1}}, got {z}"


def test_pi_puts_no_mass_on_illegal_moves(small_net):
    """A full column can never be visited, so pi must be 0 there."""
    cfg = SelfPlayConfig(sims=15, temperature_moves=0, augment_symmetry=False)
    rng = np.random.default_rng(3)
    for _ in range(3):
        b = Board()
        while not b.terminal():
            _, visits, _, _, _ = BestMovePUCT(b, small_net, 15, training=True, rng=rng)
            legal = b.legalMoves()
            for c in range(COLS):
                if c not in legal:
                    assert visits[c] == 0.0
            b.play(int(np.argmax(visits)))


def test_z_matches_the_result_of_the_game_for_each_player(small_net):
    """Every state of the winner is +1, every state of the loser is -1."""
    cfg = SelfPlayConfig(sims=8, temperature_moves=2, augment_symmetry=False)
    rng = np.random.default_rng(1)
    for _ in range(5):
        samples = play_game(small_net, cfg, rng)
        zs = {float(z) for _, _, z in samples}
        # states alternate colours, so a decisive game has both +1 and -1
        assert zs <= {-1.0, 0.0, 1.0}
        if zs != {0.0}:
            assert zs == {-1.0, 1.0}
        # consecutive states belong to alternating players
        for i in range(len(samples) - 1):
            assert float(samples[i][2]) == -float(samples[i + 1][2]) or zs == {0.0}


def test_symmetry_augmentation_doubles_and_mirrors_both_planes_and_pi(small_net):
    cfg = SelfPlayConfig(sims=8, temperature_moves=0, augment_symmetry=True)
    rng = np.random.default_rng(2)
    samples = play_game(small_net, cfg, rng)
    assert len(samples) % 2 == 0
    for i in range(0, len(samples), 2):
        (p0, pi0, z0), (p1, pi1, z1) = samples[i], samples[i + 1]
        np.testing.assert_array_equal(p1, mirror_planes(p0))
        np.testing.assert_array_equal(pi1, mirror_pi(pi0))
        assert z0 == z1, "mirroring must not change the outcome"
    cfg_off = SelfPlayConfig(sims=8, temperature_moves=0, augment_symmetry=False)
    n_off = len(play_game(small_net, cfg_off, np.random.default_rng(2)))
    assert len(samples) == 2 * n_off


# --------------------------------------------------------------------------
# Dirichlet noise
# --------------------------------------------------------------------------
def test_dirichlet_noise_is_off_when_training_is_false(small_net):
    """The arena must be deterministic: no noise outside self-play."""
    b = from_moves([3, 3])
    runs = [BestMovePUCT(b, small_net, 40, training=False)[1] for _ in range(6)]
    for v in runs[1:]:
        np.testing.assert_array_equal(v, runs[0]), "training=False must be deterministic"
    # with noise on, the visit distribution does move
    rng = np.random.default_rng(0)
    noisy = [
        BestMovePUCT(b, small_net, 40, training=True, rng=rng)[1] for _ in range(6)
    ]
    assert any(not np.array_equal(v, noisy[0]) for v in noisy[1:])


def test_dirichlet_noise_keeps_priors_legal_and_normalised():
    rng = np.random.default_rng(0)
    b = from_moves([3] * ROWS)  # column 3 full
    legal = b.legalMoves()
    prior = np.zeros(COLS)
    prior[legal] = 1.0 / len(legal)
    for _ in range(50):
        p = add_dirichlet_noise(prior, legal, 1.0, 0.25, rng)
        assert p[3] == 0.0, "noise must not resurrect an illegal move"
        assert p.sum() == pytest.approx(1.0)
        assert np.all(p >= 0)


def test_dirichlet_mixing_weight_is_respected():
    rng = np.random.default_rng(7)
    legal = list(range(COLS))
    prior = np.zeros(COLS)
    prior[0] = 1.0  # all the mass on column 0
    ps = np.stack([add_dirichlet_noise(prior, legal, 1.0, 0.25, rng) for _ in range(4000)])
    assert ps[:, 0].mean() == pytest.approx(0.75 + 0.25 / COLS, abs=0.02)
    assert ps[:, 1:].sum(axis=1).mean() == pytest.approx(0.25 * 6 / COLS, abs=0.02)


# --------------------------------------------------------------------------
# buffer and training
# --------------------------------------------------------------------------
def test_buffer_keeps_only_the_last_iterations():
    buf = ReplayBuffer(max_iters=3, max_samples=10_000)
    for i in range(5):
        buf.add_iteration([(np.zeros((3, 6, 7), np.float32),
                            np.full(7, 1 / 7, np.float32), float(i))] * 4)
    assert len(buf.buckets) == 3
    assert buf.n_samples == 12
    _, _, Z = buf.tensors()
    assert set(Z.tolist()) == {2.0, 3.0, 4.0}


def test_buffer_respects_the_sample_cap():
    buf = ReplayBuffer(max_iters=8, max_samples=25)
    for _ in range(8):
        buf.add_iteration([(np.zeros((3, 6, 7), np.float32),
                            np.full(7, 1 / 7, np.float32), 0.0)] * 10)
    assert buf.n_samples <= 30
    X, _, _ = buf.tensors()
    assert len(X) <= 25


def test_training_reduces_the_loss(small_net):
    buf = ReplayBuffer(max_iters=2, max_samples=1000)
    cfg = SelfPlayConfig(sims=8, temperature_moves=4)
    rng = np.random.default_rng(5)
    samples = []
    for _ in range(6):
        samples += play_game(small_net, cfg, rng)
    buf.add_iteration(samples)
    net = new_net(channels=16, blocks=1)
    first = train_steps(net, buf, steps=5, batch_size=32, lr=1e-3)
    later = train_steps(net, buf, steps=200, batch_size=32, lr=1e-3)
    assert later["loss_total"] < first["loss_total"]


# --------------------------------------------------------------------------
# one full smoke iteration
# --------------------------------------------------------------------------
def test_one_full_smoke_iteration(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    cfg["selfplay"]["iterations"] = 1
    cfg["selfplay"]["workers"] = 1
    cfg["selfplay"]["games_per_iter"] = 2
    cfg["train"]["steps_per_iter"] = 5
    cfg["eval"] = {"every": 0, "games": 14, "final_games": 14, "sims": 5,
                   "use_testset": False}
    # never write into the real ckpt/ directory: a test run must not destroy
    # the checkpoints of a training run
    cfg["ckpt_dir"] = str(tmp_path / "ckpt")
    log = str(tmp_path / "log.jsonl")
    out = run_pipeline(cfg, log_path=log, quiet=True)
    assert os.path.exists(out["final"])
    assert str(tmp_path) in out["final"], "the test must not write to ckpt/"
    with open(log) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 1
    rec = records[0]
    for key in ("iter", "loss_total", "loss_policy", "loss_value", "n_samples",
                "selfplay_seconds", "train_seconds", "prior_center", "value_empty"):
        assert key in rec, key
    assert rec["n_samples"] > 0
    assert np.isfinite(rec["loss_total"])


def test_generate_is_reproducible_for_a_given_seed(tmp_path, small_net):
    path = str(tmp_path / "n.pt")
    save_net(small_net, path)
    cfg = SelfPlayConfig(sims=6, temperature_moves=3)
    a = generate(path, cfg, n_games=2, workers=1, seed=42)
    b = generate(path, cfg, n_games=2, workers=1, seed=42)
    assert len(a) == len(b)
    for (x1, p1, z1), (x2, p2, z2) in zip(a, b):
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(p1, p2)
        assert z1 == z2


def test_pipeline_never_writes_outside_its_ckpt_dir(tmp_path):
    """Regression: running smoke.yaml used to overwrite a real run's ckpt/."""
    cfg = load_config("configs/smoke.yaml")
    assert cfg.get("ckpt_dir") == "ckpt/smoke", "smoke.yaml must use its own dir"
    cfg["selfplay"].update(iterations=1, workers=1, games_per_iter=1)
    cfg["train"]["steps_per_iter"] = 2
    cfg["eval"] = {"every": 0, "use_testset": False}
    cfg["ckpt_dir"] = str(tmp_path / "cp")
    before = sorted(os.listdir("ckpt")) if os.path.isdir("ckpt") else []
    run_pipeline(cfg, log_path=str(tmp_path / "l.jsonl"), quiet=True)
    after = sorted(os.listdir("ckpt")) if os.path.isdir("ckpt") else []
    assert before == after, "the pipeline wrote into ckpt/ despite ckpt_dir"
    assert os.path.exists(tmp_path / "cp" / "final.pt")
