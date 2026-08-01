"""The AlphaZero loop: self-play -> training -> evaluation -> repeat.

This is the five-step recipe of the course's "Alpha Zero Project" slide, applied
to Connect 4 instead of Breakthrough 5x5:

  1) define the network            -> model/net.py
  2) implement PUCT with it        -> search/puct.py
  3) let it play against itself    -> train/selfplay.py
  4) record the Monte-Carlo
     distributions and the results -> the (planes, pi, z) samples
  5) train, then iterate           -> here

No gating arena: as in AlphaZero (and unlike AlphaGo Zero) the newest network is
always kept.  It is simpler and cheaper, and with 12 iterations the extra
evaluation games would cost more than they are worth.
"""

from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch
import yaml

from eval.arena import play_match
from eval.testset import load_testset, mean_value, network_agreement, winning_positions
from game.connect4 import Board
from model.encode import evaluate
from model.net import load_net, new_net, save_net
from train.buffer import ReplayBuffer
from train.selfplay import SelfPlayConfig, generate, make_pool
from train.train import train_steps

CKPT_DIR = "ckpt"
LOG_PATH = "report/results/training_log.jsonl"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _eval_specs(ckpt: str, cfg: dict) -> list[tuple[str, dict, dict]]:
    """(label, spec A, spec B) for the periodic evaluations = criteria C1-C4."""
    sims = cfg["eval"].get("sims", 100)
    uct_c = cfg.get("search", {}).get("uct_c", 0.4)
    puct = {"kind": "puct", "ckpt": ckpt, "sims": sims,
            "c_puct": cfg.get("search", {}).get("c_puct", 1.0),
            "name": f"puct-{sims}"}
    return [
        ("C1_puct_vs_random", puct, {"kind": "random"}),
        ("C2_puct_vs_uct", puct, {"kind": "uct", "playouts": sims, "c": uct_c}),
        ("C3_puct_vs_alphabeta4", puct, {"kind": "alphabeta", "depth": 4}),
        ("C4_network_vs_random",
         {"kind": "network", "ckpt": ckpt, "name": "network-only"},
         {"kind": "random"}),
    ]


def run_pipeline(cfg: dict, log_path: str = LOG_PATH, quiet: bool = False) -> dict:
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    sp = cfg["selfplay"]
    tr = cfg["train"]
    ev = cfg.get("eval", {})
    iterations = int(sp["iterations"])
    workers = int(sp.get("workers", 1))
    eval_every = int(ev.get("every", 0))
    eval_k = max(1, int(ev.get("games", 28)) // 14)
    final_k = max(1, int(ev.get("final_games", 56)) // 14)

    # every config writes to its own directory: running smoke.yaml must never
    # overwrite the checkpoints of a real run
    ckpt_dir = cfg.get("ckpt_dir", CKPT_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    net = new_net(channels=cfg["net"]["channels"], blocks=cfg["net"]["blocks"])
    spcfg = SelfPlayConfig.from_cfg(cfg)
    buffer = ReplayBuffer(
        max_iters=int(tr.get("buffer_iters", 8)),
        max_samples=int(tr.get("buffer_max", 60_000)),
    )

    # fixed diagnostic positions (E7 and the value-sign sanity check)
    win_positions = winning_positions(50, seed=1234)
    testset_path = ev.get("testset_path", "report/results/testset.json")
    testset = []
    if ev.get("use_testset", True) and os.path.exists(testset_path):
        testset = load_testset(testset_path)

    pool = make_pool(workers) if workers > 1 else None
    log_file = open(log_path, "w")
    history: list[dict] = []
    gen = torch.Generator().manual_seed(seed)
    t_start = time.time()

    try:
        for it in range(1, iterations + 1):
            cur = os.path.join(ckpt_dir, "current.pt")
            save_net(net, cur, iteration=it - 1)

            t0 = time.time()
            samples = generate(
                cur, spcfg, int(sp["games_per_iter"]),
                workers=workers, seed=seed + 1_000_003 * it, pool=pool,
            )
            selfplay_s = time.time() - t0
            buffer.add_iteration(samples)

            lr = float(tr["lr"])
            if it >= int(tr.get("lr_drop_iter", 10**9)):
                lr = float(tr.get("lr_after", lr))
            t0 = time.time()
            stats = train_steps(
                net, buffer,
                steps=int(tr["steps_per_iter"]),
                batch_size=int(tr["batch_size"]),
                lr=lr,
                weight_decay=float(tr.get("weight_decay", 1e-4)),
                generator=gen,
            )
            train_s = time.time() - t0

            ckpt = os.path.join(ckpt_dir, f"iter_{it:02d}.pt")
            save_net(net, ckpt, iteration=it)

            prior_empty, value_empty = evaluate(net, Board())
            rec = {
                "iter": it,
                "loss_total": stats["loss_total"],
                "loss_policy": stats["loss_policy"],
                "loss_value": stats["loss_value"],
                "n_samples": len(samples),
                "n_buffer": buffer.n_samples,
                "selfplay_seconds": selfplay_s,
                "train_seconds": train_s,
                "lr": lr,
                # E7: the policy's probability on the central column, empty board
                "prior_center": float(prior_empty[3]),
                "prior_empty": [float(x) for x in prior_empty],
                # sanity: value head signs
                "value_empty": float(value_empty),
                "value_win_next": mean_value(net, win_positions),
                # E6: agreement with perfect play, policy head alone
                "net_agreement": network_agreement(net, testset) if testset else None,
            }

            if eval_every and (it % eval_every == 0 or it == iterations):
                last = it == iterations
                k = final_k if last else eval_k
                for label, spec_a, spec_b in _eval_specs(ckpt, cfg):
                    r = play_match(
                        spec_a=spec_a, spec_b=spec_b, k=k,
                        seed=seed + 31 * it, workers=workers, pool=pool,
                    )
                    rec[label] = r.to_dict()

            history.append(rec)
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            if not quiet:
                msg = (
                    f"iter {it:2d}/{iterations}  loss {rec['loss_total']:.4f} "
                    f"(p {rec['loss_policy']:.4f}, v {rec['loss_value']:.4f})  "
                    f"samples {rec['n_samples']:5d}  buffer {rec['n_buffer']:6d}  "
                    f"sp {selfplay_s:5.1f}s  tr {train_s:5.1f}s  "
                    f"P(center) {rec['prior_center']:.3f}  "
                    f"v(empty) {rec['value_empty']:+.3f}  "
                    f"v(win) {rec['value_win_next']:+.3f}"
                )
                for label, _, _ in _eval_specs("", cfg):
                    if label in rec:
                        msg += f"\n              {label}: {rec[label]['score']:.1%}"
                print(msg, flush=True)
    finally:
        log_file.close()
        if pool is not None:
            pool.close()
            pool.join()

    save_net(net, os.path.join(ckpt_dir, "final.pt"), iteration=iterations)
    final = os.path.join(ckpt_dir, "final.pt")
    if not quiet:
        print(f"done in {time.time() - t_start:.0f}s -> {final}", flush=True)
    return {"history": history, "final": final}
