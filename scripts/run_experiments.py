#!/usr/bin/env python3
"""Run experiments E1-E9 and write one JSON per experiment.

    python scripts/run_experiments.py --workers 6
    python scripts/run_experiments.py --only E1 E5

NO NUMBER IN THE REPORT IS WRITTEN BY HAND: everything comes from the JSON files
this script produces in report/results/, which scripts/make_figures.py then
turns into the figures and tables.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.arena import play_match  # noqa: E402
from eval.testset import load_testset  # noqa: E402
from game.connect4 import Board, from_moves  # noqa: E402
from model.encode import evaluate_batch  # noqa: E402
from model.net import load_net, new_net, save_net  # noqa: E402
from search.puct import PUCTPlayer  # noqa: E402
from train.selfplay import make_pool  # noqa: E402

RESULTS = "report/results"
RUN1_LOG = f"{RESULTS}/training_log_run1.jsonl"
RUN2_LOG = f"{RESULTS}/training_log_run2.jsonl"
FINAL = "ckpt/final.pt"


def save(name: str, data: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(f"{RESULTS}/{name}.json", "w") as f:
        json.dump(data, f, indent=1)
    print(f"  -> {RESULTS}/{name}.json", flush=True)


def read_log(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def match(pool, a, b, k, seed=20260731, depth=1):
    r = play_match(spec_a=a, spec_b=b, k=k, seed=seed, pool=pool,
                   openings_depth=depth)
    print(f"    {r}", flush=True)
    return r.to_dict()


# Both PUCT (training=False) and alpha-beta are DETERMINISTIC, so repeating an
# opening replays the identical game.  Those match-ups must widen the opening
# book (49 two-move openings -> 98 distinct games) instead of repeating it,
# otherwise the Wilson interval is computed on duplicates and is far too narrow.
DETERMINISTIC_DEPTH = 2


# --------------------------------------------------------------------------
# E1 -- the course's ladder: Random -> Flat MC -> UCB -> UCT, all at 200 playouts
# --------------------------------------------------------------------------
def E1(pool, k=2):
    P = 200
    agents = {
        "random": {"kind": "random", "name": "Aléatoire"},
        "flat": {"kind": "flat", "playouts": P, "name": f"Flat MC ({P})"},
        "ucb": {"kind": "ucb", "playouts": P, "name": f"UCB racine ({P})"},
        "uct": {"kind": "uct", "playouts": P, "name": f"UCT ({P})"},
    }
    order = ["random", "flat", "ucb", "uct"]
    out = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            out.append(match(pool, agents[a], agents[b], k))
    save("E1", {"playouts": P, "order": order,
                "labels": {k2: v["name"] for k2, v in agents.items()},
                "matches": out})


# --------------------------------------------------------------------------
# E2 -- UCT exploration constant against alpha-beta depth 4
# --------------------------------------------------------------------------
def E2(pool, k=2):
    cs = [0.2, 0.4, 0.7, 1.0]
    ab = {"kind": "alphabeta", "depth": 4}
    res = []
    for c in cs:
        res.append(match(pool, {"kind": "uct", "playouts": 200, "c": c,
                                "name": f"UCT c={c}"}, ab, k))
    save("E2", {"c_values": cs, "playouts": 200, "opponent": "alpha-bêta d.4",
                "results": res})


# --------------------------------------------------------------------------
# E3 / E4 / E6 / E7 -- read straight from the training logs (no extra compute)
# --------------------------------------------------------------------------
def logs_experiments():
    runs = {}
    for tag, path in (("run1", RUN1_LOG), ("run2", RUN2_LOG)):
        recs = read_log(path)
        if recs:
            runs[tag] = recs
    if not runs:
        print("  no training log found, skipping E3/E4/E6/E7")
        return

    e3 = {tag: {"iter": [r["iter"] for r in rs],
                "loss_total": [r["loss_total"] for r in rs],
                "loss_policy": [r["loss_policy"] for r in rs],
                "loss_value": [r["loss_value"] for r in rs]}
          for tag, rs in runs.items()}
    save("E3", e3)

    keys = ["C1_puct_vs_random", "C2_puct_vs_uct",
            "C3_puct_vs_alphabeta4", "C4_network_vs_random"]
    e4 = {}
    for tag, rs in runs.items():
        d = {"iter": [], **{k: [] for k in keys},
             **{k + "_lo": [] for k in keys}, **{k + "_hi": [] for k in keys}}
        for r in rs:
            if keys[0] not in r:
                continue
            d["iter"].append(r["iter"])
            for k in keys:
                d[k].append(r[k]["score"])
                d[k + "_lo"].append(r[k]["ci_low"])
                d[k + "_hi"].append(r[k]["ci_high"])
        d["games"] = [r[keys[0]]["games"] for r in rs if keys[0] in r]
        e4[tag] = d
    save("E4", {"criteria": keys, "runs": e4})

    e6 = {tag: {"iter": [r["iter"] for r in rs],
                "net_agreement": [r.get("net_agreement") for r in rs]}
          for tag, rs in runs.items()}
    save("E6", e6)

    e7 = {tag: {"iter": [r["iter"] for r in rs],
                "prior_center": [r["prior_center"] for r in rs],
                "value_empty": [r["value_empty"] for r in rs],
                "prior_empty": [r["prior_empty"] for r in rs]}
          for tag, rs in runs.items()}
    save("E7", e7)


# --------------------------------------------------------------------------
# E5 -- strength as a function of the simulation budget (the key figure)
# --------------------------------------------------------------------------
def E5(pool, k=2, budgets=(25, 50, 100, 200, 400, 800)):
    """98 distinct games per point: PUCT vs alpha-beta is a deterministic pair."""
    ab = {"kind": "alphabeta", "depth": 4}
    puct, uct = [], []
    for n in budgets:
        print(f"    budget {n}", flush=True)
        puct.append(match(pool, {"kind": "puct", "ckpt": FINAL, "sims": n,
                                 "name": f"PUCT {n}"}, ab, 1,
                          depth=DETERMINISTIC_DEPTH))
        uct.append(match(pool, {"kind": "uct", "playouts": n,
                                "name": f"UCT {n}"}, ab, 1,
                         depth=DETERMINISTIC_DEPTH))
    save("E5", {"budgets": list(budgets), "opponent": "alpha-bêta d.4",
                "openings_depth": DETERMINISTIC_DEPTH, "puct": puct, "uct": uct})


def CFINAL(pool, k=2):
    """C1-C4 on the final checkpoint, with enough DISTINCT games to be honest."""
    sims = 100
    puct = {"kind": "puct", "ckpt": FINAL, "sims": sims, "name": f"PUCT {sims}"}
    net = {"kind": "network", "ckpt": FINAL, "name": "Réseau seul"}
    rows = {}
    # PUCT vs random / UCT: the opponent is stochastic, repeats give new games
    rows["C1_puct_vs_random"] = match(pool, puct, {"kind": "random"}, 4)
    rows["C2_puct_vs_uct"] = match(
        pool, puct, {"kind": "uct", "playouts": sims, "name": f"UCT {sims}"}, 4)
    # PUCT vs alpha-beta: BOTH deterministic -> widen the opening book
    rows["C3_puct_vs_alphabeta4"] = match(
        pool, puct, {"kind": "alphabeta", "depth": 4}, 1,
        depth=DETERMINISTIC_DEPTH)
    rows["C4_network_vs_random"] = match(pool, net, {"kind": "random"}, 4)
    save("C", {"sims": sims, "checkpoint": FINAL, "results": rows,
               "thresholds": {"C1_puct_vs_random": 0.98, "C2_puct_vs_uct": 0.70,
                              "C3_puct_vs_alphabeta4": 0.60,
                              "C4_network_vs_random": 0.85}})


# --------------------------------------------------------------------------
# E8 -- UCT vs RAVE vs GRAVE
# --------------------------------------------------------------------------
def E8(pool, k=2, budgets=(200, 1000)):
    out = {}
    for n in budgets:
        uct = {"kind": "uct", "playouts": n, "name": f"UCT {n}"}
        rave = {"kind": "rave", "playouts": n, "name": f"RAVE {n}"}
        grave = {"kind": "grave", "playouts": n, "ref": 50, "name": f"GRAVE {n}"}
        print(f"    budget {n}", flush=True)
        out[str(n)] = [
            match(pool, grave, uct, k),
            match(pool, rave, uct, k),
            match(pool, grave, rave, k),
        ]
    save("E8", {"budgets": list(budgets), "grave_ref": 50, "results": out})


# --------------------------------------------------------------------------
# E9 -- transposition table hit rate and time per move
# --------------------------------------------------------------------------
def E9(pool, k=1):
    rnd = {"kind": "random"}
    specs = [
        {"kind": "flat", "playouts": 200, "name": "Flat MC 200"},
        {"kind": "ucb", "playouts": 200, "name": "UCB racine 200"},
        {"kind": "uct", "playouts": 200, "name": "UCT 200"},
        {"kind": "rave", "playouts": 200, "name": "RAVE 200"},
        {"kind": "grave", "playouts": 200, "ref": 50, "name": "GRAVE 200"},
        {"kind": "alphabeta", "depth": 4, "name": "Alpha-bêta d.4"},
        {"kind": "puct", "ckpt": FINAL, "sims": 100, "name": "PUCT 100"},
        {"kind": "network", "ckpt": FINAL, "name": "Réseau seul"},
    ]
    rows = []
    for s in specs:
        r = play_match(spec_a=s, spec_b=rnd, k=k, seed=20260731, pool=pool)
        rows.append({"name": s["name"], "ms_per_move": r.ms_per_move_a,
                     "tt_hit_rate": r.tt_hit_rate_a, "games": r.games})
        print(f"    {s['name']:16s} {r.ms_per_move_a:7.2f} ms/coup   "
              f"TT {r.tt_hit_rate_a:5.1%}", flush=True)
    save("E9", {"rows": rows,
                "note": "mesuré sur des parties à ouvertures équilibrées "
                        "contre l'agent aléatoire"})


# --------------------------------------------------------------------------
# diagnostic -- evidence for the report's discussion of the failed run
# --------------------------------------------------------------------------
def DIAG(pool):
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "tests"))
    from tests.test_search import block_positions, win_positions

    w30 = win_positions(30, seed=1)
    b30 = block_positions(30, seed=2)
    ts = load_testset()
    tsb = [from_moves(p["moves"]) for p in ts]
    rng = random.Random(0)
    pos = []
    while len(pos) < 300:
        b = Board()
        for _ in range(rng.randrange(0, 30)):
            if b.terminal():
                break
            b.play(rng.choice(b.legalMoves()))
        if not b.terminal():
            pos.append(b)

    out = {}
    for tag, ckdir in (("run1", "ckpt/run1"), ("run2", "ckpt")):
        rows = []
        it = 0
        while True:
            path = "UNTRAINED" if it == 0 else f"{ckdir}/iter_{it:02d}.pt"
            if it > 0 and not os.path.exists(path):
                break
            if it == 0:
                torch.manual_seed(0)  # the untrained reference must be reproducible
                net = new_net(64, 3)
            else:
                net = load_net(path)
            _, v = evaluate_batch(net, pos)
            pr = evaluate_batch(net, tsb)[0]
            agr = float(np.mean([int(int(np.argmax(pr[i])) in p["optimal"])
                                 for i, p in enumerate(ts)]))
            a = PUCTPlayer(net, sims=100)
            random.seed(0)
            win = sum(a.choose_move(b)[0] in w for b, w in w30)
            random.seed(0)
            blk = sum(a.choose_move(b)[0] == c for b, c in b30)
            rows.append({"iter": it,
                         "value_saturated_frac": float(np.mean(np.abs(v) > 0.999)),
                         "value_std": float(v.std()),
                         "value_mean": float(v.mean()),
                         "puct_takes_win": win, "puct_blocks": blk,
                         "n_tactical": 30, "policy_agreement": agr})
            print(f"    {tag} it{it:02d}  sat {rows[-1]['value_saturated_frac']:5.1%} "
                  f"win {win}/30 blk {blk}/30 agr {agr:.1%}", flush=True)
            it += 1
            if it > 40:
                break
        out[tag] = rows
    save("diagnostic", {"runs": out,
                        "note": "iter 0 = réseau non entraîné (référence)"})


ALL = {"E1": E1, "E2": E2, "E5": E5, "E8": E8, "E9": E9,
       "CFINAL": CFINAL, "DIAG": DIAG}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--k", type=int, default=2, help="repeats: 14k games per match")
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()
    torch.set_num_threads(1)
    os.makedirs(RESULTS, exist_ok=True)

    todo = args.only or (["logs"] + list(ALL))
    pool = make_pool(args.workers) if args.workers > 1 else None
    t0 = time.time()
    try:
        if "logs" in todo or args.only is None:
            print("[logs] E3 / E4 / E6 / E7 from the training logs", flush=True)
            logs_experiments()
        for name in todo:
            if name in ALL:
                print(f"[{name}]", flush=True)
                if name == "DIAG":
                    ALL[name](pool)
                else:
                    ALL[name](pool, args.k)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(f"all experiments done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
