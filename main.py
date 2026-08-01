#!/usr/bin/env python3
"""Command line entry point.

    python main.py pipeline --config configs/default.yaml
    python main.py testset  --n 200 --min-stones 20
    python main.py eval     --ckpt ckpt/final.pt --games 56
    python main.py play     --agent puct --sims 100
"""

from __future__ import annotations

import argparse
import sys

import torch


def cmd_pipeline(args: argparse.Namespace) -> int:
    from train.pipeline import load_config, run_pipeline

    cfg = load_config(args.config)
    if args.workers is not None:
        cfg["selfplay"]["workers"] = args.workers
    if args.iterations is not None:
        cfg["selfplay"]["iterations"] = args.iterations
    if args.ckpt_dir is not None:
        cfg["ckpt_dir"] = args.ckpt_dir
    run_pipeline(cfg, log_path=args.log)
    return 0


def cmd_testset(args: argparse.Namespace) -> int:
    from eval.testset import build_testset

    pos = build_testset(n=args.n, min_stones=args.min_stones, seed=args.seed,
                        path=args.out)
    print(f"{len(pos)} solved positions -> {args.out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from eval.arena import play_match

    k = max(1, args.games // 14)
    puct = {"kind": "puct", "ckpt": args.ckpt, "sims": args.sims,
            "name": f"puct-{args.sims}"}
    net = {"kind": "network", "ckpt": args.ckpt, "name": "network-only"}
    matches = [
        ("C1 PUCT vs Random", puct, {"kind": "random"}),
        ("C2 PUCT vs UCT", puct, {"kind": "uct", "playouts": args.sims}),
        ("C3 PUCT vs alpha-beta d.4", puct, {"kind": "alphabeta", "depth": 4}),
        ("C4 network only vs Random", net, {"kind": "random"}),
    ]
    for label, a, b in matches:
        r = play_match(spec_a=a, spec_b=b, k=k, workers=args.workers,
                       seed=args.seed)
        print(f"{label:32s} {r.score:6.1%} [{r.ci_low:.1%}, {r.ci_high:.1%}] "
              f"{r.wins}W/{r.draws}D/{r.losses}L n={r.games}")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    """Play a game in the terminal against the chosen agent."""
    from game.connect4 import Board
    from search.base import build_agent

    spec = {"kind": args.agent, "sims": args.sims, "playouts": args.sims,
            "ckpt": args.ckpt, "depth": args.depth}
    agent = build_agent(spec)
    b = Board()
    while not b.terminal():
        print(b)
        if b.moves % 2 == 0:
            try:
                col = int(input(f"your move {b.legalMoves()}: "))
            except (ValueError, EOFError):
                return 1
            if col not in b.legalMoves():
                print("illegal")
                continue
        else:
            col, info = agent.choose_move(b)
            print(f"{agent.name} plays {col} "
                  f"(visits {info['visits'].astype(int).tolist()})")
        b.play(col)
    print(b)
    print("result:", b.score())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="main.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("pipeline", help="run the AlphaZero loop")
    q.add_argument("--config", required=True)
    q.add_argument("--log", default="report/results/training_log.jsonl")
    q.add_argument("--workers", type=int, default=None)
    q.add_argument("--iterations", type=int, default=None)
    q.add_argument("--ckpt-dir", default=None,
                   help="where to write checkpoints (default: the config's)")
    q.set_defaults(func=cmd_pipeline)

    q = sub.add_parser("testset", help="build the exactly-solved endgame test set")
    q.add_argument("--n", type=int, default=200)
    q.add_argument("--min-stones", type=int, default=20)
    q.add_argument("--seed", type=int, default=20260731)
    q.add_argument("--out", default="report/results/testset.json")
    q.set_defaults(func=cmd_testset)

    q = sub.add_parser("eval", help="measure C1-C4 for a checkpoint")
    q.add_argument("--ckpt", default="ckpt/final.pt")
    q.add_argument("--games", type=int, default=56)
    q.add_argument("--sims", type=int, default=100)
    q.add_argument("--workers", type=int, default=1)
    q.add_argument("--seed", type=int, default=20260731)
    q.set_defaults(func=cmd_eval)

    q = sub.add_parser("play", help="play in the terminal")
    q.add_argument("--agent", default="puct")
    q.add_argument("--sims", type=int, default=100)
    q.add_argument("--ckpt", default="ckpt/final.pt")
    q.add_argument("--depth", type=int, default=4)
    q.set_defaults(func=cmd_play)

    args = p.parse_args(argv)
    torch.set_num_threads(1)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
