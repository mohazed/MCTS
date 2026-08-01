"""The `Agent` interface, frozen before any code was written (see README.md).

Every search algorithm and every baseline exposes the same two things:

    name: str
    choose_move(board) -> (column, info)

`info` is a plain dict:

    {"visits": (7,) float array,      # search visit counts, 0 for illegal moves
     "priors": (7,) float array|None, # network policy, None if there is no net
     "value":  float|None,            # network value, POV of the player to move
     "time_ms": float,
     "tt_hits": int, "tt_lookups": int}

It feeds both the report figures and the web front-end, which draws the priors
behind the visits.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from game.connect4 import COLS, Board


class Agent(Protocol):
    name: str

    def choose_move(self, board: Board) -> tuple[int, dict]: ...


def empty_info(**kwargs: Any) -> dict:
    info: dict = {
        "visits": np.zeros(COLS, dtype=np.float64),
        "priors": None,
        "value": None,
        "time_ms": 0.0,
        "tt_hits": 0,
        "tt_lookups": 0,
    }
    info.update(kwargs)
    return info


def build_agent(spec: dict) -> Agent:
    """Build an agent from a picklable description.

    Used by the arena (multiprocessing workers rebuild their own agents rather
    than receiving a pickled torch model) and by the web server.

    spec = {"kind": "puct"|"uct"|"flat"|"ucb"|"grave"|"rave"|"random"
                    |"alphabeta"|"network", ...algorithm parameters}
    """
    kind = spec["kind"]
    if kind == "random":
        from eval.baselines import RandomPlayer

        return RandomPlayer(seed=spec.get("seed"))
    if kind == "alphabeta":
        from eval.baselines import AlphaBetaPlayer

        return AlphaBetaPlayer(depth=spec.get("depth", 4))
    if kind == "flat":
        from search.flat import FlatPlayer

        return FlatPlayer(playouts=spec.get("playouts", 200))
    if kind == "ucb":
        from search.flat import UCBPlayer

        return UCBPlayer(playouts=spec.get("playouts", 200), c=spec.get("c", 0.4))
    if kind == "uct":
        from search.uct import UCTPlayer

        return UCTPlayer(playouts=spec.get("playouts", 200), c=spec.get("c", 0.4))
    if kind in ("rave", "grave"):
        from search.grave import GRAVEPlayer

        ref = 0 if kind == "rave" else spec.get("ref", 50)
        return GRAVEPlayer(
            playouts=spec.get("playouts", 200),
            ref=ref,
            bias=spec.get("bias", 1e-5),
            name=spec.get("name"),
        )
    if kind in ("puct", "network"):
        from model.net import load_net

        net = load_net(spec["ckpt"])
        if kind == "network":
            from eval.baselines import NetworkOnlyPlayer

            return NetworkOnlyPlayer(net, name=spec.get("name"))
        from search.puct import PUCTPlayer

        return PUCTPlayer(
            net,
            sims=spec.get("sims", 100),
            c_puct=spec.get("c_puct", 1.0),
            training=False,
            name=spec.get("name"),
        )
    raise ValueError(f"unknown agent kind {kind!r}")
