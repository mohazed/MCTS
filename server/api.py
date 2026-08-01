"""Minimal FastAPI backend: play Connect 4 against any of the agents.

The point of the front-end is NOT to be a web app, it is to make the course's
concepts visible: under each column it draws the network's PRIOR (grey) behind
the search's VISIT COUNT (blue), so one literally sees the search correcting the
network.

    uvicorn server.api:app --reload
"""

from __future__ import annotations

import os

# Must run before `torch` (imported transitively below) initializes its BLAS/
# OpenMP backends. On a host with a fractional CPU quota (e.g. Render's free
# tier, ~0.1-0.15 vCPU), torch's default multi-threaded intra-op parallelism
# causes threads to spawn and wait on each other for a share of that tiny
# quota, which is far slower than just running single-threaded -- turning a
# ~1ms forward pass into seconds. The tiny Connect-4 net gets nothing from
# parallelism anyway, so pin everything to 1 thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import uuid

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from game.connect4 import ALL_LINES, COLS, ROWS, Board
from search.base import build_agent

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

app = FastAPI(title="Mini-AlphaZero Connect 4")

STATIC = os.path.join(os.path.dirname(__file__), "static")
CKPT = os.environ.get("C4_CKPT", "ckpt/final.pt")

GAMES: dict[str, Board] = {}
_AGENTS: dict[tuple, object] = {}

AGENT_SPECS = {
    "puct": {"kind": "puct", "ckpt": CKPT, "label": "PUCT (réseau + recherche)"},
    "network": {"kind": "network", "ckpt": CKPT, "label": "Réseau seul (0 simulation)"},
    "uct": {"kind": "uct", "label": "UCT (playouts aléatoires)"},
    "grave": {"kind": "grave", "label": "GRAVE"},
    "flat": {"kind": "flat", "label": "Flat Monte Carlo"},
    "alphabeta-4": {"kind": "alphabeta", "depth": 4, "label": "Alpha-bêta profondeur 4"},
    "random": {"kind": "random", "label": "Aléatoire"},
}


def get_agent(name: str, sims: int):
    if name not in AGENT_SPECS:
        raise HTTPException(400, f"unknown agent {name}")
    key = (name, sims)
    if key not in _AGENTS:
        spec = dict(AGENT_SPECS[name])
        spec.pop("label", None)
        spec["sims"] = sims
        spec["playouts"] = sims
        if spec["kind"] in ("puct", "network") and not os.path.exists(CKPT):
            raise HTTPException(
                503, f"no checkpoint at {CKPT}; run the pipeline first"
            )
        _AGENTS[key] = build_agent(spec)
    return _AGENTS[key]


def winning_line(board: Board) -> list[list[int]] | None:
    """The four cells that won, in DISPLAY coordinates (row 0 = top)."""
    if board.winner is None:
        return None
    for line in ALL_LINES:
        if all(board.cells[i] == board.winner for i in line):
            return [[ROWS - 1 - i // COLS, i % COLS] for i in line]
    return None


def state(gid: str, board: Board, extra: dict | None = None) -> dict:
    out = {
        "game_id": gid,
        "grid": board.board[::-1].tolist(),  # row 0 = top, for display
        "turn": board.turn,
        "legal": board.legalMoves(),
        "terminal": board.terminal(),
        "winner": board.winner,
        "moves": board.moves,
        "last": board.last,
        "win_line": winning_line(board),
    }
    if extra:
        out.update(extra)
    return out


class MoveIn(BaseModel):
    game_id: str
    col: int


class AIIn(BaseModel):
    game_id: str
    agent: str = "puct"
    sims: int = 100


def get_game(gid: str) -> Board:
    if gid not in GAMES:
        raise HTTPException(404, "unknown game")
    return GAMES[gid]


@app.get("/api/agents")
def agents() -> dict:
    have_net = os.path.exists(CKPT)
    return {
        "agents": [
            {"id": k, "label": v["label"],
             "available": have_net or v["kind"] not in ("puct", "network")}
            for k, v in AGENT_SPECS.items()
        ],
        "checkpoint": CKPT,
        "checkpoint_found": have_net,
    }


@app.post("/api/new")
def new_game() -> dict:
    gid = uuid.uuid4().hex[:12]
    GAMES[gid] = Board()
    if len(GAMES) > 200:  # keep the process bounded
        for k in list(GAMES)[:100]:
            GAMES.pop(k, None)
    return state(gid, GAMES[gid])


@app.post("/api/move")
def move(inp: MoveIn) -> dict:
    b = get_game(inp.game_id)
    if b.terminal():
        raise HTTPException(400, "game over")
    if inp.col not in b.legalMoves():
        raise HTTPException(400, "illegal move")
    b.play(inp.col)
    return state(inp.game_id, b)


@app.post("/api/ai")
def ai_move(inp: AIIn) -> dict:
    b = get_game(inp.game_id)
    if b.terminal():
        raise HTTPException(400, "game over")
    agent = get_agent(inp.agent, max(1, min(2000, inp.sims)))
    col, info = agent.choose_move(b)
    visits = np.asarray(info["visits"], dtype=float)
    priors = info["priors"]
    b.play(col)
    return state(
        inp.game_id, b,
        {
            "col": int(col),
            "agent": agent.name,
            "visits": visits.tolist(),
            "priors": None if priors is None else np.asarray(priors, float).tolist(),
            "value": info["value"],
            "time_ms": info["time_ms"],
            "tt_hit_rate": (info["tt_hits"] / info["tt_lookups"]
                            if info["tt_lookups"] else None),
        },
    )


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


# the page loads the authors' photo from here
app.mount("/static", StaticFiles(directory=STATIC), name="static")
