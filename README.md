# Mini-AlphaZero — Puissance 4

Projet de cours **Monte Carlo Search** (Tristan Cazenave, Université Paris-Dauphine).

**Mohamed El Amine ROUIBI · Thomas SINAPI · Mohamed ZOUAD** — M2.

Un MCTS guidé par un réseau *policy + value* entraîné uniquement par self-play,
sans connaissance experte, sur le Puissance 4 (6 lignes × 7 colonnes) — la
transposition directe des slides « Alpha Zero Project » / « Projet Python » du
cours, qui décrivent le même pipeline en 5 étapes pour Breakthrough 5×5.

Le rapport est en français (`report/rapport.pdf`), le code et les commentaires
sont en anglais.

et un GUI est déployé sur **mini-alphazero-connect4.onrender.com**


---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ (testé avec 3.12.6), CPU uniquement. `pandoc` et `weasyprint` ne
servent qu'à régénérer le PDF du rapport (voir plus bas).

---

## Démarrage rapide

```bash
python -m pytest -q                                    # toute la suite de tests
python main.py pipeline --config configs/smoke.yaml    # pipeline complet, < 1 min
uvicorn server.api:app --port 8000                     # jouer dans le navigateur
```

---

## Interfaces gelées

Les trois signatures ci-dessous ont été figées **avant** d'écrire du code :
elles sont le contrat entre les trois membres du groupe.

### `Board` — `game/connect4.py`

```python
ROWS, COLS = 6, 7
EMPTY, YELLOW, RED = 0, 1, 2      # YELLOW joue en premier

class Board:
    cells:   list[int]    # 42 cases, index = row*7 + col, ligne 0 = BAS
    heights: list[int]    # 7 hauteurs de colonne
    turn:    int          # YELLOW ou RED
    moves:   int          # nombre de pions posés
    h:       int          # hash de Zobrist (trait inclus)
    last:    tuple | None # (row, col) du dernier coup
    winner:  int | None   # maintenu incrémentalement par play()

    @property
    def board(self) -> np.ndarray      # (6,7) int8, vue dérivée en LECTURE SEULE

    def legalMoves(self) -> list[int]  # colonnes jouables, ordre centre d'abord
    def play(self, col) -> None        # pose un pion, met à jour h par XOR
    def unplay(self) -> None           # défait le dernier coup (alpha-bêta)
    def terminal(self) -> bool
    def score(self) -> float           # 1.0 YELLOW gagne, 0.0 RED gagne, 0.5 nul
    def playout(self) -> float         # playout aléatoire EN PLACE, renvoie score()
    def playoutAMAF(self, played) -> float
    def copy(self) -> Board
    def mirror(self) -> Board          # symétrie gauche/droite
    def move_code(self, col) -> int    # code AMAF de la case remplie + joueur
```

**Convention de score** : `[0, 1]` du point de vue de YELLOW, exactement comme
le cours (`1.0 if White wins, 0.0 else, 0.5 draw`), pour que les codes UCT /
RAVE / GRAVE / PUCT du cours se transcrivent tels quels, y compris l'idiome
`if board.turn == RED: Q = 1 - Q`.

> **Écart assumé par rapport à la spécification initiale.** Celle-ci annotait
> `board` et `heights` en `np.ndarray`. Le stockage canonique est ici une liste
> Python plate, `board` étant une propriété dérivée qui rend bien le `(6,7) int8`
> spécifié. Raison : l'indexation scalaire numpy domine le coût des playouts, et
> UCT/GRAVE en font des dizaines de millions (E5 va jusqu'à 800 playouts).
> Mesure : **36 000 parties aléatoires/seconde**. Ce n'est pas un bitboard — la
> représentation reste d'un entier par case, comme dans les snippets du cours.

### `Agent` — `search/base.py`

```python
class Agent(Protocol):
    name: str
    def choose_move(self, board: Board) -> tuple[int, dict]: ...

# le dict :
# {"visits": (7,) float, "priors": (7,) float | None, "value": float | None,
#  "time_ms": float, "tt_hits": int, "tt_lookups": int}
```

Ce `dict` sert aux figures du rapport **et** à l'affichage du front (les barres
prior / visites). `visits` vaut 0 sur les colonnes illégales.

### Format d'un échantillon d'entraînement

```python
(planes, pi, z)
#  planes : np.ndarray (3, 6, 7) float32 — pions du joueur au trait,
#           pions de l'adversaire, plan constant à 1.0
#  pi     : np.ndarray (7,)     float32 — visites normalisées, somme = 1
#  z      : float32 ∈ {-1, 0, +1} — résultat, POINT DE VUE DU JOUEUR AU TRAIT
```

Le pont entre les deux conventions de signe est la fonction **unique**
`model.encode.to_pov(score, turn)` (`z = 2·score − 1` si le joueur au trait est
YELLOW, `1 − 2·score` sinon). C'est le bug n°1 de ce type de projet ; il est
testé explicitement (`tests/test_model.py`).

---

## Arborescence

```
game/connect4.py     moteur, Zobrist (85 nombres), symétrie, codes AMAF
search/tt.py         table de transposition (dict sur board.h)
search/flat.py       Flat Monte Carlo + UCB à la racine
search/uct.py        UCT récursif + BestMoveUCT
search/grave.py      playoutAMAF, updateAMAF, RAVE, GRAVE
search/puct.py       PUCT guidé par le réseau
search/base.py       interface Agent + fabrique d'agents
model/encode.py      encodage 3×6×7, to_pov, symétrie, softmax masqué
model/net.py         réseau résiduel 2 têtes (226 k paramètres)
train/               selfplay, buffer, train, pipeline
eval/                arena (ouvertures équilibrées, Wilson, Elo), baselines, testset
server/              FastAPI + page unique (prior gris / visites bleues)
scripts/             run_experiments.py, make_figures.py
tests/               test_game, test_search, test_model, test_pipeline
configs/             smoke.yaml, default.yaml, tuned.yaml, long.yaml
report/              rapport.tex, rapport.pdf, figures/, results/
```

---

## Reproduire tous les chiffres du rapport

Dans l'ordre. Les temps sont mesurés sur un Mac 8 cœurs, CPU seul.

```bash
# 0. tests (~1,5 min) — 136 tests
python -m pytest -q

# 1. jeu de test de finales : 200 positions à 20 pions résolues exactement
#    par alpha-bêta sans limite de profondeur (~2 min, mis en cache)
python main.py testset --n 200 --min-stones 20

# 2. run 1 — configuration initiale (~15 min, 6 workers)
python main.py pipeline --config configs/default.yaml \
       --log report/results/training_log_run1.jsonl

# 3. run 2 — configuration corrigée après diagnostic du run 1 (~20 min)
#    c'est ce run qui produit ckpt/final.pt
python main.py pipeline --config configs/tuned.yaml \
       --log report/results/training_log_run2.jsonl

# 4. expériences E1–E9, critères C1–C4, diagnostic -> report/results/*.json
#    (~10 min). --k 4 = 56 parties par confrontation ; c'est CETTE commande qui
#    a produit tous les JSON du rapport.
python scripts/run_experiments.py --workers 6 --k 4

# 5. figures + tableaux -> report/figures/*.png, report/results/tables.md
python scripts/make_figures.py

# 6. rapport PDF (LaTeX -> PDF)
python scripts/make_pdf.py
```

Le rapport est un document LaTeX (`report/rapport.tex`, XeLaTeX). Il faut un
moteur ; le plus simple à installer est `tectonic`, un binaire unique qui
télécharge lui-même les paquets dont il a besoin :

```bash
brew install tectonic        # macOS ; sinon : cargo install tectonic
python scripts/make_pdf.py
```

Une installation TeX Live / MacTeX fournissant `xelatex` fonctionne aussi ;
`scripts/make_pdf.py` essaie `tectonic`, puis `latexmk`, puis `xelatex`.

Commandes utiles :

```bash
python main.py eval --ckpt ckpt/final.pt --games 56 --workers 6  # C1–C4 seuls
python main.py play --agent puct --sims 200                      # jouer au terminal
python scripts/run_experiments.py --only E5 --workers 6          # une expérience
```

**Graines.** Toutes les expériences utilisent `seed = 20260731`. Les nombres de
Zobrist sont tirés d'une graine fixe (`game/connect4.py`) pour que les hash
soient identiques dans tous les processus de self-play.

---

## Interface web

```bash
uvicorn server.api:app --port 8000     # puis http://localhost:8000
```

Raccourcis clavier : **0**–**6** pour jouer une colonne, **espace** pour
lancer/mettre en pause, **N** pour une nouvelle partie.

Deux modes : **Humain contre IA**, et **IA contre IA** — on choisit alors un agent
et un budget de simulations pour chaque couleur, puis on lance la partie
(`▶ Lancer`), on la met en pause, ou on l'avance coup par coup (`⏭ Un coup`),
avec un délai réglable entre les coups. Attention : une partie ne mesure rien
(les agents sont souvent déterministes et le premier joueur est avantagé) ; pour
un chiffre, utiliser `python main.py eval`, qui joue des ouvertures équilibrées.

Sous chaque colonne, deux barres superposées : en **gris** le *prior* du réseau
`P(a|s)`, en **bleu** le nombre de **visites** de la recherche `N(s,a)`. On voit
donc directement la recherche corriger le réseau. Une jauge donne la valeur de
la position dans `[-1, +1]` du point de vue du joueur au trait. Le sélecteur
permet de comparer PUCT, le réseau seul (0 simulation), UCT, GRAVE, Flat MC,
alpha-bêta et l'aléatoire, avec un budget réglable de 25 à 800 simulations.

Le checkpoint chargé est `ckpt/final.pt` (variable d'environnement `C4_CKPT`
pour en choisir un autre).

---

## Répartition du travail

- **Mohamed El Amine ROUIBI** — `game/`, `search/` hors réseau (tt, flat, uct,
  grave). Expériences E1, E2, E5. Rapport §2–3.
- **Thomas SINAPI** — `model/`, `train/`, `search/puct.py`. Expériences E3, E4,
  E7. Rapport §4–5.
- **Mohamed ZOUAD** — `eval/` (arena, baselines, testset), `server/`,
  `scripts/`. Expériences E5, E6, E9. Figures. Rapport §6–7.
- **À trois** — les interfaces gelées ci-dessus, le diagnostic du run 1, et le
  rapport §1, §9 et §10.
