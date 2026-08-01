#!/usr/bin/env python3
"""Build every figure and every table of the report from report/results/*.json.

    python scripts/make_figures.py

Reads ONLY the JSON files produced by scripts/run_experiments.py, writes PNGs
into report/figures/ and the generated markdown tables into
report/results/tables.md.  No number is typed by hand anywhere.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = "report/results"
FIGURES = "report/figures"


def load(name: str):
    path = f"{RESULTS}/{name}.json"
    if not os.path.exists(path):
        print(f"  (missing {path}, skipped)")
        return None
    with open(path) as f:
        return json.load(f)


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def ci(m: dict) -> str:
    return f"{pct(m['score'])} [{pct(m['ci_low'])}, {pct(m['ci_high'])}]"


def errbars(scores, los, his):
    # clamp at 0: the Wilson interval is not centred on p, so at p = 0 or 1
    # rounding can put the bound a fraction of an ulp on the wrong side
    return [[max(0.0, s - lo) for s, lo in zip(scores, los)],
            [max(0.0, hi - s) for s, hi in zip(scores, his)]]


def finish(fig, ax, name: str, title: str, xlabel: str, ylabel: str, legend=True):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(f"{FIGURES}/{name}.png", dpi=150)
    plt.close(fig)
    print(f"  -> {FIGURES}/{name}.png")


# --------------------------------------------------------------------------
def fig_E2():
    d = load("E2")
    if not d:
        return
    s = [m["score"] for m in d["results"]]
    lo = [m["ci_low"] for m in d["results"]]
    hi = [m["ci_high"] for m in d["results"]]
    n = d["results"][0]["games"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(d["c_values"], s, yerr=errbars(s, lo, hi), marker="o",
                capsize=4, label=f"UCT ({d['playouts']} playouts)")
    ax.axhline(0.5, linestyle="--", color="grey", linewidth=1,
               label="parité (50 %)")
    ax.set_ylim(0, 1)
    finish(fig, ax, "fig_E2_constante_exploration",
           f"E2 — Constante d'exploration de UCT contre {d['opponent']}\n"
           f"({n} parties par point, intervalles de Wilson à 95 %)",
           "constante d'exploration $c$", "taux de victoire de UCT")


def fig_E3():
    d = load("E3")
    if not d:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, lab in zip(
        axes,
        ["loss_total", "loss_policy", "loss_value"],
        ["perte totale", "perte policy (entropie croisée)", "perte value (MSE)"],
    ):
        for tag, style in (("run1", "--"), ("run2", "-")):
            if tag in d:
                ax.plot(d[tag]["iter"], d[tag][key], style, marker="o",
                        markersize=3, label=RUN_LABEL[tag])
        ax.set_xlabel("itération")
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("E3 — Courbes d'apprentissage par itération", fontsize=11)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(f"{FIGURES}/fig_E3_courbes_apprentissage.png", dpi=150)
    plt.close(fig)
    print(f"  -> {FIGURES}/fig_E3_courbes_apprentissage.png")


CRIT_LABEL = {
    "C1_puct_vs_random": "C1 — PUCT vs Aléatoire (seuil 98 %)",
    "C2_puct_vs_uct": "C2 — PUCT vs UCT (seuil 70 %)",
    "C3_puct_vs_alphabeta4": "C3 — PUCT vs alpha-bêta d.4 (seuil 60 %)",
    "C4_network_vs_random": "C4 — Réseau seul vs Aléatoire (seuil 85 %)",
}
CRIT_THRESHOLD = {"C1_puct_vs_random": 0.98, "C2_puct_vs_uct": 0.70,
                  "C3_puct_vs_alphabeta4": 0.60, "C4_network_vs_random": 0.85}
RUN_LABEL = {"run1": "run 1 (default.yaml)", "run2": "run 2 (tuned.yaml)"}


def fig_E4():
    d = load("E4")
    if not d:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key in zip(axes.ravel(), d["criteria"]):
        for tag, style in (("run1", "--"), ("run2", "-")):
            r = d["runs"].get(tag)
            if not r or not r["iter"]:
                continue
            ax.errorbar(r["iter"], r[key],
                        yerr=errbars(r[key], r[key + "_lo"], r[key + "_hi"]),
                        fmt=style, marker="o", capsize=3, markersize=4,
                        label=RUN_LABEL[tag])
        ax.axhline(CRIT_THRESHOLD[key], color="red", linestyle=":", linewidth=1.2,
                   label="seuil visé")
        ax.set_ylim(0, 1.05)
        ax.set_title(CRIT_LABEL[key], fontsize=10)
        ax.set_xlabel("itération")
        ax.set_ylabel("taux de victoire")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("E4 — Force au fil des itérations (intervalles de Wilson à 95 %)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/fig_E4_force_par_iteration.png", dpi=150)
    plt.close(fig)
    print(f"  -> {FIGURES}/fig_E4_force_par_iteration.png")


def fig_E5():
    d = load("E5")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key, lab, mk in (("puct", "PUCT (réseau final)", "o"),
                         ("uct", "UCT (playouts aléatoires)", "s")):
        s = [m["score"] for m in d[key]]
        lo = [m["ci_low"] for m in d[key]]
        hi = [m["ci_high"] for m in d[key]]
        ax.errorbar(d["budgets"], s, yerr=errbars(s, lo, hi), marker=mk,
                    capsize=4, label=lab)
    ax.axhline(0.5, linestyle="--", color="grey", linewidth=1, label="parité")
    ax.set_xscale("log")
    ax.set_xticks(d["budgets"])
    ax.set_xticklabels([str(b) for b in d["budgets"]])
    # a log axis adds its own minor ticks, which collide with the labels above
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0, 1)
    n = d["puct"][0]["games"]
    finish(fig, ax, "fig_E5_budget_simulations",
           f"E5 — Force en fonction du budget, contre {d['opponent']}\n"
           f"{n} parties distinctes par point (ouvertures à 2 coups),\n"
           f"intervalles de Wilson à 95 %",
           "budget (simulations ou playouts, échelle log)",
           "taux de victoire")


def fig_E6():
    d = load("E6")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for tag, style in (("run1", "--"), ("run2", "-")):
        if tag in d and any(x is not None for x in d[tag]["net_agreement"]):
            xs = [i for i, v in zip(d[tag]["iter"], d[tag]["net_agreement"])
                  if v is not None]
            ys = [v for v in d[tag]["net_agreement"] if v is not None]
            ax.plot(xs, ys, style, marker="o", markersize=4, label=RUN_LABEL[tag])
    diag = load("diagnostic")
    if diag:
        base = diag["runs"]["run2"][0]["policy_agreement"]
        ax.axhline(base, color="grey", linestyle=":",
                   label=f"réseau non entraîné ({pct(base)})")
    ax.set_ylim(0, 1)
    finish(fig, ax, "fig_E6_accord_jeu_parfait",
           "E6 — Accord de la tête policy avec le jeu parfait\n"
           "(200 finales à 20 pions résolues exactement par alpha-bêta)",
           "itération", "taux d'accord (argmax de la policy)")


def fig_E7():
    d = load("E7")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for tag, style in (("run1", "--"), ("run2", "-")):
        if tag in d:
            ax.plot(d[tag]["iter"], d[tag]["prior_center"], style, marker="o",
                    markersize=4, label=RUN_LABEL[tag])
    ax.axhline(1 / 7, color="grey", linestyle=":", label="uniforme (1/7 ≈ 0,143)")
    ax.set_ylim(0, 1)
    finish(fig, ax, "fig_E7_prior_colonne_centrale",
           "E7 — Probabilité donnée par la policy à la colonne centrale\n"
           "sur le plateau vide (le meilleur coup d'ouverture du jeu résolu)",
           "itération", "$P(\\mathrm{colonne}\\ 3\\ |\\ \\mathrm{plateau\\ vide})$")


def fig_diagnostic():
    d = load("diagnostic")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for tag, style in (("run1", "--"), ("run2", "-")):
        rows = d["runs"].get(tag)
        if not rows:
            continue
        it = [r["iter"] for r in rows]
        axes[0].plot(it, [r["value_saturated_frac"] for r in rows], style,
                     marker="o", markersize=4, label=RUN_LABEL[tag])
        axes[1].plot(it, [(r["puct_takes_win"] + r["puct_blocks"]) / (2 * r["n_tactical"])
                          for r in rows], style, marker="o", markersize=4,
                     label=RUN_LABEL[tag])
    axes[0].set_xlabel("itération (0 = réseau non entraîné)")
    axes[0].set_ylabel("fraction de positions avec $|v| = 1$")
    axes[0].set_title("Saturation de la tête value", fontsize=10)
    axes[1].set_xlabel("itération (0 = réseau non entraîné)")
    axes[1].set_ylabel("score tactique de PUCT (100 sim.)")
    axes[1].set_title("Tactique de PUCT : gains immédiats pris\net pertes immédiates parées",
                      fontsize=10)
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Diagnostic — la saturation de la value dégrade la tactique de PUCT",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/fig_diagnostic_saturation.png", dpi=150)
    plt.close(fig)
    print(f"  -> {FIGURES}/fig_diagnostic_saturation.png")


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def tables() -> str:
    out = []

    d = load("E1")
    if d:
        out.append(f"\n### Tableau E1 — l'échelle du cours, {d['playouts']} playouts\n")
        out.append("| A | B | victoire de A | IC Wilson 95 % | V/N/D | parties | Elo |")
        out.append("|---|---|---|---|---|---|---|")
        for m in d["matches"]:
            out.append(f"| {m['name_a']} | {m['name_b']} | {pct(m['score'])} | "
                       f"[{pct(m['ci_low'])}, {pct(m['ci_high'])}] | "
                       f"{m['wins']}/{m['draws']}/{m['losses']} | {m['games']} | "
                       f"{m['elo']:+.0f} |")

    d = load("C")
    if d:
        out.append(f"\n### Tableau C — critères de réussite, réseau final "
                   f"({d['checkpoint']}, {d['sims']} simulations)\n")
        out.append("| Critère | Seuil | Mesuré | IC Wilson 95 % | parties "
                   "distinctes | Atteint |")
        out.append("|---|---|---|---|---|---|")
        for k, m in d["results"].items():
            thr = d["thresholds"][k]
            out.append(f"| {CRIT_LABEL[k].split(' (')[0]} | {pct(thr)} | "
                       f"**{pct(m['score'])}** | [{pct(m['ci_low'])}, "
                       f"{pct(m['ci_high'])}] | {m['games']} | "
                       f"{'oui' if m['score'] >= thr else 'NON'} |")

    d = load("E4")
    if d:
        for tag in ("run1", "run2"):
            r = d["runs"].get(tag)
            if not r or not r["iter"]:
                continue
            out.append(f"\n### Tableau C1–C4 (suivi en cours d'entraînement) — "
                       f"{RUN_LABEL[tag]}, dernière itération\n")
            out.append("| Critère | Seuil | Mesuré | IC Wilson 95 % | parties | Atteint |")
            out.append("|---|---|---|---|---|---|")
            for k in d["criteria"]:
                v, lo, hi = r[k][-1], r[k + "_lo"][-1], r[k + "_hi"][-1]
                thr = CRIT_THRESHOLD[k]
                out.append(f"| {CRIT_LABEL[k].split(' (')[0]} | {pct(thr)} | "
                           f"**{pct(v)}** | [{pct(lo)}, {pct(hi)}] | "
                           f"{r['games'][-1]} | {'oui' if v >= thr else 'NON'} |")

    d = load("E8")
    if d:
        out.append(f"\n### Tableau E8 — UCT / RAVE / GRAVE (ref = {d['grave_ref']})\n")
        out.append("| budget | confrontation | victoire de A | IC Wilson 95 % | V/N/D |")
        out.append("|---|---|---|---|---|")
        for n in d["budgets"]:
            for m in d["results"][str(n)]:
                out.append(f"| {n} | {m['name_a']} vs {m['name_b']} | "
                           f"{pct(m['score'])} | [{pct(m['ci_low'])}, "
                           f"{pct(m['ci_high'])}] | "
                           f"{m['wins']}/{m['draws']}/{m['losses']} |")

    d = load("E9")
    if d:
        out.append("\n### Tableau E9 — coût par coup et table de transposition\n")
        out.append("| agent | ms / coup | taux de hit de la table |")
        out.append("|---|---|---|")
        for r in d["rows"]:
            hit = "—" if r["tt_hit_rate"] == 0 else pct(r["tt_hit_rate"])
            out.append(f"| {r['name']} | {r['ms_per_move']:.2f} | {hit} |")

    d = load("diagnostic")
    if d:
        out.append("\n### Tableau D — diagnostic du run 1 (§8)\n")
        out.append("| checkpoint | value saturée $|v|=1$ | écart-type de $v$ | "
                   "gains pris /30 | parades /30 | accord policy |")
        out.append("|---|---|---|---|---|---|")
        for r in d["runs"]["run1"]:
            name = "non entraîné" if r["iter"] == 0 else f"iter {r['iter']:02d}"
            out.append(f"| {name} | {pct(r['value_saturated_frac'])} | "
                       f"{r['value_std']:.3f} | {r['puct_takes_win']} | "
                       f"{r['puct_blocks']} | {pct(r['policy_agreement'])} |")
    return "\n".join(out) + "\n"


def main() -> int:
    os.makedirs(FIGURES, exist_ok=True)
    for f in (fig_E2, fig_E3, fig_E4, fig_E5, fig_E6, fig_E7, fig_diagnostic):
        f()
    md = tables()
    with open(f"{RESULTS}/tables.md", "w") as fh:
        fh.write("<!-- GENERATED by scripts/make_figures.py -- do not edit -->\n")
        fh.write(md)
    print(f"  -> {RESULTS}/tables.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
