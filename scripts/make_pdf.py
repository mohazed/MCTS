#!/usr/bin/env python3
"""Compile report/rapport.tex into report/rapport.pdf.

    python scripts/make_pdf.py

The report is a LaTeX document (XeLaTeX: it uses fontspec and babel-french).
Any of the usual engines works; `tectonic` is the easiest to install because it
is a single binary that downloads the packages it needs on first run:

    brew install tectonic          # macOS
    cargo install tectonic         # anywhere with Rust

Otherwise a TeX Live / MacTeX installation providing `xelatex` (plus `latexmk`
for the cross-reference passes) does the job.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

TEX = "report/rapport.tex"
PDF = "report/rapport.pdf"


def run(cmd: list[str], cwd: str | None = None) -> int:
    print("  $ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=cwd)


def main() -> int:
    if not os.path.exists(TEX):
        print(f"missing {TEX}")
        return 1

    if shutil.which("tectonic"):
        # tectonic resolves cross-references on its own
        rc = run(["tectonic", "rapport.tex", "--outdir", "."], cwd="report")
        if rc == 0:
            print(f"-> {PDF}")
            return 0
        print("tectonic failed; trying a local TeX installation")

    if shutil.which("latexmk"):
        rc = run(["latexmk", "-xelatex", "-interaction=nonstopmode",
                  "-halt-on-error", "rapport.tex"], cwd="report")
        if rc == 0:
            print(f"-> {PDF}")
            return 0

    if shutil.which("xelatex"):
        for _ in range(3):  # three passes: TOC + references
            rc = run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                      "rapport.tex"], cwd="report")
            if rc != 0:
                return rc
        print(f"-> {PDF}")
        return 0

    print(
        "No LaTeX engine found.\n"
        "Install one of:\n"
        "  brew install tectonic        (single binary, recommended)\n"
        "  brew install --cask mactex   (full TeX Live)\n"
        "then re-run: python scripts/make_pdf.py"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
