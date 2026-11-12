"""Run a Jupyter notebook's code cells in order, without Jupyter.

The competition asks for one notebook that reproduces the submission
(``notebooks/pipeline.ipynb``). Where Jupyter, a kernel or nbclient are not installed (the
project laptop, a bare Slurm node), this runner executes the notebook's code cells top to
bottom in one Python namespace with plain ``exec``, as "Run All" would, and prints each
cell's index and wall time. Nothing but the standard library is needed to read the
notebook: an ``.ipynb`` file is JSON.

``--set NAME=VALUE`` overrides a configuration variable. The overrides are applied right
after the cell tagged ``parameters`` (the papermill convention), so a run can change
``LIMIT``, ``FROM_MAPS``, ``WORK_DIR`` ... without editing the notebook. ``VALUE`` is read
as a Python literal (``3``, ``True``, ``None``, ``'cuda'``) and, when it is not one, kept as
a plain string (a path). Only names the parameters cell defines can be set, so a typo fails
instead of being ignored.

Differences from a kernel: IPython magics and shell escapes are not supported (the
pipeline notebook uses none), the last expression of a cell is not displayed, and
matplotlib runs on the non-interactive Agg backend; figures are closed after each cell,
and saved first as PNG under ``--figures DIR`` when given. The working directory is the
notebook's folder during the run, as in Jupyter.

Usage:
    PYTHONPATH=src python scripts/run_notebook.py notebooks/pipeline.ipynb \\
        --set LIMIT=3 --set FROM_MAPS=True --set WORK_DIR=/tmp/nbrun --figures /tmp/nbrun/figures
"""
from __future__ import annotations

import argparse
import ast
import json
import linecache
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

PARAMETERS_TAG = "parameters"


def _print(msg: str) -> None:
    print(msg, flush=True)


@dataclass(frozen=True)
class Cell:
    """One code cell: its position among all cells, its source and its tags."""

    index: int
    source: str
    tags: tuple[str, ...]


def load_cells(path: str | Path) -> list[Cell]:
    """The code cells of an nbformat-4 notebook, in order (markdown and raw cells skipped)."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4:
        raise ValueError(f"{path}: nbformat {nb.get('nbformat')}, expected 4")
    cells = []
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", "")
        cells.append(Cell(i, "".join(src) if isinstance(src, list) else src,
                          tuple(c.get("metadata", {}).get("tags", ()))))
    return cells


def parse_overrides(pairs: list[str]) -> dict[str, object]:
    """``["LIMIT=3", "WORK_DIR=C:/x"]`` to ``{"LIMIT": 3, "WORK_DIR": "C:/x"}``."""
    out = {}
    for pair in pairs:
        name, sep, raw = pair.partition("=")
        name = name.strip()
        if not sep or not name.isidentifier():
            raise ValueError(f"--set expects NAME=VALUE, got {pair!r}")
        try:
            out[name] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[name] = raw
    return out


def assigned_names(source: str) -> set[str]:
    """The names a cell assigns at its top level (``A = 1``, ``A: int = 1``, ``A, B = ...``)."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
        for t in targets:
            names |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    return names


def _close_figures(figures: Path | None, cell: Cell, log) -> None:
    """Save (when asked) and close every open matplotlib figure, if matplotlib is in use."""
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return
    for k, num in enumerate(plt.get_fignums(), 1):
        if figures is not None:
            figures.mkdir(parents=True, exist_ok=True)
            out = figures / f"cell{cell.index:02d}_{k}.png"
            plt.figure(num).savefig(out, dpi=90)
            log(f"        figure -> {out}")
    plt.close("all")


def run(path: str | Path, overrides: dict[str, object] | None = None, figures: str | Path | None = None,
        cwd: str | Path | None = None, log=_print) -> dict:
    """Execute the notebook's code cells in one namespace and return that namespace.

    Raises:
        ValueError: overrides given but no cell is tagged ``parameters``, or an override
            names a variable the parameters cell does not define.
        Exception: whatever a cell raises, after logging which cell it was.
    """
    path = Path(path).resolve()
    cells = load_cells(path)
    overrides = dict(overrides or {})
    tagged = [c for c in cells if PARAMETERS_TAG in c.tags]
    if overrides and not tagged:
        raise ValueError(f"{path.name} has no cell tagged {PARAMETERS_TAG!r} to apply --set after")
    figures = Path(figures).resolve() if figures else None
    os.environ.setdefault("MPLBACKEND", "Agg")
    warnings.filterwarnings("ignore", message=".*non-interactive.*cannot be shown")

    ns: dict = {"__name__": "__main__"}
    old_cwd = Path.cwd()
    os.chdir(cwd or path.parent)
    t_all = time.time()
    try:
        for n, cell in enumerate(cells, 1):
            name = f"<{path.name} cell {cell.index}>"
            linecache.cache[name] = (len(cell.source), None, cell.source.splitlines(True), name)
            log(f"[cell {cell.index:2d}] ({n}/{len(cells)}) running")
            t0 = time.time()
            try:
                exec(compile(cell.source, name, "exec"), ns)
            except BaseException:
                log(f"[cell {cell.index:2d}] FAILED after {time.time() - t0:.1f}s")
                raise
            if PARAMETERS_TAG in cell.tags and overrides:
                unknown = sorted(set(overrides) - assigned_names(cell.source))
                if unknown:
                    raise ValueError(f"--set names a variable the parameters cell does not define: {unknown}")
                ns.update(overrides)
                log("        overrides: " + ", ".join(f"{k}={v!r}" for k, v in overrides.items()))
            _close_figures(figures, cell, log)
            log(f"[cell {cell.index:2d}] done in {time.time() - t0:.1f}s")
    finally:
        os.chdir(old_cwd)
    log(f"all {len(cells)} code cells ran in {time.time() - t_all:.1f}s")
    return ns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                    help="override a variable of the parameters cell (repeatable)")
    ap.add_argument("--figures", default=None, help="save every figure as PNG in this folder")
    ap.add_argument("--cwd", default=None, help="working directory (default: the notebook's folder)")
    args = ap.parse_args(argv)
    try:
        run(args.notebook, parse_overrides(args.overrides), args.figures, args.cwd)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
