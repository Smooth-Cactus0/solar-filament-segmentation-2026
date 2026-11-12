"""Experiment configs with `base:` inheritance.

Screening experiments are one-variable changes from a baseline, so each screening
config is a few lines -- `base: scr_r0_base_1024.yaml` plus the override -- and the
diff *is* the experiment. `base` is resolved relative to the file that names it;
dicts merge recursively, everything else (scalars, lists) is replaced.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """``base`` updated by ``override``; nested dicts merge, other values replace."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: Path, seen: tuple[Path, ...]) -> dict:
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ValueError(f"config inheritance cycle: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base = raw.pop("base", None)
    if base is None:
        return raw
    parent = _load((path.parent / base).resolve(), (*seen, path))
    return deep_merge(parent, raw)


def load_config(path: str | Path) -> dict:
    """Load a YAML config, resolving its `base:` chain."""
    return _load(Path(path).resolve(), ())
