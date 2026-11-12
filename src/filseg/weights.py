"""Model weights stored in the repository as parts.

GitHub refuses any file above 100 MB, and the fold checkpoints are 128 to 181 MB each. Every
checkpoint is therefore committed as numbered parts (``<name>.part0``, ``<name>.part1``, ...)
that are plain byte slices of the original file. :func:`assemble` joins them back, once, and
checks every joined file against ``SHA256SUMS``, so a model is never loaded from a truncated
or mismatched file.

    from filseg import weights
    weights.assemble("weights")        # writes weights/<name>.pt next to its parts
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PART = ".part"
SUMS = "SHA256SUMS"
MAX_PART_BYTES = 95 * 1000 * 1000      # under GitHub's 100 MB limit per file
_CHUNK = 1 << 20


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def parts_of(path: str | Path) -> list[Path]:
    """The parts of ``path`` in order (``path.part0``, ``path.part1``, ...); empty if none."""
    path = Path(path)
    found = sorted(path.parent.glob(path.name + PART + "*"), key=lambda p: int(p.name.rsplit(PART, 1)[1]))
    if [int(p.name.rsplit(PART, 1)[1]) for p in found] != list(range(len(found))):
        raise FileNotFoundError(f"the parts of {path.name} are not numbered 0..{len(found) - 1}: "
                                f"{[p.name for p in found]}")
    return found


def split(path: str | Path, out_dir: str | Path, max_bytes: int = MAX_PART_BYTES) -> list[Path]:
    """Write ``path`` into ``out_dir`` as equal parts of at most ``max_bytes`` each."""
    path, out_dir = Path(path), Path(out_dir)
    size = path.stat().st_size
    n = max(1, -(-size // max_bytes))
    step = -(-size // n)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with open(path, "rb") as src:
        for i in range(n):
            part = out_dir / f"{path.name}{PART}{i}"
            with open(part, "wb") as dst:
                left = step
                while left > 0:
                    block = src.read(min(_CHUNK, left))
                    if not block:
                        break
                    dst.write(block)
                    left -= len(block)
            written.append(part)
    return written


def read_sums(weights_dir: str | Path) -> dict[str, str]:
    """``{file name: sha256}`` from ``SHA256SUMS`` in ``weights_dir`` (empty if the file is absent)."""
    f = Path(weights_dir) / SUMS
    if not f.is_file():
        return {}
    rows = (line.split(None, 1) for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
    return {name.strip().lstrip("*"): digest for digest, name in rows}


def assemble(weights_dir: str | Path, verify: bool = True) -> list[Path]:
    """Join every ``<name>.part*`` group of ``weights_dir`` into ``<name>`` and return the files.

    A file that already exists with the right checksum is left alone, so the call is cheap
    after the first run.

    Raises:
        ValueError: a joined file does not match its line in ``SHA256SUMS``.
    """
    weights_dir = Path(weights_dir)
    sums = read_sums(weights_dir) if verify else {}
    names = sorted({p.name.rsplit(PART, 1)[0] for p in weights_dir.glob("*" + PART + "*")})
    out = []
    for name in names:
        target = weights_dir / name
        want = sums.get(name)
        if not (target.is_file() and (want is None or sha256(target) == want)):
            tmp = target.with_name(target.name + ".tmp")
            with open(tmp, "wb") as dst:
                for part in parts_of(target):
                    with open(part, "rb") as src:
                        for block in iter(lambda: src.read(_CHUNK), b""):
                            dst.write(block)
            if want is not None and sha256(tmp) != want:
                tmp.unlink()
                raise ValueError(f"{name}: the joined parts do not match {SUMS}; fetch the parts again")
            tmp.replace(target)
        out.append(target)
    return out
