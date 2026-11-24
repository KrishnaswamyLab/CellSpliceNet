"""Load ordered GOBP RNA splicing gene symbols from MSigDB .grp file."""

from __future__ import annotations

from pathlib import Path


def load_gobp_symbols(grp_path: str | Path) -> list[str]:
    path = Path(grp_path)
    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line == "GOBP_RNA_SPLICING":
            continue
        symbols.append(line)
    return symbols
