"""EPD opening book loader for benchmark matches."""

from __future__ import annotations

from pathlib import Path


def load_openings(path: Path) -> list[str]:
    """Load FEN positions from an EPD file (one per line).

    Lines starting with '#' and blank lines are skipped.
    """
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
