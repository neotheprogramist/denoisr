import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, cast

PipelinePhase = Literal[
    "",
    "init",
    "fetched",
    "model_initialized",
    "phase1_complete",
    "phase2_complete",
    "phase3_complete",
]
_KNOWN_PHASES: Final[frozenset[str]] = frozenset(
    {
        "",
        "init",
        "fetched",
        "model_initialized",
        "phase1_complete",
        "phase2_complete",
        "phase3_complete",
    }
)


def _normalize_phase(raw: object) -> PipelinePhase:
    if isinstance(raw, str) and raw in _KNOWN_PHASES:
        return cast(PipelinePhase, raw)
    raise ValueError(
        "Invalid pipeline phase in state file: "
        f"{raw!r}. Expected one of {sorted(_KNOWN_PHASES)}"
    )


@dataclass
class PipelineState:
    """Persisted pipeline state that enables resume after interruption."""

    phase: PipelinePhase = "init"
    last_checkpoint: str = ""
    last_data: str = ""
    started_at: str = ""
    updated_at: str = ""

    def save(self, path: Path) -> None:
        """Serialize state to *path* as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        """Load state from *path*, returning a fresh instance when missing."""
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(
                f"Invalid pipeline state JSON at {path}: expected object, got {type(raw).__name__}"
            )
        clean = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        clean["phase"] = _normalize_phase(clean.get("phase", "init"))
        return cls(**clean)

