import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PipelineState:
    """Persisted pipeline state that enables resume after interruption.

    Intentionally mutable: the runner updates ``phase`` and other fields
    as the pipeline progresses.
    """

    phase: str = "init"
    last_checkpoint: str = ""
    last_data: str = ""
    started_at: str = ""
    updated_at: str = ""

    def save(self, path: Path) -> None:
        """Serialize state to *path* as pretty-printed JSON.

        Creates parent directories if they do not exist.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        """Load state from *path*, returning a fresh instance when missing."""
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
