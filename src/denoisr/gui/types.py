"""Data types for the chess GUI and match engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True)
class TimeControl:
    base_seconds: float
    increment: float

    def __post_init__(self) -> None:
        if self.base_seconds < 0:
            raise ValueError(
                f"base_seconds must be >= 0, got {self.base_seconds}"
            )
        if self.increment < 0:
            raise ValueError(
                f"increment must be >= 0, got {self.increment}"
            )


@dataclass(frozen=True)
class EngineConfig:
    command: str
    args: tuple[str, ...]
    name: str


@dataclass(frozen=True)
class GameResult:
    moves: tuple[str, ...]
    result: str
    reason: str
    engine1_color: str

    def __post_init__(self) -> None:
        valid_results = {"1-0", "0-1", "1/2-1/2", "*"}
        if self.result not in valid_results:
            raise ValueError(
                f"result must be one of {valid_results}, got {self.result!r}"
            )
        if self.engine1_color not in {"white", "black"}:
            raise ValueError(
                f"engine1_color must be 'white' or 'black', "
                f"got {self.engine1_color!r}"
            )


@dataclass(frozen=True)
class MatchConfig:
    engine1: EngineConfig
    engine2: EngineConfig
    games: int
    time_control: TimeControl
    concurrency: int = 1

    def __post_init__(self) -> None:
        if self.games <= 0:
            raise ValueError(f"games must be > 0, got {self.games}")
        if self.concurrency <= 0:
            raise ValueError(
                f"concurrency must be > 0, got {self.concurrency}"
            )


class GameOutcome(Enum):
    WIN = auto()
    DRAW = auto()
    LOSS = auto()
