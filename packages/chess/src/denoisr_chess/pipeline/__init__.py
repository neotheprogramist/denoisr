"""Domain-agnostic pipeline orchestration helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ALL_STEPS",
    "DataConfig",
    "ModelSectionConfig",
    "OutputConfig",
    "Phase1Config",
    "Phase2Config",
    "Phase3Config",
    "PipelineConfig",
    "PipelineRunner",
    "PipelineState",
    "load_config",
    "required_env_vars",
]

_EXPORTS = {
    "ALL_STEPS": ("denoisr_chess.pipeline.runner", "ALL_STEPS"),
    "DataConfig": ("denoisr_chess.pipeline.config", "DataConfig"),
    "ModelSectionConfig": ("denoisr_chess.pipeline.config", "ModelSectionConfig"),
    "OutputConfig": ("denoisr_chess.pipeline.config", "OutputConfig"),
    "Phase1Config": ("denoisr_chess.pipeline.config", "Phase1Config"),
    "Phase2Config": ("denoisr_chess.pipeline.config", "Phase2Config"),
    "Phase3Config": ("denoisr_chess.pipeline.config", "Phase3Config"),
    "PipelineConfig": ("denoisr_chess.pipeline.config", "PipelineConfig"),
    "PipelineRunner": ("denoisr_chess.pipeline.runner", "PipelineRunner"),
    "PipelineState": ("denoisr_chess.pipeline.state", "PipelineState"),
    "load_config": ("denoisr_chess.pipeline.config", "load_config"),
    "required_env_vars": ("denoisr_chess.pipeline.config", "required_env_vars"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module protocol
        raise AttributeError(name) from exc
    module = import_module(module_name)
    return getattr(module, attr_name)
