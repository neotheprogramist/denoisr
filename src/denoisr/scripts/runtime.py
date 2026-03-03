"""Shared script runtime helpers.

Provides:
- `.env` loading (no external dependency)
- env-aware argparse arguments
- consistent logging setup to console + file
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")

DEFAULT_ENV_FILE = ".env"
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_PREFIX = "denoisr"


def build_parser(description: str) -> argparse.ArgumentParser:
    """Create a strict parser with clap-like ergonomics."""
    return argparse.ArgumentParser(
        description=description,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def load_env_file(path: str | Path | None = None) -> Path:
    """Load KEY=VALUE pairs from an env file into os.environ.

    Existing environment variables are not overwritten.
    If an explicit path is given and doesn't exist, raises FileNotFoundError.
    The default .env path is optional (logs info if missing).
    """
    env_path = Path(path) if path is not None else Path(DEFAULT_ENV_FILE)
    if not env_path.exists():
        if path is not None:
            raise FileNotFoundError(
                f"Env file not found: {env_path}. "
                "Pass an existing file path or omit to use optional default .env"
            )
        return env_path

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
    return env_path


def _parse_env_bool(raw: str, env_var: str) -> bool:
    norm = raw.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value for {env_var}: {raw!r} "
        "(expected one of 1/0, true/false, yes/no, on/off)"
    )


def add_env_argument(
    parser: argparse.ArgumentParser,
    *flags: str,
    env_var: str,
    required: bool = True,
    default: Any | None = None,
    **kwargs: Any,
) -> argparse.Action:
    """Add argparse option with env fallback and fail-fast required behavior."""
    help_text = kwargs.get("help", "")
    if help_text:
        kwargs["help"] = f"{help_text} [env: {env_var}]"
    else:
        kwargs["help"] = f"[env: {env_var}]"

    action = kwargs.get("action")
    arg_type = kwargs.get("type")
    env_raw = os.environ.get(env_var)
    is_bool_optional_action = action is argparse.BooleanOptionalAction

    env_value: Any | None = None
    if env_raw is not None:
        if action == "store_true" or action == "store_false" or is_bool_optional_action:
            env_value = _parse_env_bool(env_raw, env_var)
        elif arg_type is not None:
            try:
                env_value = arg_type(env_raw)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"Invalid value for {env_var}={env_raw!r}"
                ) from exc
        else:
            env_value = env_raw

    if action == "store_true":
        kwargs["default"] = bool(env_value) if env_value is not None else False
        return parser.add_argument(*flags, **kwargs)
    if action == "store_false":
        kwargs["default"] = not bool(env_value) if env_value is not None else True
        return parser.add_argument(*flags, **kwargs)
    if is_bool_optional_action:
        if env_value is not None:
            kwargs["default"] = bool(env_value)
            kwargs["required"] = False
        elif default is not None:
            kwargs["default"] = bool(default)
            kwargs["required"] = False
        else:
            kwargs["required"] = required
        return parser.add_argument(*flags, **kwargs)

    if env_value is not None:
        kwargs["default"] = env_value
        kwargs["required"] = False
    elif default is not None:
        kwargs["default"] = default
        kwargs["required"] = False
    else:
        kwargs["required"] = required

    return parser.add_argument(*flags, **kwargs)


def configure_logging(
    *,
    level: int = logging.INFO,
    filename: str | Path | None = None,
) -> Path:
    """Configure root logging to console and log file."""
    env_log_path = os.environ.get("DENOISR_LOG_FILE", "").strip()
    if filename is not None:
        log_path = Path(filename)
    elif env_log_path:
        log_path = Path(env_log_path)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = DEFAULT_LOG_DIR / f"{DEFAULT_LOG_PREFIX}_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["DENOISR_LOG_FILE"] = str(log_path)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_path
