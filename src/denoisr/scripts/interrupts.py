"""Helpers for graceful Ctrl+C handling in CLI entry points."""

from __future__ import annotations

import functools
import logging
import sys
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def graceful_main(
    script_name: str | None = None,
    *,
    logger: logging.Logger | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a CLI main function and exit cleanly on Ctrl+C.

    The wrapped function exits with code 130 (conventional SIGINT exit code)
    instead of printing a traceback.
    """

    def _decorate(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def _wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                prefix = f"{script_name}: " if script_name else ""
                message = (
                    f"{prefix}Interrupted by user (Ctrl+C). "
                    "Exiting gracefully."
                )
                active_logger = logger or logging.getLogger(func.__module__)
                if active_logger.handlers or logging.getLogger().handlers:
                    active_logger.warning(message)
                else:
                    print(message, file=sys.stderr)
                raise SystemExit(130) from None

        return _wrapped

    return _decorate

