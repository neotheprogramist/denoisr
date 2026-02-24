import io
import logging

import pytest

from denoisr.scripts.interrupts import graceful_main


def test_graceful_main_passthrough() -> None:
    called = {"value": False}

    @graceful_main("test-script")
    def _main() -> None:
        called["value"] = True

    _main()
    assert called["value"] is True


def test_graceful_main_keyboardinterrupt_exits_130_and_logs() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("tests.graceful_main")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    try:

        @graceful_main("test-script", logger=logger)
        def _main() -> None:
            raise KeyboardInterrupt

        with pytest.raises(SystemExit) as exc:
            _main()

        assert exc.value.code == 130
        assert "test-script: Interrupted by user (Ctrl+C)." in stream.getvalue()
    finally:
        logger.removeHandler(handler)
        handler.close()
