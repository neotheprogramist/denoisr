import pytest

from denoisr.gui.types import (
    EngineConfig,
    GameResult,
    MatchConfig,
    TimeControl,
)


class TestTimeControl:
    def test_create(self) -> None:
        tc = TimeControl(base_seconds=10.0, increment=0.1)
        assert tc.base_seconds == 10.0
        assert tc.increment == 0.1

    def test_frozen(self) -> None:
        tc = TimeControl(base_seconds=10.0, increment=0.1)
        with pytest.raises(AttributeError):
            tc.base_seconds = 5.0  # type: ignore[misc]

    def test_rejects_negative_time(self) -> None:
        with pytest.raises(ValueError, match="base_seconds"):
            TimeControl(base_seconds=-1.0, increment=0.1)

    def test_rejects_negative_increment(self) -> None:
        with pytest.raises(ValueError, match="increment"):
            TimeControl(base_seconds=10.0, increment=-0.5)


class TestEngineConfig:
    def test_create(self) -> None:
        ec = EngineConfig(
            command="uv",
            args=("run", "denoisr-play", "--checkpoint", "model.pt"),
            name="Denoisr",
        )
        assert ec.command == "uv"
        assert ec.args == ("run", "denoisr-play", "--checkpoint", "model.pt")

    def test_frozen(self) -> None:
        ec = EngineConfig(command="uv", args=(), name="test")
        with pytest.raises(AttributeError):
            ec.name = "other"  # type: ignore[misc]


class TestGameResult:
    def test_create(self) -> None:
        gr = GameResult(
            moves=("e2e4", "e7e5"),
            result="1-0",
            reason="checkmate",
            engine1_color="white",
        )
        assert gr.result == "1-0"
        assert gr.reason == "checkmate"

    def test_rejects_invalid_result(self) -> None:
        with pytest.raises(ValueError, match="result"):
            GameResult(
                moves=(),
                result="2-0",
                reason="checkmate",
                engine1_color="white",
            )

    def test_rejects_invalid_color(self) -> None:
        with pytest.raises(ValueError, match="engine1_color"):
            GameResult(
                moves=(),
                result="1-0",
                reason="checkmate",
                engine1_color="red",
            )


class TestMatchConfig:
    def test_create(self) -> None:
        e1 = EngineConfig(command="eng1", args=(), name="Engine 1")
        e2 = EngineConfig(command="eng2", args=(), name="Engine 2")
        tc = TimeControl(base_seconds=10.0, increment=0.1)
        mc = MatchConfig(engine1=e1, engine2=e2, games=100, time_control=tc)
        assert mc.games == 100
    def test_rejects_zero_games(self) -> None:
        e1 = EngineConfig(command="eng1", args=(), name="E1")
        e2 = EngineConfig(command="eng2", args=(), name="E2")
        tc = TimeControl(base_seconds=10.0, increment=0.1)
        with pytest.raises(ValueError, match="games"):
            MatchConfig(engine1=e1, engine2=e2, games=0, time_control=tc)
