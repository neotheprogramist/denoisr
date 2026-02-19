from pathlib import Path

import pytest

from denoisr.data.pgn_streamer import SimplePGNStreamer
from denoisr.types import GameRecord

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"


class TestSimplePGNStreamer:
    @pytest.fixture
    def streamer(self) -> SimplePGNStreamer:
        return SimplePGNStreamer()

    def test_streams_correct_count(
        self, streamer: SimplePGNStreamer
    ) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        assert len(games) == 3

    def test_scholars_mate(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        game = games[0]
        assert game.result == 1.0
        assert len(game.actions) == 7  # 4 white + 3 black moves

    def test_fools_mate(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        game = games[1]
        assert game.result == -1.0
        assert len(game.actions) == 4

    def test_draw(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        game = games[2]
        assert game.result == 0.0

    def test_actions_are_valid(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        for game in games:
            for action in game.actions:
                assert 0 <= action.from_square < 64
                assert 0 <= action.to_square < 64

    def test_empty_file(
        self, streamer: SimplePGNStreamer, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty.pgn"
        empty.write_text("")
        games = list(streamer.stream(empty))
        assert len(games) == 0
