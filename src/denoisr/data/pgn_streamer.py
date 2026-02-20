from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import chess.pgn

from denoisr.types import Action, GameRecord

_RESULT_MAP = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


class SimplePGNStreamer:
    def stream(self, path: Path) -> Iterator[GameRecord]:
        suffix = "".join(path.suffixes)
        if suffix.endswith(".zst"):
            yield from self._stream_zst(path)
        else:
            yield from self._stream_pgn(path)

    def _stream_pgn(self, path: Path) -> Iterator[GameRecord]:
        with open(path) as f:
            yield from self._parse_games(f)

    def _stream_zst(self, path: Path) -> Iterator[GameRecord]:
        import io

        import zstandard as zstd

        with open(path, "rb") as fh:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            yield from self._parse_games(text_stream)

    def _parse_games(
        self, stream: TextIO
    ) -> Iterator[GameRecord]:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            result_str = game.headers.get("Result", "*")
            result = _RESULT_MAP.get(result_str)
            if result is None:
                continue
            eco_code = game.headers.get("ECO")
            actions = tuple(
                Action(m.from_square, m.to_square, m.promotion)
                for m in game.mainline_moves()
            )
            yield GameRecord(actions=actions, result=result, eco_code=eco_code)
