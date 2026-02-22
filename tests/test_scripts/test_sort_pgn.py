from pathlib import Path

import pytest

from denoisr.data.game_format import read_game_records
from denoisr.scripts.sort_pgn import (
    _bucket_name,
    _min_elo,
    _parse_ranges,
    sort_pgn_to_games,
)


class TestParseRanges:
    def test_standard_ranges(self) -> None:
        ranges = _parse_ranges("0-1200,1200-1600,1600-2000,2000-2400,2400+")
        assert ranges == [
            (0, 1200),
            (1200, 1600),
            (1600, 2000),
            (2000, 2400),
            (2400, None),
        ]

    def test_single_range(self) -> None:
        assert _parse_ranges("1000+") == [(1000, None)]


class TestBucketName:
    def test_bounded(self) -> None:
        assert _bucket_name(1200, 1600) == "1200-1600"

    def test_unbounded(self) -> None:
        assert _bucket_name(2400, None) == "2400+"


class TestMinElo:
    def test_both_present(self) -> None:
        assert _min_elo(1500, 1800) == 1500

    def test_one_missing(self) -> None:
        assert _min_elo(1500, None) == 1500
        assert _min_elo(None, 1800) == 1800

    def test_both_missing(self) -> None:
        assert _min_elo(None, None) is None


def _write_test_pgn(path: Path, games: list[dict[str, object]]) -> None:
    """Write a PGN file with game dicts.

    Each dict may contain: white_elo, black_elo, result, eco, moves.
    Defaults: result="1-0", eco=None, moves="1. e4 e5 2. Nf3 Nc6".
    """
    with open(path, "w") as f:
        for g in games:
            f.write('[Event "Test"]\n')
            f.write('[Site "Test"]\n')
            f.write('[Date "2024.01.01"]\n')
            f.write('[Round "1"]\n')
            f.write('[White "Player1"]\n')
            f.write('[Black "Player2"]\n')
            result = str(g.get("result", "1-0"))
            f.write(f'[Result "{result}"]\n')
            if "white_elo" in g:
                f.write(f'[WhiteElo "{g["white_elo"]}"]\n')
            if "black_elo" in g:
                f.write(f'[BlackElo "{g["black_elo"]}"]\n')
            if "eco" in g:
                f.write(f'[ECO "{g["eco"]}"]\n')
            moves = str(g.get("moves", "1. e4 e5 2. Nf3 Nc6"))
            f.write(f"\n{moves} {result}\n\n")


class TestSortCreatesGamesFiles:
    def test_sort_creates_games_files(self, tmp_path: Path) -> None:
        """Three games at different Elos land in the correct .games files."""
        pgn_path = tmp_path / "input.pgn"
        _write_test_pgn(
            pgn_path,
            [
                {"white_elo": 1000, "black_elo": 1100, "result": "1-0"},
                {"white_elo": 1500, "black_elo": 1400, "result": "0-1"},
                {"white_elo": 2500, "black_elo": 2600, "result": "1/2-1/2"},
            ],
        )

        output_dir = tmp_path / "sorted"
        ranges = _parse_ranges("0-1200,1200-1600,2400+")
        counts = sort_pgn_to_games(pgn_path, output_dir, ranges)

        # Verify counts
        assert counts == {"0-1200": 1, "1200-1600": 1, "2400+": 1}

        # Verify .games files exist and contain correct records
        low_records = list(read_game_records(output_dir / "0-1200.games"))
        assert len(low_records) == 1
        assert low_records[0].white_elo == 1000
        assert low_records[0].black_elo == 1100
        assert low_records[0].result == pytest.approx(1.0)

        mid_records = list(read_game_records(output_dir / "1200-1600.games"))
        assert len(mid_records) == 1
        assert mid_records[0].white_elo == 1500
        assert mid_records[0].black_elo == 1400
        assert mid_records[0].result == pytest.approx(-1.0)

        hi_records = list(read_game_records(output_dir / "2400+.games"))
        assert len(hi_records) == 1
        assert hi_records[0].white_elo == 2500
        assert hi_records[0].black_elo == 2600
        assert hi_records[0].result == pytest.approx(0.0)

    def test_moves_round_trip(self, tmp_path: Path) -> None:
        """Moves survive the PGN -> sort -> .games round-trip."""
        pgn_path = tmp_path / "moves.pgn"
        _write_test_pgn(
            pgn_path,
            [{"white_elo": 1500, "black_elo": 1500, "moves": "1. e4 e5"}],
        )
        output_dir = tmp_path / "out"
        ranges = _parse_ranges("0-2000")
        sort_pgn_to_games(pgn_path, output_dir, ranges)

        records = list(read_game_records(output_dir / "0-2000.games"))
        assert len(records) == 1
        assert len(records[0].actions) == 2  # e4, e5


class TestSortSkipsGamesWithoutElo:
    def test_sort_skips_games_without_elo(self, tmp_path: Path) -> None:
        """Games without Elo headers are skipped."""
        pgn_path = tmp_path / "mixed.pgn"
        _write_test_pgn(
            pgn_path,
            [
                {"white_elo": 1500, "black_elo": 1500, "result": "1-0"},
                {"result": "0-1"},  # no Elo at all
                {"white_elo": 1500, "result": "1-0"},  # only one Elo
            ],
        )
        output_dir = tmp_path / "out"
        ranges = _parse_ranges("0-2000")
        counts = sort_pgn_to_games(pgn_path, output_dir, ranges)

        # First game: min(1500,1500) = 1500 -> bucket
        # Second game: no Elo -> skipped
        # Third game: min_elo(1500, None) = 1500 -> bucket
        assert counts == {"0-2000": 2}
        records = list(read_game_records(output_dir / "0-2000.games"))
        assert len(records) == 2
