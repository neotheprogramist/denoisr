from pathlib import Path

from denoisr.scripts.sort_pgn import _parse_ranges, _bucket_name, _min_elo


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
