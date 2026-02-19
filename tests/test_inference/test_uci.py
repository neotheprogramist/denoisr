from denoisr.inference.uci import format_bestmove, parse_go, parse_position


class TestParsePosition:
    def test_startpos(self) -> None:
        fen, moves = parse_position("position startpos")
        assert fen is None
        assert moves == []

    def test_startpos_with_moves(self) -> None:
        fen, moves = parse_position("position startpos moves e2e4 e7e5")
        assert fen is None
        assert moves == ["e2e4", "e7e5"]

    def test_fen(self) -> None:
        cmd = "position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        fen, moves = parse_position(cmd)
        assert fen is not None
        assert "rnbqkbnr" in fen
        assert moves == []

    def test_fen_with_moves(self) -> None:
        cmd = "position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 moves e7e5"
        fen, moves = parse_position(cmd)
        assert fen is not None
        assert moves == ["e7e5"]


class TestParseGo:
    def test_movetime(self) -> None:
        params = parse_go("go movetime 1000")
        assert params["movetime"] == 1000

    def test_depth(self) -> None:
        params = parse_go("go depth 10")
        assert params["depth"] == 10

    def test_infinite(self) -> None:
        params = parse_go("go infinite")
        assert params.get("infinite") is True

    def test_wtime_btime(self) -> None:
        params = parse_go("go wtime 60000 btime 60000 winc 1000 binc 1000")
        assert params["wtime"] == 60000
        assert params["btime"] == 60000


class TestFormatBestmove:
    def test_simple_move(self) -> None:
        assert format_bestmove("e2e4") == "bestmove e2e4"

    def test_promotion(self) -> None:
        assert format_bestmove("a7a8q") == "bestmove a7a8q"
