import pytest

from denoisr.evaluation.benchmark import (
    BenchmarkConfig,
    build_cutechess_command,
    parse_cutechess_output,
)


class TestBuildCommand:
    def test_basic_command(self) -> None:
        config = BenchmarkConfig(
            engine_cmd="./denoisr",
            opponent_cmd="stockfish",
            games=100,
            time_control="10+0.1",
        )
        cmd = build_cutechess_command(config)
        assert "cutechess-cli" in cmd
        assert "-games 100" in cmd
        assert "-engine cmd=./denoisr" in cmd
        assert "-engine cmd=stockfish" in cmd

    def test_sprt_parameters(self) -> None:
        config = BenchmarkConfig(
            engine_cmd="./denoisr",
            opponent_cmd="stockfish",
            games=1000,
            time_control="10+0.1",
            sprt_elo0=0,
            sprt_elo1=50,
        )
        cmd = build_cutechess_command(config)
        assert "sprt" in cmd


class TestParseOutput:
    def test_parse_elo(self) -> None:
        output = "Elo difference: 42.3 +/- 15.1, LOS: 99.2 %, DrawRatio: 30.5 %"
        result = parse_cutechess_output(output)
        assert abs(result["elo_diff"] - 42.3) < 0.1
        assert abs(result["elo_error"] - 15.1) < 0.1

    def test_parse_sprt_accept(self) -> None:
        output = "SPRT: llr 2.97 (100.0%), lbound -2.94, ubound 2.94 - H1 was accepted"
        result = parse_cutechess_output(output)
        assert result["sprt_result"] == "H1"
