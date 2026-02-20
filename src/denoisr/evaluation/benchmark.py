import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    engine_cmd: str
    opponent_cmd: str
    games: int = 100
    time_control: str = "10+0.1"
    sprt_elo0: int | None = None
    sprt_elo1: int | None = None
    concurrency: int = 1


def build_cutechess_command(config: BenchmarkConfig) -> str:
    parts = [
        "cutechess-cli",
        f"-engine cmd={config.engine_cmd} proto=uci",
        f"-engine cmd={config.opponent_cmd} proto=uci",
        f"-games {config.games}",
        f"-each tc={config.time_control}",
        f"-concurrency {config.concurrency}",
    ]
    if config.sprt_elo0 is not None and config.sprt_elo1 is not None:
        parts.append(
            f"-sprt elo0={config.sprt_elo0} elo1={config.sprt_elo1}"
            " alpha=0.05 beta=0.05"
        )
    return " ".join(parts)


def parse_cutechess_output(output: str) -> dict[str, float | str]:
    result: dict[str, float | str] = {}

    elo_match = re.search(
        r"Elo difference: ([-\d.]+) \+/- ([\d.]+)", output
    )
    if elo_match:
        result["elo_diff"] = float(elo_match.group(1))
        result["elo_error"] = float(elo_match.group(2))

    los_match = re.search(r"LOS: ([\d.]+)", output)
    if los_match:
        result["los"] = float(los_match.group(1))

    if "H1 was accepted" in output:
        result["sprt_result"] = "H1"
    elif "H0 was accepted" in output:
        result["sprt_result"] = "H0"

    return result
