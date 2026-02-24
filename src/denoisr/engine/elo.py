"""Elo rating and SPRT computation for engine-vs-engine matches."""

from __future__ import annotations

import math


def compute_elo(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """Compute Elo difference and 95% confidence error margin."""
    total = wins + draws + losses
    if total == 0:
        return (0.0, 0.0)

    score = (wins + draws / 2) / total

    if score <= 0.0:
        return (float("-inf"), 0.0)
    if score >= 1.0:
        return (float("inf"), 0.0)

    elo_diff = -400.0 * math.log10(1.0 / score - 1.0)

    variance = (
        wins * (1.0 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score**2
    ) / total**2

    derivative = 400.0 / (score * (1.0 - score) * math.log(10))
    error_95 = derivative * math.sqrt(variance) * 1.96

    return (elo_diff, error_95)


def likelihood_of_superiority(wins: int, draws: int, losses: int) -> float:
    """Compute LOS (likelihood of superiority) as a percentage."""
    total = wins + draws + losses
    if total == 0:
        return 50.0

    score = (wins + draws / 2) / total

    if score <= 0.0:
        return 0.0
    if score >= 1.0:
        return 100.0

    variance = (
        wins * (1.0 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score**2
    ) / total**2

    if variance <= 0.0:
        return 50.0

    z = (score - 0.5) / math.sqrt(variance)
    los = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    return los * 100.0


def sprt_test(
    wins: int,
    draws: int,
    losses: int,
    elo0: float,
    elo1: float,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> str | None:
    """Sequential Probability Ratio Test."""
    total = wins + draws + losses
    if total == 0:
        return None

    score = (wins + draws / 2) / total
    if score <= 0.0 or score >= 1.0:
        return None

    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)

    p0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
    p1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))

    llr = total * (
        score * math.log(p1 / p0) + (1.0 - score) * math.log((1.0 - p1) / (1.0 - p0))
    )

    if llr >= upper:
        return "H1"
    if llr <= lower:
        return "H0"
    return None
