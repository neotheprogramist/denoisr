from denoisr_chess.engine.elo import compute_elo, likelihood_of_superiority, sprt_test


class TestComputeElo:
    def test_even_score_is_zero_elo(self) -> None:
        elo, error = compute_elo(wins=50, draws=0, losses=50)
        assert abs(elo) < 0.1

    def test_all_wins_is_large_positive(self) -> None:
        elo, _error = compute_elo(wins=100, draws=0, losses=0)
        assert elo == float("inf")

    def test_all_losses_is_large_negative(self) -> None:
        elo, _error = compute_elo(wins=0, draws=0, losses=100)
        assert elo == float("-inf")

    def test_75_percent_score(self) -> None:
        elo, _error = compute_elo(wins=75, draws=0, losses=25)
        assert abs(elo - 190.85) < 1.0

    def test_all_draws_is_zero_elo(self) -> None:
        elo, _error = compute_elo(wins=0, draws=100, losses=0)
        assert abs(elo) < 0.1

    def test_error_decreases_with_more_games(self) -> None:
        _, error_small = compute_elo(wins=7, draws=3, losses=10)
        _, error_large = compute_elo(wins=70, draws=30, losses=100)
        assert error_large < error_small

    def test_zero_games_returns_zero(self) -> None:
        elo, error = compute_elo(wins=0, draws=0, losses=0)
        assert elo == 0.0
        assert error == 0.0


class TestLikelihoodOfSuperiority:
    def test_even_score_is_near_50(self) -> None:
        los = likelihood_of_superiority(wins=50, draws=0, losses=50)
        assert abs(los - 50.0) < 1.0

    def test_strong_winner_is_near_100(self) -> None:
        los = likelihood_of_superiority(wins=90, draws=5, losses=5)
        assert los > 99.0

    def test_strong_loser_is_near_0(self) -> None:
        los = likelihood_of_superiority(wins=5, draws=5, losses=90)
        assert los < 1.0

    def test_zero_games_returns_50(self) -> None:
        los = likelihood_of_superiority(wins=0, draws=0, losses=0)
        assert abs(los - 50.0) < 0.1


class TestSprtTest:
    def test_clear_h1_accepted(self) -> None:
        result = sprt_test(wins=90, draws=5, losses=5, elo0=0.0, elo1=50.0)
        assert result == "H1"

    def test_clear_h0_accepted(self) -> None:
        result = sprt_test(wins=150, draws=0, losses=150, elo0=0.0, elo1=50.0)
        assert result == "H0"

    def test_inconclusive_returns_none(self) -> None:
        result = sprt_test(wins=3, draws=1, losses=2, elo0=0.0, elo1=50.0)
        assert result is None

    def test_zero_games_returns_none(self) -> None:
        result = sprt_test(wins=0, draws=0, losses=0, elo0=0.0, elo1=50.0)
        assert result is None
