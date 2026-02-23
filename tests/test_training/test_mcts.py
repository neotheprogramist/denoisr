import pytest
import torch

from denoisr.training.mcts import MCTS, MCTSConfig

from conftest import SMALL_D_S


class _MockPolicyValue:
    def __init__(self, d_s: int) -> None:
        self.d_s = d_s

    def predict(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy = torch.ones(64, 64) / (64 * 64)
        value = torch.tensor([0.33, 0.34, 0.33])
        return policy, value


class _MockWorldModel:
    def __init__(self, d_s: int) -> None:
        self.d_s = d_s

    def predict_next(
        self, state: torch.Tensor, f: int, t: int
    ) -> tuple[torch.Tensor, float]:
        return torch.randn(64, self.d_s), 0.0


class TestMCTS:
    @pytest.fixture
    def config(self) -> MCTSConfig:
        return MCTSConfig(
            num_simulations=50,
            c_puct=1.4,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        )

    @pytest.fixture
    def mcts(self, config: MCTSConfig) -> MCTS:
        pv = _MockPolicyValue(SMALL_D_S)
        wm = _MockWorldModel(SMALL_D_S)
        return MCTS(
            policy_value_fn=pv.predict,
            world_model_fn=wm.predict_next,
            config=config,
        )

    def test_search_returns_valid_distribution(
        self, mcts: MCTS
    ) -> None:
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        legal_mask[12, 20] = True
        legal_mask[1, 18] = True

        visit_dist = mcts.search(state, legal_mask)
        assert visit_dist.shape == (64, 64)
        assert visit_dist.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_search_only_legal_moves(self, mcts: MCTS) -> None:
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        legal_mask[12, 20] = True

        visit_dist = mcts.search(state, legal_mask)
        illegal_mass = visit_dist[~legal_mask].sum().item()
        assert illegal_mass == pytest.approx(0.0, abs=1e-7)

    def test_more_sims_concentrates_distribution(self) -> None:
        pv = _MockPolicyValue(SMALL_D_S)
        wm = _MockWorldModel(SMALL_D_S)
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        legal_mask[12, 20] = True

        config_few = MCTSConfig(num_simulations=10, c_puct=1.4)
        config_many = MCTSConfig(num_simulations=200, c_puct=1.4)
        mcts_few = MCTS(pv.predict, wm.predict_next, config_few)
        mcts_many = MCTS(pv.predict, wm.predict_next, config_many)

        dist_few = mcts_few.search(state, legal_mask)
        dist_many = mcts_many.search(state, legal_mask)

        entropy_few = -(
            dist_few[dist_few > 0] * dist_few[dist_few > 0].log()
        ).sum()
        entropy_many = -(
            dist_many[dist_many > 0] * dist_many[dist_many > 0].log()
        ).sum()
        assert entropy_many <= entropy_few + 0.5

    def test_visit_counts_nonnegative(self, mcts: MCTS) -> None:
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        visit_dist = mcts.search(state, legal_mask)
        assert (visit_dist >= 0).all()

    def test_search_accepts_logit_policy(self) -> None:
        class _LogitPV:
            def predict(
                self, state: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                logits = torch.full((64, 64), -5.0)
                logits[12, 28] = 5.0
                logits[12, 20] = 1.0
                value = torch.tensor([0.6, 0.2, 0.2])
                return logits, value

        wm = _MockWorldModel(SMALL_D_S)
        mcts = MCTS(
            policy_value_fn=_LogitPV().predict,
            world_model_fn=wm.predict_next,
            config=MCTSConfig(num_simulations=50, c_puct=1.4),
        )
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        legal_mask[12, 20] = True
        dist = mcts.search(state, legal_mask)
        assert dist.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert dist[12, 28].item() > dist[12, 20].item()
