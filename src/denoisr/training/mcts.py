import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class MCTSConfig:
    num_simulations: int = 100
    c_puct: float = 1.4
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0


class _Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "state")

    def __init__(self, prior: float, state: Tensor | None = None) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[tuple[int, int], "_Node"] = {}
        self.state = state

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        exploration = (
            c_puct
            * self.prior
            * math.sqrt(parent_visits)
            / (1 + self.visit_count)
        )
        return self.q_value + exploration


PolicyValueFn = Callable[[Tensor], tuple[Tensor, Tensor]]
WorldModelFn = Callable[[Tensor, int, int], tuple[Tensor, float]]


class MCTS:
    """Monte Carlo Tree Search in latent space.

    Uses a policy-value function for leaf evaluation and a world model
    for state transitions. Returns a visit-count distribution over
    legal moves after num_simulations rollouts.
    """

    def __init__(
        self,
        policy_value_fn: PolicyValueFn,
        world_model_fn: WorldModelFn,
        config: MCTSConfig,
    ) -> None:
        self._pv = policy_value_fn
        self._wm = world_model_fn
        self._config = config

    def search(self, root_state: Tensor, legal_mask: Tensor) -> Tensor:
        root = _Node(prior=0.0, state=root_state)

        policy, _ = self._pv(root_state)
        policy = policy * legal_mask.float()
        total = policy.sum()
        if total > 0:
            policy = policy / total

        legal_indices = legal_mask.nonzero(as_tuple=False)
        if len(legal_indices) > 0 and self._config.dirichlet_epsilon > 0:
            noise = torch.distributions.Dirichlet(
                torch.full(
                    (len(legal_indices),), self._config.dirichlet_alpha
                )
            ).sample()
            eps = self._config.dirichlet_epsilon
            for idx, (f, t) in enumerate(legal_indices.tolist()):
                orig = policy[f, t].item()
                policy[f, t] = (1 - eps) * orig + eps * noise[idx].item()

        for f, t in legal_indices.tolist():
            root.children[(f, t)] = _Node(prior=policy[f, t].item())

        for _ in range(self._config.num_simulations):
            self._simulate(root)

        visit_dist = torch.zeros(64, 64)
        for (f, t), child in root.children.items():
            visit_dist[f, t] = child.visit_count

        total_visits = visit_dist.sum()
        if total_visits > 0:
            if self._config.temperature == 0:
                best = visit_dist.argmax()
                visit_dist = torch.zeros(64, 64)
                visit_dist.view(-1)[best] = 1.0
            else:
                visit_dist = visit_dist / total_visits

        return visit_dist

    def _simulate(self, root: _Node) -> float:
        node = root
        path: list[_Node] = [node]
        action_taken: tuple[int, int] | None = None

        while node.children and node.visit_count > 0:
            best_action = max(
                node.children.keys(),
                key=lambda a: node.children[a].ucb_score(
                    node.visit_count, self._config.c_puct
                ),
            )
            action_taken = best_action
            node = node.children[best_action]
            path.append(node)

        if node.state is None and action_taken is not None:
            parent = path[-2]
            f, t = action_taken
            state, reward = self._wm(parent.state, f, t)
            node.state = state
        else:
            reward = 0.0

        if node.state is not None:
            policy, value = self._pv(node.state)
            leaf_value = (value[0] - value[2]).item() + reward

            if not node.children:
                for fi in range(64):
                    for ti in range(64):
                        p = policy[fi, ti].item()
                        if p > 1e-8:
                            node.children[(fi, ti)] = _Node(prior=p)
        else:
            leaf_value = 0.0

        for n in path:
            n.visit_count += 1
            n.value_sum += leaf_value

        return leaf_value
