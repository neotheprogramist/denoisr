from torch import Tensor, nn


class ChessConsistencyProjector(nn.Module):
    """SimSiam-style consistency projector.

    Mean-pools the 64 latent tokens and projects to a low-dimensional
    space for computing consistency loss between predicted and actual
    next states. Used with stop-gradient on the target branch to
    prevent latent state collapse (EfficientZero).
    """

    def __init__(self, d_s: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(d_s, proj_dim),
            nn.Mish(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=1)
        out: Tensor = self.projector(pooled)
        return out
