"""The smallest model that fits the representation: a dilated 1-D CNN over route position.

The task is a local, translation-equivariant map along a route -- what the body should do at
position s depends on the geometry within a couple of metres of s, and on nothing far away.
That is a convolution, and saying so is not a modelling claim worth defending in a paper; it is
just the right shape. An MLP over the flattened 64x7 profile would have ~30x the parameters and
would have to relearn translation equivariance from data; a Transformer would add global
attention the task does not use.

Receptive field is the one design number that matters. Dilations 1, 2, 4 with kernel 5 give
5 + 8 + 16 = 29 samples, about 3.6 m of route at the sampling used -- comfortably wider than
the 0.72 m anticipation lead the label carries, so the network can see a beam coming before it
has to start ducking.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .route_profile import N_CHANNELS


class DuckCNN(nn.Module):
    def __init__(self, ch: int = 16, n_in: int = N_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=8, dilation=4), nn.ReLU(),
            nn.Conv1d(ch, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, C) -> (B, N). Output is a dip in metres and cannot be negative."""
        y = self.net(x.transpose(1, 2)).squeeze(1)
        return torch.relu(y)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DuckMLP(nn.Module):
    """Baseline for the architecture choice: same task, no translation equivariance."""

    def __init__(self, n_samples: int = 64, n_in: int = N_CHANNELS, hidden: int = 64):
        super().__init__()
        self.n = n_samples
        self.net = nn.Sequential(
            nn.Linear(n_samples * n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_samples),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x.reshape(x.shape[0], -1)))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


MODELS = {"cnn": DuckCNN, "mlp": DuckMLP}
