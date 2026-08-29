"""Phase 3 network: a small dilated residual TCN predicting a RESIDUAL on the demand.

The residual parameterisation is the design decision that matters:

    q = clip(q_required + delta_theta, 0, 1),   q_required = g_inv(clearance - margin)

`q_required` is a closed-form pointwise quantity -- the crouch that would clear the beam if the
body could change envelope instantly. Handing it to the network as an input AND as the base of
the prediction means the network never has to rediscover "low clearance implies crouch", which
Phase 2 spent most of its capacity on. What is left for it to learn is exactly what the
optimiser adds on top of the pointwise demand: anticipation (start early, because the body
lags), smoothing (do not chatter), recovery (come back up), composition (merge two nearby beams
into one crouch, split distant ones), and correction of the surrogate's own error.

Dilations 1/2/4/8 with kernel 3, TWO convolutions per residual block, give a receptive field of
61 samples (1 + sum of 4*d). On a 12 m route sampled at 64 points that is ~11 m -- the whole
route -- so both beams of any composition in the dataset are visible at once, which is the
minimum for merge/split to be representable at all. `receptive_field` computes this rather than
restating it, because a hand-counted figure is exactly the kind of number that drifts.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..learn.route_profile import N_CHANNELS

N_IN = N_CHANNELS + 1          # profile channels plus q_required


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class DuckTCN(nn.Module):
    """Residual TCN. Output is a residual on q_required, not q itself."""

    def __init__(self, ch: int = 24, dilations=(1, 2, 4, 8), n_in: int = N_IN):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 1)
        self.blocks = nn.ModuleList([ResidualBlock(ch, d) for d in dilations])
        self.out = nn.Conv1d(ch, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)          # start as the identity on q_required

    def forward(self, x: torch.Tensor, q_req: torch.Tensor) -> torch.Tensor:
        """(B, N, C), (B, N) -> (B, N). Residual is added to the demand and clipped."""
        h = self.inp(torch.cat([x, q_req[..., None]], dim=-1).transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return torch.clamp(q_req + self.out(h).squeeze(1), 0.0, 1.0)

    @property
    def receptive_field(self) -> int:
        rf = 1
        for b in self.blocks:
            d = b.conv1.dilation[0]
            rf += 2 * 2 * d                      # two kernel-3 convs per block
        return rf

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DemandOnly(nn.Module):
    """Control: emit the pointwise demand with no network at all.

    This is the baseline the residual TCN must beat, and it is a real competitor -- it already
    satisfies the clearance constraint pointwise. What it cannot do is anticipate, smooth or
    compose, so the gap between it and the optimiser is precisely the part the network is being
    asked to learn.
    """

    n_params = 0

    def forward(self, x, q_req):
        return torch.clamp(q_req, 0.0, 1.0)
