# -*- coding: utf-8 -*-
"""Process-constrained hydrologic sequence model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProcessConstrainedHydro(nn.Module):
    """LSTM backbone with baseflow/quickflow split and optional Gaussian head.

    Outputs over `horizon`:
      - q_total (softplus sum of base+quick)
      - q_base, q_quick
      - if probabilistic: mu, log_sigma for total flow
    """

    def __init__(
        self,
        n_features: int = 3,
        hidden: int = 64,
        horizon: int = 7,
        num_layers: int = 2,
        dropout: float = 0.1,
        probabilistic: bool = True,
    ):
        super().__init__()
        self.horizon = horizon
        self.probabilistic = probabilistic
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_base = nn.Linear(hidden, horizon)
        self.head_quick = nn.Linear(hidden, horizon)
        nn.init.zeros_(self.head_base.weight)
        nn.init.zeros_(self.head_quick.weight)
        nn.init.constant_(self.head_base.bias, 0.5)
        nn.init.constant_(self.head_quick.bias, 0.5)
        if probabilistic:
            self.head_log_sigma = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, T, F]
        enc, _ = self.encoder(x)
        h = enc[:, -1, :]
        q_base = F.softplus(self.head_base(h))
        q_quick = F.softplus(self.head_quick(h))
        q_total = q_base + q_quick
        out = {"q_total": q_total, "q_base": q_base, "q_quick": q_quick}
        if self.probabilistic:
            out["log_sigma"] = self.head_log_sigma(h).clamp(-5, 2)
            out["mu"] = q_total
        return out


def mass_balance_penalty(
    q_pred: torch.Tensor,
    precip_future: torch.Tensor,
    runoff_ratio_max: float = 1.25,
) -> torch.Tensor:
    """Soft constraint: cumulative runoff should not greatly exceed cumulative precip."""
    # q_pred, precip_future: [B, H]
    sum_q = q_pred.sum(dim=-1)
    sum_p = precip_future.sum(dim=-1).clamp_min(1e-3)
    ratio = sum_q / sum_p
    return F.relu(ratio - runoff_ratio_max).mean()


def hydro_loss(
    out: dict[str, torch.Tensor],
    y: torch.Tensor,
    precip_future: torch.Tensor,
    w_nll: float = 1.0,
    w_mb: float = 0.05,
    w_split: float = 0.01,
    w_bias: float = 0.0,
    loss_type: str = "nll",
) -> tuple[torch.Tensor, dict[str, float]]:
    q = out["q_total"]
    lt = loss_type.lower()
    if lt in {"log_mse", "logmse"}:
        point = F.mse_loss(torch.log1p(q.clamp_min(0)), torch.log1p(y.clamp_min(0)))
    elif lt in {"mse", "l2"}:
        point = F.mse_loss(q, y)
    elif "log_sigma" in out:
        var = torch.exp(2 * out["log_sigma"])
        nll = 0.5 * (torch.log(var) + (y - out["mu"]) ** 2 / var).mean()
        point = nll
    else:
        point = F.mse_loss(q, y)
    mb = mass_balance_penalty(q, precip_future) if w_mb > 0 else q.new_zeros(())
    split = (
        F.relu(0.05 - out["q_base"].mean()) + F.relu(0.05 - out["q_quick"].mean())
        if w_split > 0
        else q.new_zeros(())
    )
    # Pull batch-mean prediction toward observed mean (fights systematic underprediction)
    bias = (q.mean() - y.mean()).pow(2) if w_bias > 0 else q.new_zeros(())
    total = w_nll * point + w_mb * mb + w_split * split + w_bias * bias
    stats = {
        "point": float(point.detach()),
        "mb": float(mb.detach()) if torch.is_tensor(mb) else float(mb),
        "split": float(split.detach()) if torch.is_tensor(split) else float(split),
        "bias": float(bias.detach()) if torch.is_tensor(bias) else float(bias),
        "total": float(total.detach()),
    }
    return total, stats
