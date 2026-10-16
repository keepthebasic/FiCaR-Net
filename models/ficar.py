# -*- coding: utf-8 -*-
"""FiCaR-Net: Forecast-informed Cascading Risk Network.

Deep fusion (not module stacking) of:
  1) Spectral Error Carrier  — forecast/analysis residual in frequency bands
  2) Dual-Timescale Hydro    — slow/fast states + mass-balance soft constraint
  3) Directed Cascade Router — shared nutrient head on Q_obs vs Q_pred paths
  4) Horizon-conditional risk — per-lead concentration / exceedance
  5) Decision calibration    — post-hoc threshold utilities (no train-time stack)

Compatible hydro I/O with ProcessConstrainedHydro so existing hydro_loss works.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hydro import hydro_loss, mass_balance_penalty
from src.models.nutrient import NutrientRiskHead, nutrient_loss


class SpectralErrorCarrier(nn.Module):
    """Split sequence features into low/high frequency bands along time.

    When analysis forcings are unavailable, treats the input itself as the
    carrier and learns a gated dual-band representation (DTAF / FilterTS idea).
    """

    def __init__(self, n_features: int, low_frac: float = 0.25):
        super().__init__()
        self.low_frac = low_frac
        self.gate = nn.Sequential(
            nn.Linear(n_features * 2, n_features),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(n_features * 2, n_features)

    def forward(
        self,
        forecast: torch.Tensor,
        analysis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # forecast: [B, T, F]
        residual = forecast if analysis is None else forecast - analysis
        # rFFT along time
        spec = torch.fft.rfft(residual, dim=1)
        t = residual.size(1)
        n_freq = spec.size(1)
        cut = max(1, int(n_freq * self.low_frac))
        low = torch.zeros_like(spec)
        high = torch.zeros_like(spec)
        low[:, :cut] = spec[:, :cut]
        high[:, cut:] = spec[:, cut:]
        x_low = torch.fft.irfft(low, n=t, dim=1)
        x_high = torch.fft.irfft(high, n=t, dim=1)
        cat = torch.cat([x_low, x_high], dim=-1)
        g = self.gate(cat)
        fused = self.proj(cat)
        return g * fused + (1.0 - g) * residual


class FiCaRHydro(nn.Module):
    """Dual-timescale process-constrained LSTM with spectral carrier.

    Outputs match ProcessConstrainedHydro for drop-in training.
    """

    def __init__(
        self,
        n_features: int = 3,
        hidden: int = 64,
        horizon: int = 7,
        num_layers: int = 2,
        dropout: float = 0.1,
        probabilistic: bool = True,
        use_spectral: bool = True,
        low_frac: float = 0.25,
    ):
        super().__init__()
        self.horizon = horizon
        self.probabilistic = probabilistic
        self.use_spectral = use_spectral
        self.carrier = SpectralErrorCarrier(n_features, low_frac=low_frac) if use_spectral else None
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # slow / fast state projections (LSTD idea; coupled via softplus heads)
        self.proj_slow = nn.Linear(hidden, hidden)
        self.proj_fast = nn.Linear(hidden, hidden)
        self.head_base = nn.Linear(hidden, horizon)
        self.head_quick = nn.Linear(hidden, horizon)
        # Init near ~1 mm/d so softplus does not start in the dead near-zero zone
        nn.init.zeros_(self.head_base.weight)
        nn.init.zeros_(self.head_quick.weight)
        nn.init.constant_(self.head_base.bias, 0.5)
        nn.init.constant_(self.head_quick.bias, 0.5)
        if probabilistic:
            self.head_log_sigma = nn.Linear(hidden, horizon)

    def forward(
        self,
        x: torch.Tensor,
        analysis: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.carrier is not None:
            x = self.carrier(x, analysis)
        enc, _ = self.encoder(x)
        h = enc[:, -1, :]
        h_slow = torch.tanh(self.proj_slow(h))
        h_fast = torch.tanh(self.proj_fast(h))
        q_base = F.softplus(self.head_base(h_slow))
        q_quick = F.softplus(self.head_quick(h_fast))
        q_total = q_base + q_quick
        out = {
            "q_total": q_total,
            "q_base": q_base,
            "q_quick": q_quick,
            "h_slow": h_slow,
            "h_fast": h_fast,
        }
        if self.probabilistic:
            # sigma from fused state
            h_fuse = 0.5 * (h_slow + h_fast)
            out["log_sigma"] = self.head_log_sigma(h_fuse).clamp(-5, 2)
            out["mu"] = q_total
        return out


class CascadeRouter(nn.Module):
    """Directed cascade: shared nutrient head on obs vs predicted flow paths.

    path='obs'  -> E3-style (truth / analysis flow features)
    path='pred' -> E4-style (forecast-driven flow features)
    """

    def __init__(self, risk_head: NutrientRiskHead | None = None, lookback: int = 14, hidden: int = 64):
        super().__init__()
        self.risk_head = risk_head or NutrientRiskHead(lookback=lookback, hidden=hidden)

    def forward(
        self,
        x_seq: torch.Tensor,
        static: torch.Tensor,
        path: str = "pred",
    ) -> dict[str, torch.Tensor]:
        out = self.risk_head(x_seq, static)
        out["path"] = path  # type: ignore[assignment]
        return out

    def forward_both(
        self,
        x_obs: torch.Tensor,
        x_pred: torch.Tensor,
        static: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "obs": self.forward(x_obs, static, path="obs"),
            "pred": self.forward(x_pred, static, path="pred"),
        }


class FiCaRNet(nn.Module):
    """Full cascade system: FiCaRHydro + CascadeRouter."""

    def __init__(
        self,
        n_force: int = 3,
        hidden: int = 64,
        horizon: int = 7,
        lookback: int = 14,
        use_spectral: bool = True,
    ):
        super().__init__()
        self.hydro = FiCaRHydro(
            n_features=n_force,
            hidden=hidden,
            horizon=horizon,
            use_spectral=use_spectral,
        )
        self.router = CascadeRouter(lookback=lookback, hidden=hidden)

    def forward_hydro(self, x: torch.Tensor, analysis: torch.Tensor | None = None):
        return self.hydro(x, analysis)

    def forward_risk(self, x_seq: torch.Tensor, static: torch.Tensor, path: str = "pred"):
        return self.router(x_seq, static, path=path)


def cascade_attribution_loss(
    out_obs: dict[str, torch.Tensor],
    out_pred: dict[str, torch.Tensor],
    conc_true: torch.Tensor,
    exceed_true: torch.Tensor,
    w_obs: float = 1.0,
    w_pred: float = 1.0,
    pos_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train shared head on both paths; pred path carries cascade signal."""
    loss_obs, s_obs = nutrient_loss(out_obs, conc_true, exceed_true, pos_weight=pos_weight)
    loss_pred, s_pred = nutrient_loss(out_pred, conc_true, exceed_true, pos_weight=pos_weight)
    total = w_obs * loss_obs + w_pred * loss_pred
    stats = {
        "obs": s_obs["total"],
        "pred": s_pred["total"],
        "total": float(total.detach()),
    }
    return total, stats


@torch.no_grad()
def decision_calibrate_threshold(
    probs: torch.Tensor,
    labels: torch.Tensor,
    cost_fn=None,
    grid: int = 51,
) -> dict[str, float]:
    """Stage-5 style post-hoc threshold scan (FlowCP/NCC idea, lightweight).

    Minimizes cost_sensitive-like objective: miss weighted higher than false alarm.
    """
    if cost_fn is None:
        def cost_fn(p, y, thr):  # noqa: E306
            pred = (p >= thr).float()
            miss = ((pred == 0) & (y == 1)).float().mean()
            far = ((pred == 1) & (y == 0)).float().mean()
            return 2.0 * miss + 1.0 * far

    best_thr, best_c = 0.5, float("inf")
    for i in range(grid):
        thr = i / max(grid - 1, 1)
        c = float(cost_fn(probs, labels, thr))
        if c < best_c:
            best_c, best_thr = c, thr
    return {"threshold": best_thr, "cost": best_c}


# re-export for trainers
__all__ = [
    "SpectralErrorCarrier",
    "FiCaRHydro",
    "CascadeRouter",
    "FiCaRNet",
    "cascade_attribution_loss",
    "decision_calibrate_threshold",
    "hydro_loss",
    "mass_balance_penalty",
    "nutrient_loss",
]
