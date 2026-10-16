# -*- coding: utf-8 -*-
"""Nutrient concentration / exceedance risk head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NutrientRiskHead(nn.Module):
    def __init__(self, seq_features: int = 3, lookback: int = 14, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(seq_features, hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),  # conc, exceed_logit
        )

    def forward(self, x_seq: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        enc, _ = self.lstm(x_seq)
        h = enc[:, -1, :]
        inp = torch.cat([h, static], dim=-1)
        raw = self.fc(inp)
        conc = F.softplus(raw[:, 0])
        logit = raw[:, 1]
        return {"conc": conc, "exceed_logit": logit, "exceed_prob": torch.sigmoid(logit)}


def nutrient_loss(
    out: dict[str, torch.Tensor],
    conc_true: torch.Tensor,
    exceed_true: torch.Tensor,
    w_conc: float = 1.0,
    w_ex: float = 1.0,
    pos_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    conc_l = F.mse_loss(out["conc"], conc_true)
    pw = torch.tensor([pos_weight], device=conc_true.device)
    ex_l = F.binary_cross_entropy_with_logits(
        out["exceed_logit"], exceed_true, pos_weight=pw
    )
    total = w_conc * conc_l + w_ex * ex_l
    return total, {"conc": float(conc_l.detach()), "ex": float(ex_l.detach()), "total": float(total.detach())}
