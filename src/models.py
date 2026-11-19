"""Step 4 - Forecasting models with a low / middle / high (10 / 50 / 90 %) output.

Output shape of every model: (batch, 24 hours, 2 targets, 3 quantiles) = 72 numbers.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import FEATURES, HORIZON, QUANTILES, TARGETS

N_IN = len(FEATURES)
N_OUT = HORIZON * len(TARGETS) * len(QUANTILES)


class PinballLoss(nn.Module):
    """Quantile loss: punishes 'low' more when too high, 'high' more when too low."""

    def __init__(self, quantiles=QUANTILES):
        super().__init__()
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # pred (B, 24, 2, 3), y (B, 24, 2)
        err = y.unsqueeze(-1) - pred
        return torch.maximum(self.q * err, (self.q - 1) * err).mean()


def _reshape(out: torch.Tensor) -> torch.Tensor:
    out = out.view(-1, HORIZON, len(TARGETS), len(QUANTILES))
    # sort so that low <= middle <= high (no quantile crossing)
    return torch.sort(out, dim=-1).values


class LSTMForecaster(nn.Module):
    """Reads the 168 hours step by step; its final memory summarises the week."""

    def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(N_IN, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, N_OUT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return _reshape(self.head(h[-1]))


class CausalConvBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.norm = nn.BatchNorm1d(ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = nn.functional.pad(x, (self.pad, 0))           # causal: only look back
        return x + self.drop(self.act(self.norm(self.conv(y))))


class CNNForecaster(nn.Module):
    """Dilated 1D convolutions (dilation 1, 2, 4 ... 32) see the whole week (receptive field 190 h)."""

    def __init__(self, filters: int = 64, layers: int = 6, kernel: int = 4, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(N_IN, filters, 1)
        self.blocks = nn.Sequential(*[CausalConvBlock(filters, kernel, 2 ** i, dropout) for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * filters, 128), nn.ReLU(), nn.Linear(128, N_OUT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.inp(x.transpose(1, 2)))       # (B, filters, 168)
        summary = torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1)
        return _reshape(self.head(summary))


MODEL_CLASSES = {"lstm": LSTMForecaster, "cnn": CNNForecaster}


def build_model(name: str) -> nn.Module:
    return MODEL_CLASSES[name]()
