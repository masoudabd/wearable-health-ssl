"""Small 1D ResNet encoder for tri-axial inertial windows. Input (B, C, T) -> embedding (B, D)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, stride: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.skip = (
            nn.Identity()
            if (c_in == c_out and stride == 1)
            else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, in_ch: int = 3, widths=(32, 64, 128), k: int = 7, emb_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, widths[0], k, padding=k // 2, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
        )
        blocks, c = [], widths[0]
        for w in widths:
            blocks += [ResBlock(c, w, k, stride=2), ResBlock(w, w, k, stride=1)]
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Linear(c, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        h = self.blocks(self.stem(x)).mean(dim=-1)
        return self.proj(h)


@torch.no_grad()
def embed(encoder: nn.Module, X: np.ndarray, batch_size: int = 1024, device: str = "cpu") -> np.ndarray:
    """X: numpy (N, T, C). Returns numpy (N, D)."""
    encoder.eval().to(device)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).float().permute(0, 2, 1).to(device)
        out.append(encoder(xb).cpu().numpy())
    return np.concatenate(out)
