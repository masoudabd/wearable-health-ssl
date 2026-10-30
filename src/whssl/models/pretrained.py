"""Frozen public encoders. Each wrapper takes raw tri-axial acceleration in g at its native rate,
resamples to what the encoder was trained on, and returns one embedding per window.

Oxford SSL (harnet), Yuan et al., npj Digital Medicine 2024. Academic-use licence:
cite with "Copyright (c) 2022, University of Oxford"; never use for commercial or employer work.
Trained on wrist Axivity AX3 data (device frame, units g) resampled to 30 Hz.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from ..signal import resample  # noqa: F401  (re-exported; callers used pretrained.resample)

ROOT = Path(__file__).resolve().parents[3]
OXFORD_REPO = Path(os.environ.get("WHSSL_DATA", ROOT / "data")) / "raw" / "oxford_ssl_wearables" / "ssl-wearables-main" / "ssl-wearables-main"

HARNET = {  # name -> (window seconds, embedding dim)
    "harnet5": (5, 512),
    "harnet10": (10, 1024),
    "harnet30": (30, 1024),
}
HARNET_FS = 30


def load_harnet(name: str = "harnet10", pretrained: bool = True) -> torch.nn.Module:
    """Returns the frozen feature extractor. pretrained=False gives the same architecture, random init
    (the 'random' control arm)."""
    m = torch.hub.load(str(OXFORD_REPO), name, source="local", class_num=2, pretrained=pretrained, trust_repo=True)
    fe = m.feature_extractor.eval()
    for p in fe.parameters():
        p.requires_grad_(False)
    return fe


class HarnetEncoder(torch.nn.Module):
    """Pretrained harnet feature extractor with gradients on: (B, 3, T) in g at 30 Hz -> (B, D)."""

    def __init__(self, fe: torch.nn.Module, emb_dim: int):
        super().__init__()
        self.fe = fe
        self.emb_dim = emb_dim

    def forward(self, x):
        return self.fe(x).flatten(1)


def harnet_for_finetuning(name: str = "harnet5"):
    """(encoder, head) for end-to-end fine-tuning as in Yuan et al. 2024: all layers trainable, their evaluation
    head (Linear D->512, ReLU, Linear 512->1 logit) freshly initialised from the torch RNG."""
    m = torch.hub.load(str(OXFORD_REPO), name, source="local", class_num=1, pretrained=True, trust_repo=True)
    for p in m.parameters():
        p.requires_grad_(True)
    return HarnetEncoder(m.feature_extractor, HARNET[name][1]), m.classifier


def linear_head_from_probe(clf, emb_dim: int) -> torch.nn.Linear:
    """The fitted probe (StandardScaler + LogisticRegression pipeline) as one linear layer on the raw embedding, so that
    sigmoid(head(z)) equals the probe's probability: w = coef / scale, b = intercept - sum(coef * mean / scale)."""
    sc, lr = clf.steps[0][1], clf.steps[-1][1]
    w = lr.coef_[0] / sc.scale_
    b = lr.intercept_[0] - float((lr.coef_[0] * sc.mean_ / sc.scale_).sum())
    head = torch.nn.Linear(emb_dim, 1)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(w[None, :], dtype=torch.float32))
        head.bias.fill_(b)
    return head


@torch.no_grad()
def harnet_embed(fe: torch.nn.Module, windows_g_30hz: np.ndarray, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    """windows: (N, T, 3) in g at 30 Hz, T = window seconds * 30. Returns (N, D)."""
    fe.to(device)
    out = []
    for i in range(0, len(windows_g_30hz), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(windows_g_30hz[i : i + batch_size])).float().permute(0, 2, 1)
        out.append(fe(xb.to(device)).flatten(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))
