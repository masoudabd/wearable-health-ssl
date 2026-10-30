"""Multi-task self-supervision for inertial windows, after Yuan et al. (npj Digit Med 2024):
three binary pretext tasks (arrow of time, segment permutation, time warping), one head each.
Labels come from the transformation itself, so no human annotation is used."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TASKS = ("aot", "perm", "warp")
N_SEG = 4  # segments shuffled by the permutation task
WARP_STRENGTH = 0.35  # local speeds drawn from [1 - s, 1 + s]
_PERM_INDEX: Dict[Tuple[int, Tuple[int, ...]], torch.Tensor] = {}


def _perm_index(T: int, order: Tuple[int, ...]) -> torch.Tensor:
    """Time indices that put the segments of a length-T window in `order` (segments as torch.tensor_split cuts them)."""
    key = (T, order)
    if key not in _PERM_INDEX:
        segs = torch.tensor_split(torch.arange(T), len(order))
        _PERM_INDEX[key] = torch.cat([segs[j] for j in order])
    return _PERM_INDEX[key]


def _draw_random(labels: Dict[str, torch.Tensor], B: int) -> Tuple[List[Optional[Tuple[int, ...]]], torch.Tensor]:
    """Permutation orders and warp knots, drawn sample by sample in the order of the original loop-based version
    (permutation, rejecting the identity, then warp knots). The RNG stream, and so every CPU result, is unchanged."""
    ident = torch.arange(N_SEG)
    orders: List[Optional[Tuple[int, ...]]] = [None] * B
    knots = torch.empty(B, 6)
    perm, warp = labels["perm"].tolist(), labels["warp"].tolist()
    for i in range(B):
        if perm[i]:
            o = torch.randperm(N_SEG)
            while torch.equal(o, ident):
                o = torch.randperm(N_SEG)
            orders[i] = tuple(o.tolist())
        if warp[i]:
            knots[i] = torch.rand(6) * 2 * WARP_STRENGTH + (1 - WARP_STRENGTH)
    return orders, knots


def make_pretext_batch(xb: torch.Tensor):
    """xb: (B, C, T) on CPU. Each task is applied independently with probability 0.5, in the order arrow of time,
    permutation, time warp (a smooth monotonic re-timing). Random draws are per sample; the transforms run on the
    whole batch. Output is bit-identical to the first, per-sample version (tests/test_pretext.py)."""
    B, C, T = xb.shape
    labels = {t: (torch.rand(B) < 0.5) for t in TASKS}
    orders, knots = _draw_random(labels, B)
    out = xb.clone()
    a, p, w = labels["aot"], labels["perm"], labels["warp"]
    if a.any():
        out[a] = out[a].flip(-1)
    if p.any():
        idx = torch.stack([_perm_index(T, orders[i]) for i in torch.nonzero(p).flatten().tolist()])
        out[p] = out[p].gather(-1, idx.unsqueeze(1).expand(-1, C, -1))
    if w.any():
        speed = F.interpolate(knots[w].unsqueeze(1), size=T, mode="linear", align_corners=True).squeeze(1)
        pos = torch.cumsum(speed, 1)
        pos = (pos - pos[:, :1]) / (pos[:, -1:] - pos[:, :1]) * (T - 1)
        lo = pos.floor().long().clamp(0, T - 2)
        frac = (pos - lo).unsqueeze(1)
        lo = lo.unsqueeze(1).expand(-1, C, -1)
        xw = out[w]
        out[w] = xw.gather(-1, lo) * (1 - frac) + xw.gather(-1, lo + 1) * frac
    return out, {t: v.float() for t, v in labels.items()}


class MultiTaskSSL(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({t: nn.Linear(encoder.emb_dim, 1) for t in TASKS})

    def forward(self, x):
        z = self.encoder(x)
        return {t: h(z).squeeze(-1) for t, h in self.heads.items()}


def pretrain(
    encoder: nn.Module,
    X: np.ndarray,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    log: Optional[Callable[[str], None]] = print,
) -> Dict[str, list]:
    """X: unlabeled windows (N, T, C), already standardized. Trains encoder in place."""
    torch.manual_seed(seed)
    model = MultiTaskSSL(encoder).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.from_numpy(X).float().permute(0, 2, 1)
    hist: Dict[str, list] = {"loss": [], **{f"acc_{t}": [] for t in TASKS}}
    for ep in range(epochs):
        t0 = time.time()
        perm = torch.randperm(len(Xt))
        tot, n, correct = 0.0, 0, {t: 0.0 for t in TASKS}
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            xb, yb = make_pretext_batch(Xt[perm[i : i + batch_size]])
            logits = model(xb.to(device))
            loss = sum(F.binary_cross_entropy_with_logits(logits[t], yb[t].to(device)) for t in TASKS)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
            for t in TASKS:
                correct[t] += ((logits[t].detach().cpu() > 0).float() == yb[t]).float().sum().item()
        hist["loss"].append(tot / n)
        for t in TASKS:
            hist[f"acc_{t}"].append(correct[t] / n)
        if log:
            accs = " ".join(f"{t}={correct[t] / n:.3f}" for t in TASKS)
            log(f"  ssl epoch {ep + 1}/{epochs} loss={tot / n:.4f} {accs} ({time.time() - t0:.0f}s)")
    return hist
