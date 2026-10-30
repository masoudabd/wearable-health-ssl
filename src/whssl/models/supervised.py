"""End-to-end supervised training with checkpoint selection on VALIDATION subjects."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


class Classifier(nn.Module):
    """encoder (B, C, T) -> (B, D), then head (B, D) -> (B, 1); the default head is linear."""

    def __init__(self, encoder: nn.Module, head: nn.Module | None = None):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.emb_dim, 1) if head is None else head

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)


@torch.no_grad()
def predict(model: Classifier, X: np.ndarray, batch_size: int = 1024, device: str = "cpu") -> np.ndarray:
    model.eval().to(device)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).float().permute(0, 2, 1).to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def train_supervised(
    encoder: nn.Module,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    log=print,
    patience: int | None = None,
    optimizer: str = "adamw",
    head: nn.Module | None = None,
    eval_before: bool = False,
) -> Classifier:
    """Keeps the epoch with the best validation AUPRC. patience stops training after that many epochs without a
    better one (None trains all epochs). optimizer "adamw" (weight decay 1e-4) or "adam_amsgrad" (no weight decay,
    the Oxford fine-tuning recipe). eval_before also scores the model before any training, as epoch 0, so a
    warm-started model is kept unchanged when training never beats it on validation. The returned model carries
    best_epoch (0 = untrained) and epochs_run."""
    torch.manual_seed(seed)
    model = Classifier(encoder, head).to(device)
    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer == "adam_amsgrad":
        opt = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    else:
        raise ValueError(f"unknown optimizer {optimizer}")
    pos_weight = torch.tensor((1 - y_fit.mean()) / max(y_fit.mean(), 1e-6), dtype=torch.float32, device=device)
    Xt = torch.from_numpy(X_fit).float().permute(0, 2, 1)
    yt = torch.from_numpy(y_fit).float()
    best, best_state, best_epoch = -1.0, None, 0
    if eval_before:
        best = average_precision_score(y_val, predict(model, X_val, device=device))
        best_state = copy.deepcopy(model.state_dict())
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            b = perm[i : i + batch_size]
            loss = F.binary_cross_entropy_with_logits(
                model(Xt[b].to(device)), yt[b].to(device), pos_weight=pos_weight
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = average_precision_score(y_val, predict(model, X_val, device=device))
        if log:
            log(f"  sup epoch {ep + 1}/{epochs} val_auprc={val:.4f}")
        if val > best:
            best, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), ep + 1
        elif patience is not None and ep + 1 - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.best_epoch, model.epochs_run = best_epoch, ep + 1
    return model
