"""The baseline 1D-CNN and its trainer."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

ACC_SCALE = 4.0  # g: the canonical clip range, so inputs are in [-1, 1]
GYRO_SCALE = 8.0  # rad/s


def make_inputs(x: np.ndarray, channels: str) -> np.ndarray:
    """(N, T, 6) windows -> (N, C, T): scaled axes plus magnitude channels.

    channels="imu": acc xyz, gyro xyz, |acc|, |gyro|  (8 channels)
    channels="acc": acc xyz, |acc|                    (4 channels)
    """
    acc = x[:, :, :3] / ACC_SCALE
    acc_mag = np.linalg.norm(x[:, :, :3], axis=2, keepdims=True) / ACC_SCALE
    if channels == "acc":
        parts = [acc, acc_mag]
    else:
        gyro = x[:, :, 3:] / GYRO_SCALE
        gyro_mag = np.linalg.norm(x[:, :, 3:], axis=2, keepdims=True) / GYRO_SCALE
        parts = [acc, gyro, acc_mag, gyro_mag]
    return np.concatenate(parts, axis=2).transpose(0, 2, 1).astype(np.float32)


class FallCNN(nn.Module):
    """3 conv blocks (32/64/128, kernel 5, BN, ReLU, max-pool) -> GAP -> dropout -> 1 logit."""

    def __init__(self, in_ch: int, widths=(32, 64, 128), dropout: float = 0.3) -> None:
        super().__init__()
        layers, c = [], in_ch
        for w in widths:
            layers += [nn.Conv1d(c, w, 5, padding=2), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).mean(dim=2)).squeeze(1)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_proba(model: FallCNN, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(x), batch):
        out.append(torch.sigmoid(model(torch.from_numpy(x[i : i + batch]).to(dev))).cpu().numpy())
    return np.concatenate(out)


def fit(
    xtr: np.ndarray,
    ytr: np.ndarray,
    wtr: np.ndarray,
    xval: np.ndarray,
    yval: np.ndarray,
    seed: int,
    max_epochs: int = 40,
    patience: int = 6,
    batch: int = 256,
    augment=None,
) -> FallCNN:
    """AdamW + class-weighted BCE; early stopping on validation average precision (subject-wise
    validation set). ``wtr`` scales each sample's loss (used to up-weight mined hard negatives).
    ``augment`` (optional callable on a training batch of inputs) generates synthetic variants on
    the fly; the validation set is never augmented."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    model = FallCNN(xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    pos_weight = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), device=dev)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    xt = torch.from_numpy(xtr).to(dev)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    wt = torch.from_numpy(wtr.astype(np.float32)).to(dev)
    best_ap, best_state, bad = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(xt))).to(dev)
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            xb = xt[idx] if augment is None else augment(xt[idx])
            loss = (bce(model(xb), yt[idx]) * wt[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        ap = average_precision_score(yval, predict_proba(model, xval))
        if ap > best_ap:
            best_ap, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def adapt_batchnorm(model: FallCNN, x: np.ndarray, batch: int = 512) -> FallCNN:
    """AdaBN: re-estimate BatchNorm running statistics on unlabelled target windows (no labels,
    no gradient steps); all other weights are untouched."""
    import copy

    m = copy.deepcopy(model)
    dev = next(m.parameters()).device
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.reset_running_stats()
            mod.momentum = None  # cumulative average over all batches
    m.train()
    with torch.no_grad():
        for i in range(0, len(x), batch):
            m(torch.from_numpy(x[i : i + batch]).to(dev))
    m.eval()
    return m


def finetune(
    model: FallCNN,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xval: np.ndarray,
    yval: np.ndarray,
    seed: int,
    freeze_features: bool,
    lr: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 5,
    batch: int = 64,
    augment=None,
) -> FallCNN:
    """Continue training ``model`` on a few target windows. ``freeze_features`` trains only the
    output layer (BatchNorm statistics frozen too). The starting weights are a candidate for the
    best-on-validation checkpoint, so fine-tuning never returns something worse than its start."""
    import copy

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = next(model.parameters()).device
    m = copy.deepcopy(model)
    params = list(m.head.parameters()) if freeze_features else list(m.parameters())
    for p in m.parameters():
        p.requires_grad = False
    for p in params:
        p.requires_grad = True
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    pos_weight = torch.tensor(max((ytr == 0).sum(), 1) / max((ytr == 1).sum(), 1), device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xt = torch.from_numpy(xtr).to(dev)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    best_ap = average_precision_score(yval, predict_proba(m, xval))
    best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    bad = 0
    for _ in range(max_epochs):
        m.train()
        if freeze_features:
            m.features.eval()
        perm = torch.from_numpy(rng.permutation(len(xt))).to(dev)
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            if len(idx) < 2:
                continue
            xb = xt[idx] if augment is None else augment(xt[idx])
            loss = bce(m(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        ap = average_precision_score(yval, predict_proba(m, xval))
        if ap > best_ap:
            best_ap, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m
