"""The fall model: canonical windows (or a wearer's whole signal) in, fall scores out.

A ``FallModel`` is the baseline 1D-CNN plus the choice of input channels. Callers only ever pass
canonical data (accelerometer in g, gyroscope in rad/s, 25 Hz); how it is encoded for the network
(scaled axes plus magnitude channels) and how training augments it (in signal units, on the GPU) stay
inside this module.

* ``FallModel.train``: AdamW, class-weighted BCE, early stopping on validation average precision
* ``scores`` / ``score_signal``: scores for windows, or for every 4 s window (2 s stride) of a signal
* ``continue_training`` (adaptation) and ``adapt_batchnorm`` (AdaBN)
* ``save`` / ``load``
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from .canonical import FS
from .windows import STRIDE_S, WIN_S

_ACC_SCALE = 4.0  # g: the canonical clip range, so encoded axes are in [-1, 1]
_GYRO_SCALE = 8.0  # rad/s
CHANNELS = ("acc", "imu")
_IN_CHANNELS = {"acc": 4, "imu": 8}  # acc xyz, |acc|  /  acc xyz, gyro xyz, |acc|, |gyro|
_SIGNAL_CHANNELS = {"acc": 3, "imu": 6}


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ------------------------------------------------------------------------------------ encoding
def _encode_np(x: np.ndarray, channels: str) -> np.ndarray:
    """(N, T, >=3|6) canonical windows -> (N, C, T) network inputs."""
    acc = x[:, :, :3] / _ACC_SCALE
    acc_mag = np.linalg.norm(x[:, :, :3], axis=2, keepdims=True) / _ACC_SCALE
    if channels == "acc":
        parts = [acc, acc_mag]
    else:
        gyro = x[:, :, 3:6] / _GYRO_SCALE
        gyro_mag = np.linalg.norm(x[:, :, 3:6], axis=2, keepdims=True) / _GYRO_SCALE
        parts = [acc, gyro, acc_mag, gyro_mag]
    return np.concatenate(parts, axis=2).transpose(0, 2, 1).astype(np.float32)


def _encode_torch(x: torch.Tensor, channels: str) -> torch.Tensor:
    """(B, T, >=3|6) canonical windows on the device -> (B, C, T) network inputs."""
    acc = x[:, :, :3].transpose(1, 2)
    parts, mags = [acc / _ACC_SCALE], [acc.norm(dim=1, keepdim=True) / _ACC_SCALE]
    if channels == "imu":
        gyro = x[:, :, 3:6].transpose(1, 2)
        parts.append(gyro / _GYRO_SCALE)
        mags.append(gyro.norm(dim=1, keepdim=True) / _GYRO_SCALE)
    return torch.cat(parts + mags, dim=1)


def _signal(x: np.ndarray, channels: str) -> np.ndarray:
    need = _SIGNAL_CHANNELS[channels]
    if x.ndim != 3 or x.shape[2] < need:
        raise ValueError(f"'{channels}' needs canonical windows (N, T, >={need}), got {x.shape}")
    return np.ascontiguousarray(x[:, :, :need], dtype=np.float32)


@torch.no_grad()
def _predict(net: FallCNN, xin: np.ndarray, batch: int = 2048) -> np.ndarray:
    net.eval()
    dev = next(net.parameters()).device
    out = [torch.sigmoid(net(torch.from_numpy(xin[i : i + batch]).to(dev))).cpu().numpy()
           for i in range(0, len(xin), batch)]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


class _Batches:
    """Training windows on the device; ``take`` yields network inputs, augmented if requested."""

    def __init__(self, x: np.ndarray, channels: str, augment, dev: torch.device) -> None:
        self.channels, self.augment = channels, augment
        self.data = torch.from_numpy(_signal(x, channels) if augment else _encode_np(x, channels)).to(dev)

    def __len__(self) -> int:
        return len(self.data)

    def take(self, idx: torch.Tensor) -> torch.Tensor:
        if self.augment is None:
            return self.data[idx]
        return _encode_torch(self.augment(self.data[idx]), self.channels)


# --------------------------------------------------------------------------------------- model
class FallModel:
    def __init__(self, net: FallCNN, channels: str) -> None:
        if channels not in CHANNELS:
            raise ValueError(f"channels must be one of {CHANNELS}")
        self.net, self.channels = net, channels

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    # -------------------------------------------------------------------------------- training
    @classmethod
    def train(cls, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *, channels: str,
              seed: int, weights: np.ndarray | None = None, augment=None, max_epochs: int = 40, patience: int = 6,
              batch: int = 256) -> "FallModel":
        """Train from scratch on canonical windows ``x`` (labels ``y``: 1 = fall window). AdamW,
        class-weighted BCE (``weights`` scale each window's loss), early stopping on average precision of
        the validation windows, which are never augmented. ``augment`` maps a batch of canonical windows to
        new ones (``augment.Augment``)."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        dev = device()
        net = FallCNN(_IN_CHANNELS[channels]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        pos_weight = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), device=dev)
        bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        data = _Batches(x, channels, augment, dev)
        yt = torch.from_numpy(y.astype(np.float32)).to(dev)
        wt = torch.from_numpy((np.ones(len(y)) if weights is None else weights).astype(np.float32)).to(dev)
        xval = _encode_np(x_val, channels)
        best_ap, best_state, bad = -1.0, None, 0
        for _ in range(max_epochs):
            net.train()
            perm = torch.from_numpy(rng.permutation(len(data))).to(dev)
            for i in range(0, len(perm), batch):
                idx = perm[i : i + batch]
                loss = (bce(net(data.take(idx)), yt[idx]) * wt[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            ap = average_precision_score(y_val, _predict(net, xval))
            if ap > best_ap:
                best_ap, bad = ap, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return cls(net, channels)

    def continue_training(self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *, seed: int,
                          freeze_features: bool, augment=None, lr: float = 1e-4, max_epochs: int = 30,
                          patience: int = 5, batch: int = 64) -> "FallModel":
        """Adaptation: a copy trained further on ``x``. ``freeze_features`` trains only the output layer
        (BatchNorm statistics frozen too). The starting weights compete for the best-on-validation
        checkpoint, so the result is never worse than the start on the validation windows."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        dev = next(self.net.parameters()).device
        m = copy.deepcopy(self.net)
        params = list(m.head.parameters()) if freeze_features else list(m.parameters())
        for p in m.parameters():
            p.requires_grad = False
        for p in params:
            p.requires_grad = True
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
        pos_weight = torch.tensor(max((y == 0).sum(), 1) / max((y == 1).sum(), 1), device=dev)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        data = _Batches(x, self.channels, augment, dev)
        yt = torch.from_numpy(y.astype(np.float32)).to(dev)
        xval = _encode_np(x_val, self.channels)
        best_ap = average_precision_score(y_val, _predict(m, xval))
        best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        bad = 0
        for _ in range(max_epochs):
            m.train()
            if freeze_features:
                m.features.eval()
            perm = torch.from_numpy(rng.permutation(len(data))).to(dev)
            for i in range(0, len(perm), batch):
                idx = perm[i : i + batch]
                if len(idx) < 2:
                    continue
                loss = bce(m(data.take(idx)), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            ap = average_precision_score(y_val, _predict(m, xval))
            if ap > best_ap:
                best_ap, bad = ap, 0
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        m.load_state_dict(best_state)
        m.eval()
        return FallModel(m, self.channels)

    def adapt_batchnorm(self, x: np.ndarray, batch: int = 512) -> "FallModel":
        """AdaBN: a copy whose BatchNorm statistics are re-estimated on unlabelled windows ``x``."""
        m = copy.deepcopy(self.net)
        dev = next(m.parameters()).device
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.reset_running_stats()
                mod.momentum = None  # cumulative average over all batches
        xin = _encode_np(x, self.channels)
        m.train()
        with torch.no_grad():
            for i in range(0, len(xin), batch):
                m(torch.from_numpy(xin[i : i + batch]).to(dev))
        m.eval()
        return FallModel(m, self.channels)

    # ------------------------------------------------------------------------------- scoring
    def scores(self, x: np.ndarray) -> np.ndarray:
        """Fall probability for each canonical window (N, T, C)."""
        return _predict(self.net, _encode_np(_signal(x, self.channels), self.channels))

    @torch.no_grad()
    def score_signal(self, signal: np.ndarray, chunk: int = 16384) -> np.ndarray:
        """Fall probability for every 4 s window (2 s stride, from the start) of a canonical signal (n, C)."""
        self.net.eval()
        dev = next(self.net.parameters()).device
        win, stride = round(WIN_S * FS), round(STRIDE_S * FS)
        s = torch.from_numpy(np.ascontiguousarray(signal[:, : _SIGNAL_CHANNELS[self.channels]], dtype=np.float32))
        w = s.unfold(0, win, stride)  # (n_windows, C, win), a view
        out = [torch.sigmoid(self.net(_encode_torch(w[i : i + chunk].to(dev).transpose(1, 2), self.channels))).cpu()
               for i in range(0, len(w), chunk)]
        return torch.cat(out).numpy() if out else np.zeros(0, np.float32)

    # ------------------------------------------------------------------------------- storage
    def save(self, path: Path) -> None:
        torch.save(self.net.state_dict(), path)

    @classmethod
    def load(cls, path: Path, channels: str) -> "FallModel":
        net = FallCNN(_IN_CHANNELS[channels]).to(device())
        net.load_state_dict(torch.load(path, map_location=device()))
        net.eval()
        return cls(net, channels)
