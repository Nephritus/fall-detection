"""Train-time augmentation: synthetic variants of training windows, generated on the GPU per batch.

Operates on model inputs ``(B, C, T)`` as built by ``model.make_inputs`` (scaled axes followed by
magnitude channels), so every epoch sees fresh variants of every training window. Validation and
test windows are never augmented. Each transform fires independently per window with probability
``p``:

* rotation: forearm roll about the x axis (the weakly constrained degree of freedom of the wrist
  frame) plus a small tilt about a random axis; applied identically to accel and gyro
* mirror: left/right-hand flip (accel y, gyro x and z negate, as in ``canonical.mirror_right_to_left``)
* gain: per-sensor amplitude scaling
* time warp: stretch about the window centre plus a small shift (linear interpolation)
* jitter: Gaussian noise on every axis
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .canonical import ACC_CLIP_G, FS
from .model import ACC_SCALE, GYRO_SCALE


def _rotation(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation matrices, (B,) angles about (B, 3) unit axes -> (B, 3, 3)."""
    zero = torch.zeros_like(angle)
    k = torch.stack(
        [zero, -axis[:, 2], axis[:, 1], axis[:, 2], zero, -axis[:, 0], -axis[:, 1], axis[:, 0], zero], dim=1
    ).view(-1, 3, 3)
    s, c = torch.sin(angle).view(-1, 1, 1), torch.cos(angle).view(-1, 1, 1)
    return torch.eye(3, device=angle.device) + s * k + (1 - c) * (k @ k)


TRANSFORMS = ("rotation", "mirror", "gain", "stretch", "shift", "jitter")


@dataclass
class Augment:
    p: float = 0.5
    roll_deg: float = 30.0
    tilt_deg: float = 10.0
    gain: float = 0.10
    stretch: float = 0.10
    shift_s: float = 0.20
    jitter_acc_g: float = 0.02
    jitter_gyro: float = 0.05  # rad/s
    enabled: tuple[str, ...] = TRANSFORMS  # subset of TRANSFORMS; the rest are skipped (ablation)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        dev = x.device
        has_gyro = c == 8
        sig = [x[:, 0:3] * ACC_SCALE] + ([x[:, 3:6] * GYRO_SCALE] if has_gyro else [])  # g, rad/s

        def fire() -> torch.Tensor:
            return (torch.rand(b, device=dev) < self.p).float()

        def uni(lo: float, hi: float) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(b, device=dev)

        # rotation: roll about x, then tilt about a random axis (identity where the draw does not fire)
        if "rotation" in self.enabled:
            roll = torch.deg2rad(uni(-self.roll_deg, self.roll_deg)) * fire()
            ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(b, 3)
            axis = F.normalize(torch.randn(b, 3, device=dev), dim=1)
            tilt = torch.deg2rad(uni(-self.tilt_deg, self.tilt_deg)) * fire()
            rot = _rotation(tilt, axis) @ _rotation(roll, ex)
            sig = [torch.bmm(rot, s) for s in sig]

        # mirror
        if "mirror" in self.enabled:
            flip = fire().view(b, 1, 1)
            signs = [torch.tensor([1.0, -1.0, 1.0], device=dev), torch.tensor([-1.0, 1.0, -1.0], device=dev)]
            sig = [s * (1 - flip + flip * sg.view(1, 3, 1)) for s, sg in zip(sig, signs)]

        # gain (independent per sensor)
        if "gain" in self.enabled:
            sig = [s * (1 + uni(-self.gain, self.gain) * fire()).view(b, 1, 1) for s in sig]

        # time warp: source position = centre + (i - centre) * stretch + shift, linear interpolation
        if "stretch" in self.enabled or "shift" in self.enabled:
            scale = torch.ones(b, device=dev)
            shift = torch.zeros(b, device=dev)
            if "stretch" in self.enabled:
                scale = torch.exp(uni(-math.log1p(self.stretch), math.log1p(self.stretch)) * fire())
            if "shift" in self.enabled:
                shift = uni(-self.shift_s * FS, self.shift_s * FS) * fire()
            i = torch.arange(t, device=dev, dtype=x.dtype).view(1, t)
            mid = (t - 1) / 2
            src = mid + (i - mid) * scale.view(b, 1) + shift.view(b, 1)
            grid = torch.stack([2 * src / (t - 1) - 1, torch.zeros_like(src)], dim=2).view(b, 1, t, 2)
            sig = [
                F.grid_sample(s.unsqueeze(2), grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(2)
                for s in sig
            ]

        # jitter
        if "jitter" in self.enabled:
            noise = fire().view(b, 1, 1)
            sig[0] = sig[0] + noise * self.jitter_acc_g * torch.randn_like(sig[0])
            if has_gyro:
                sig[1] = sig[1] + noise * self.jitter_gyro * torch.randn_like(sig[1])

        acc = sig[0].clamp(-ACC_CLIP_G, ACC_CLIP_G) / ACC_SCALE
        parts = [acc]
        mags = [acc.norm(dim=1, keepdim=True)]
        if has_gyro:
            gyro = sig[1] / GYRO_SCALE
            parts.append(gyro)
            mags.append(gyro.norm(dim=1, keepdim=True))
        return torch.cat(parts + mags, dim=1)
