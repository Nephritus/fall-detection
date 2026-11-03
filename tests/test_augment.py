import numpy as np
import pytest
import torch

from falldet.augment import TRANSFORMS, Augment


def windows(c: int = 6, n: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(n, 100, c) * 0.5
    x[:, :, 0] -= 0.8
    return x


def only(*names: str, **kw) -> Augment:
    return Augment(p=1.0, enabled=names, **kw)


def test_disabled_augmentation_is_the_identity():
    x = windows()
    torch.testing.assert_close(Augment(p=0.0)(x), x)


@pytest.mark.parametrize("c", [3, 6])
def test_shape_and_layout_are_kept(c):
    x = windows(c)
    assert Augment()(x).shape == x.shape


def test_rotation_keeps_magnitudes_of_both_sensors():
    x = windows()
    out = only("rotation")(x)
    torch.testing.assert_close(out[:, :, :3].norm(dim=2), x[:, :, :3].norm(dim=2), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[:, :, 3:].norm(dim=2), x[:, :, 3:].norm(dim=2), rtol=1e-5, atol=1e-5)


def test_mirror_flips_lateral_accel_and_the_other_two_gyro_axes():
    x = windows()
    out = only("mirror")(x)
    sign = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    torch.testing.assert_close(out, x * sign)


def test_accelerometer_stays_within_the_canonical_clip_range():
    x = windows(3)
    x[:, 50, 0] = 3.9
    out = only("gain", gain=0.5)(x)
    assert out[:, :, :3].abs().max() <= 4.0


def test_every_transform_changes_the_signal():
    x = windows()
    for t in TRANSFORMS:
        assert not torch.allclose(only(t)(x), x), t
