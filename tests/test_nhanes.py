import numpy as np
import pytest

from falldet.canonical import FS, to_grid
from falldet.nhanes import gait_epochs, resample, wear_seconds, worn_windows


def test_blockwise_resampling_matches_resampling_in_one_go():
    fs = 80
    t = np.arange(int(2.5 * 3600 * fs)) / fs  # crosses two hour boundaries
    x = np.stack([np.sin(2 * np.pi * 1.3 * t), np.cos(2 * np.pi * 0.2 * t), 1 + 0.1 * np.sin(2 * np.pi * 3 * t)], axis=1)
    whole = to_grid(t, x, fs, np.arange(int(t[-1] * FS) + 1) / FS)
    blocks = resample(x.astype(np.float32), fs)
    assert blocks.shape == whole.shape
    assert np.abs(blocks - whole).max() < 1e-3


def _signal(seconds, step_hz=None, amp=0.3):
    t = np.arange(int(seconds * FS)) / FS
    acc = np.zeros((len(t), 3))
    acc[:, 2] = 1.0
    if step_hz:
        acc[:, 2] += amp * np.sin(2 * np.pi * step_hz * t)
    return acc


def test_gait_needs_a_step_rhythm_and_enough_movement():
    assert gait_epochs(_signal(60, step_hz=1.8)).all()
    assert not gait_epochs(_signal(60)).any()  # still
    assert not gait_epochs(_signal(60, step_hz=0.2)).any()  # too slow for steps
    assert not gait_epochs(_signal(60, step_hz=1.8, amp=0.02)).any()  # too little movement


def test_gait_does_not_depend_on_how_the_device_is_rotated():
    acc = _signal(60, step_hz=1.8)
    c, s = np.cos(1.0), np.sin(1.0)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    assert (gait_epochs(acc @ rot.T) == gait_epochs(acc)).all()


def test_an_hour_without_movement_is_not_worn_and_movement_is():
    rng = np.random.default_rng(0)
    still = _signal(2 * 3600)
    moving = _signal(2 * 3600) + rng.normal(0, 0.1, (2 * 3600 * FS, 3))
    worn = wear_seconds(np.concatenate([moving, still, moving]))
    assert worn[: 2 * 3600 - 3600].all()
    assert not worn[2 * 3600 + 900: 4 * 3600 - 900].any()
    assert worn[-(2 * 3600 - 3600):].all()


def test_window_is_worn_only_if_every_second_is():
    wear = np.ones(20, bool)
    wear[7] = False
    # windows start every 2 s and last 4 s: 0-4, 2-6, 4-8, 6-10, 8-12, ...
    assert worn_windows(wear, 9).tolist() == [True, True, False, False, True, True, True, True, True]


def test_windows_running_past_the_recorded_seconds_are_not_worn():
    assert worn_windows(np.ones(5, bool), 3).tolist() == [True, False, False]
