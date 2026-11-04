import numpy as np
import pytest

from falldet.canonical import FS
from falldet.stillness import post_motion, stays_still


def signal(seconds, g=1.0):
    acc = np.zeros((int(seconds * FS), 3))
    acc[:, 2] = g
    return acc


def test_motion_is_the_per_second_spread_of_the_acceleration_vector():
    acc = signal(6)
    acc[2 * FS:3 * FS:2, 2] += 0.1  # second 2: z alternates around its mean
    acc[2 * FS + 1:3 * FS:2, 2] -= 0.1
    acc[4 * FS:5 * FS, 0] = np.linspace(-0.2, 0.2, FS)  # second 4: the wrist rotates, |a| barely changes
    m = post_motion(acc, np.array([0.0, 2.0]), 4)
    assert m[0, [0, 1, 3]] == pytest.approx([0, 0, 0], abs=1e-6)
    assert m[0, 2] == pytest.approx(0.1, rel=0.01)
    assert m[1, 2] == pytest.approx(np.linspace(-0.2, 0.2, FS).std(), rel=1e-4)


def test_a_still_wrist_with_a_calibration_offset_is_still():
    m = post_motion(signal(5, g=1.06), np.array([0.0]), 4)
    assert stays_still(m, 4, 0.01).tolist() == [True]


def test_motion_past_the_signal_end_is_unknown():
    m = post_motion(signal(5), np.array([3.0]), 4)
    assert m[0, :2] == pytest.approx([0, 0], abs=1e-6)
    assert np.isnan(m[0, 2:]).all()


def test_strict_rule_needs_every_second_still_mean_rule_the_average():
    m = np.array([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert not stays_still(m, 10, 0.05, "strict")[0]
    assert stays_still(m, 10, 0.05, "mean")[0]  # 0.03 on average
    assert stays_still(m, 10, 0.05, "most")[0]  # 9 of 10 seconds still
    assert stays_still(m, 2, 0.05, "strict")[0]  # the burst comes after the wait


def test_no_wait_always_passes_and_unknown_motion_never_does():
    m = np.array([[0.5, np.nan], [0.0, 0.0]])
    assert stays_still(m, 0, 0.05).tolist() == [True, True]
    assert stays_still(np.array([[0.0, np.nan]]), 2, 0.05).tolist() == [False]


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        stays_still(np.zeros((1, 3)), 2, 0.05, "loose")
