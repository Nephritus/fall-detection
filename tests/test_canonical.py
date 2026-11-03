import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from falldet.canonical import REFERENCE_HANGING_G, align_mounting, mirror_right_to_left


def hanging_samples(rotation: Rotation, mirror: bool, n: int = 500, seed: int = 0) -> np.ndarray:
    """Walking-like samples whose mean is the reference direction seen through an unknown mounting."""
    rng = np.random.default_rng(seed)
    ref = REFERENCE_HANGING_G / np.linalg.norm(REFERENCE_HANGING_G)
    if mirror:
        ref = ref * np.array([1.0, -1.0, 1.0])
    true = rng.normal(0, 0.2, (n, 3)) + ref
    true -= true.mean(axis=0) - ref  # exact mean
    return rotation.inv().apply(true)


@pytest.mark.parametrize("wrist", ["left", "right"])
def test_reference_mean_lands_on_the_left_wrist_reference(wrist):
    raw = hanging_samples(Rotation.from_euler("xyz", [40, -70, 120], degrees=True), mirror=wrist == "right")
    al = align_mounting(raw, wrist)
    acc, _ = al.apply(raw)
    target = REFERENCE_HANGING_G / np.linalg.norm(REFERENCE_HANGING_G)
    np.testing.assert_allclose(acc.mean(axis=0), target, atol=1e-9)


def test_alignment_is_a_rotation_plus_mirror_and_keeps_magnitudes():
    raw = hanging_samples(Rotation.from_euler("z", 90, degrees=True), mirror=True)
    acc, gyro = align_mounting(raw, "right").apply(raw, raw)
    np.testing.assert_allclose(np.linalg.norm(acc, axis=1), np.linalg.norm(raw, axis=1))
    np.testing.assert_allclose(np.linalg.norm(gyro, axis=1), np.linalg.norm(raw, axis=1))


def test_right_wrist_gyro_is_mirrored_as_an_axial_vector():
    raw = hanging_samples(Rotation.identity(), mirror=True)
    al = align_mounting(raw, "right")
    w = np.array([[0.3, -1.2, 0.7]])
    _, gyro = al.apply(raw[:1], w)
    expected = mirror_right_to_left(raw[:1] @ al.rotation.T, w @ al.rotation.T)[1]
    np.testing.assert_allclose(gyro, expected)
    assert not np.allclose(gyro, (w @ al.rotation.T) * np.array([1.0, -1.0, 1.0]))  # not the polar flip


def test_weak_reference_mean_excludes_the_wearer():
    al = align_mounting(np.array([[0.3, 0.2, 0.1], [0.5, 0.0, 0.2]]), "left")
    assert al.excluded and al.norm_g == pytest.approx(np.linalg.norm([0.4, 0.1, 0.15]))
    with pytest.raises(ValueError):
        al.apply(np.zeros((1, 3)))


def test_minimum_norm_is_a_parameter():
    raw = np.tile([[0.6, 0.0, 0.0]], (10, 1))
    assert align_mounting(raw, "left").excluded
    assert not align_mounting(raw, "left", min_norm_g=0.5).excluded


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        align_mounting(np.zeros((0, 3)), "left")
    with pytest.raises(ValueError):
        align_mounting(np.ones((3, 3)), "both")
