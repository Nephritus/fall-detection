import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from falldet.augment import Augment
from falldet.model import FallModel


def toy(n: int, seed: int, gyro: bool = False):
    """Canonical 4 s windows: ADL is a hanging arm with noise, a fall adds a 3 g spike."""
    rng = np.random.default_rng(seed)
    c = 6 if gyro else 3
    x = rng.normal(0, 0.05, (n, 100, c)).astype(np.float32)
    x[:, :, 0] -= 1.0
    y = (np.arange(n) % 4 == 0).astype(np.int64)
    pos = rng.integers(20, 80, n)
    x[y == 1, pos[y == 1], 0] += 3.0
    if gyro:
        x[y == 1, pos[y == 1], 3] += 4.0
    return x, y


@pytest.fixture(scope="module")
def trained():
    x, y = toy(400, 0)
    xv, yv = toy(120, 1)
    return FallModel.train(x, y, xv, yv, channels="acc", seed=0, max_epochs=4, patience=4), (xv, yv)


def test_training_learns_a_separable_problem(trained):
    model, (xv, yv) = trained
    assert average_precision_score(yv, model.scores(xv)) > 0.9


def test_accel_model_ignores_extra_gyro_channels(trained):
    model, (xv, _) = trained
    x6 = np.concatenate([xv, np.random.default_rng(2).normal(size=xv.shape).astype(np.float32)], axis=2)
    np.testing.assert_allclose(model.scores(x6), model.scores(xv))


def test_signal_scores_match_window_scores(trained):
    model, _ = trained
    signal = np.concatenate([w for w in toy(6, 3)[0]])  # 24 s of signal
    starts = np.arange(0, len(signal) - 100 + 1, 50)
    windows = np.stack([signal[s : s + 100] for s in starts])
    np.testing.assert_allclose(model.score_signal(signal), model.scores(windows), atol=1e-5)


def test_save_and_load_give_the_same_scores(trained, tmp_path):
    model, (xv, _) = trained
    model.save(tmp_path / "m.pt")
    np.testing.assert_array_equal(FallModel.load(tmp_path / "m.pt", "acc").scores(xv), model.scores(xv))


def test_continue_training_is_never_worse_on_validation(trained):
    model, (xv, yv) = trained
    x, y = toy(40, 4)
    before = average_precision_score(yv, model.scores(xv))
    after = model.continue_training(x, y, xv, yv, seed=0, freeze_features=False, max_epochs=2)
    assert average_precision_score(yv, after.scores(xv)) >= before


def test_imu_model_trains_with_augmentation():
    x, y = toy(200, 5, gyro=True)
    xv, yv = toy(80, 6, gyro=True)
    m = FallModel.train(x, y, xv, yv, channels="imu", seed=0, augment=Augment(), max_epochs=2)
    assert m.scores(xv).shape == (80,)
    with pytest.raises(ValueError):
        m.scores(x[:, :, :3])  # an IMU model needs the gyroscope
