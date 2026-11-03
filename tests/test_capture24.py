import numpy as np
import pandas as pd
import pytest

from falldet.capture24 import LABELS, UNLABELLED, Capture24

N_SAMPLES = 25 * 60  # one minute at 25 Hz: 29 windows of 4 s with a 2 s stride


@pytest.fixture
def source(tmp_path):
    """Prepared recordings for 10 pool people, 4 held-out people and one excluded person."""
    rows = []
    for i in [*range(1, 11), 101, 102, 103, 104, 50]:
        pid = f"P{i:03d}"
        if i == 50:
            rows.append({"pid": pid, "hours": 24.0, "excluded": "walking mean norm 0.20 g"})
            continue
        acc = np.full((N_SAMPLES, 3), float(i), np.float32)
        acc[:, 0] = np.arange(N_SAMPLES)  # x encodes the sample index, so windows can be located
        lab = np.full(N_SAMPLES, LABELS.index("walking"), np.uint8)
        lab[:40] = UNLABELLED
        np.savez(tmp_path / f"{pid}.npz", acc=acc, lab=lab)
        rows.append({"pid": pid, "hours": 24.0, "excluded": np.nan})
    pd.DataFrame(rows).to_csv(tmp_path / "prepare_log.csv", index=False)
    return Capture24(tmp_path)


def test_split_keeps_held_out_people_apart_and_excludes_unusable_ones(source):
    s = source.split(seed=0)
    assert s.held_out == ["P101", "P102", "P103", "P104"]
    assert len(s.validation) == 2 and len(s.train) == 8
    assert not set(s.train) & set(s.validation)
    assert "P050" not in s.train + s.validation + s.held_out


def test_split_depends_on_seed_but_is_reproducible(source):
    assert source.split(3) == source.split(3)
    assert {tuple(source.split(k).validation) for k in range(10)} != {tuple(source.split(0).validation)}


def test_windows_are_4_s_canonical_windows_on_the_stride_grid(source):
    w = source.windows(["P001"], k=5, seed=0, purpose="train")
    assert w.shape == (5, 100, 3)
    starts = w[:, 0, 0]
    assert np.all(starts % 50 == 0)  # window starts are multiples of the 2 s stride
    assert np.all(np.diff(w[:, :, 0], axis=1) == 1)  # contiguous samples


def test_draws_do_not_depend_on_call_order_or_other_people(source):
    a = source.windows(["P001", "P002"], k=4, seed=1, purpose="train", fold="weda")
    source.windows(["P003"], k=4, seed=1, purpose="validation")  # an unrelated draw in between
    b = source.windows(["P002"], k=4, seed=1, purpose="train", fold="weda")
    np.testing.assert_array_equal(a[4:], b)


def test_purpose_and_fold_give_independent_draws(source):
    base = source.windows(["P001"], k=6, seed=1, purpose="train", fold="weda")
    for other in (source.windows(["P001"], k=6, seed=1, purpose="validation", fold="weda"),
                  source.windows(["P001"], k=6, seed=1, purpose="train", fold="umafall"),
                  source.windows(["P001"], k=6, seed=2, purpose="train", fold="weda")):
        assert not np.array_equal(np.sort(base[:, 0, 0]), np.sort(other[:, 0, 0]))


def test_draw_never_repeats_a_window_and_is_capped_by_the_recording(source):
    w = source.windows(["P001"], k=1000, seed=0, purpose="train")
    assert len(w) == 29 and len(np.unique(w[:, 0, 0])) == 29


def test_labels_and_exposure_per_window(source):
    lab = source.window_labels("P001")
    assert len(lab) == 29
    assert lab[0] == "walking"  # centre of the first window is sample 50, past the unlabelled start
    assert set(source.labels("P001")[:40]) == {"unlabelled"}
    assert source.hours("P001") == pytest.approx(29 * 2 / 3600)
