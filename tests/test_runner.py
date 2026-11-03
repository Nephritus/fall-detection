import numpy as np
import pandas as pd
import pytest

from falldet import runner
from falldet.model import FallModel
from falldet.runner import FallData, Results, cv_folds, final_fold, lodo_folds


@pytest.fixture(scope="module")
def data():
    """3 datasets x 4 subjects x 30 windows; accel z carries the dataset index so training sets are traceable."""
    rng = np.random.default_rng(0)
    rows, xs = [], []
    for di, ds in enumerate(["alpha", "beta", "gamma"]):
        for s in range(4):
            for w in range(30):
                fall = w % 5 == 0
                x = rng.normal(0, 0.05, (100, 6)).astype(np.float32)
                x[:, 0] -= 1.0
                x[:, 2] = di
                if fall:
                    x[50, 0] += 3.0
                xs.append(x)
                rows.append({"dataset": ds, "subject": f"{ds}:S{s}", "label": int(fall), "rec": len(rows)})
    meta = pd.DataFrame(rows)
    return FallData(np.stack(xs), meta, meta.label.to_numpy())


def subjects(data, idx):
    return set(data.meta.subject.iloc[idx])


def test_lodo_folds_hold_out_whole_datasets(data):
    for f in lodo_folds(data, seed=0):
        assert set(data.meta.dataset.iloc[f.test]) == {f.name}
        assert f.name not in f.training_datasets
        assert not subjects(data, f.train) & subjects(data, f.validation)
        assert len(f.train) + len(f.validation) + len(f.test) == len(data.y)


def test_cv_and_final_folds_are_subject_wise(data):
    for f in cv_folds(data, seed=1, k=3):
        assert not subjects(data, f.train) & subjects(data, f.test)
        assert not subjects(data, f.train) & subjects(data, f.validation)
    f = final_fold(data, seed=0)
    assert len(f.test) == 0 and len(f.train) + len(f.validation) == len(data.y)


def test_folds_are_reproducible(data):
    a, b = lodo_folds(data, 3), lodo_folds(data, 3)
    assert all(np.array_equal(x.validation, y.validation) for x, y in zip(a, b))


def test_with_training_only_narrows(data):
    f = lodo_folds(data, 0)[0]
    assert len(f.with_training(f.train[:10]).train) == 10
    with pytest.raises(ValueError):
        f.with_training(f.test[:3])


@pytest.fixture
def spy(monkeypatch):
    """Record which datasets (accel z) each training run saw, and train for one epoch only."""
    seen = []
    real = FallModel.train.__func__

    def train(cls, x, y, xv, yv, **kw):
        seen.append(sorted({int(round(v)) for v in np.unique(x[:, 0, 2])}))
        return real(cls, x, y, xv, yv, **{**kw, "max_epochs": 1})

    monkeypatch.setattr(runner.FallModel, "train", classmethod(train))
    return seen


def test_train_model_never_sees_the_held_out_dataset(data, spy):
    fold = lodo_folds(data, 0)[1]  # beta held out
    trained = fold.train_model("acc")
    assert spy == [[0, 2]]
    val_falls = trained.model.scores(data.windows(fold.validation, "acc"))[data.y[fold.validation] == 1]
    assert (val_falls >= trained.threshold).mean() >= 0.95


def test_extra_windows_join_or_replace_the_training_set(data, spy):
    fold = lodo_folds(data, 0)[0]
    extra = np.full((8, 100, 3), 7.0, np.float32), np.zeros(8, np.int64)
    fold.train_model("acc", extra=extra)
    fold.train_model("acc", extra=extra, own_windows=False)
    assert spy == [[1, 2, 7], [7]]


def test_out_of_dataset_scores_come_from_models_that_never_saw_the_dataset(data, spy):
    fold = lodo_folds(data, 0)[0]  # alpha held out; training datasets beta (1) and gamma (2)
    scores = fold.out_of_dataset_fall_scores("acc")
    assert set(scores) == {"beta", "gamma"}
    assert spy == [[2], [1]]  # beta scored by a gamma-only model, gamma by a beta-only model
    assert len(scores["beta"]) == int(((data.meta.dataset == "beta") & (data.y == 1)).sum())


def test_results_round_trip_and_resume(tmp_path):
    r = Results(tmp_path)
    assert not r.has("acc", 0, "weda")
    r.save("acc", 0, "weda", [{"auroc": 0.9}], {"p": np.arange(3.0)})
    r.save("imu", 1, "cv-fold2", [{"auroc": 0.8}, {"auroc": 0.7}])
    assert r.has("acc", 0, "weda") and not r.has("acc", 1, "weda")
    t = r.table().sort_values(["condition", "auroc"])
    assert t[["condition", "seed", "fold"]].values.tolist() == [["acc", 0, "weda"], ["imu", 1, "cv-fold2"], ["imu", 1, "cv-fold2"]]
    np.testing.assert_array_equal(r.arrays("acc", 0, "weda")["p"], np.arange(3.0))


def test_results_ignore_an_unfinished_save(tmp_path):
    r = Results(tmp_path)
    (tmp_path / "acc" / "seed0").mkdir(parents=True)
    (tmp_path / "acc" / "seed0" / "weda.tmp.csv").write_text("auroc\n0.5\n")
    assert not r.has("acc", 0, "weda") and r.table().empty
