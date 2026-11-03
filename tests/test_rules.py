import numpy as np
import pytest

from falldet.rules import adl_budget, keep_falls, lodo_median, lodo_pooled, lodo_worst


def test_keep_falls_keeps_the_requested_share_of_falls():
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    labels = (rng.random(1000) < 0.3).astype(int)
    for share in (0.5, 0.9, 0.95, 0.98):
        t = keep_falls(scores, labels, share)
        assert (scores[labels == 1] >= t).mean() >= share
        # the next fall score above t would keep fewer than the share
        higher = np.sort(scores[labels == 1])
        higher = higher[higher > t]
        if len(higher):
            assert (scores[labels == 1] >= higher[0]).mean() < share


def test_keep_falls_ignores_adl_scores():
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    labels = np.array([1, 1, 0, 0])
    assert keep_falls(scores, labels, 0.5) == 0.1  # "lower" quantile of the two fall scores


def test_lodo_pooled_weights_datasets_equally_not_by_size():
    big = np.full(1000, 0.9)  # a large dataset of easy falls
    small = np.linspace(0.0, 0.5, 10)  # a small dataset of hard falls
    t = lodo_pooled({"big": big, "small": small}, share=0.95)
    # equal weight: the small dataset is half the mass, so most of its falls must be kept
    assert (small >= t).mean() >= 0.9
    assert lodo_pooled([np.concatenate([big, small])], 0.95) > t  # pooled by size, the small one is ignored


def test_lodo_worst_lets_every_dataset_keep_its_share():
    groups = {"a": np.linspace(0.5, 1, 100), "b": np.linspace(0, 1, 100), "c": np.linspace(0.8, 1, 100)}
    t = lodo_worst(groups, 0.95)
    assert all((g >= t).mean() >= 0.95 for g in groups.values())
    assert lodo_median(groups, 0.95) >= t


def test_adl_budget_respects_the_false_alarm_rate():
    rng = np.random.default_rng(1)
    adl = rng.random(18000)  # 10 h of 2 s strides
    t = adl_budget(adl, false_alarms_per_hour=25)
    assert (adl >= t).sum() / 10 <= 25 + 1


def test_rules_refuse_empty_inputs():
    with pytest.raises(ValueError):
        keep_falls(np.array([0.5]), np.array([0]))
    with pytest.raises(ValueError):
        lodo_pooled({"a": np.array([])})
