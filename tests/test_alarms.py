import numpy as np
import pytest

from falldet.alarms import FalseAlarms, per_day_ci, sensitivity_at_budget


def one_wearer(times, scores, hours=24.0, **kw):
    return FalseAlarms([np.array(times, float)], [np.array(scores, float)], [hours], **kw)


def test_detections_within_refractory_merge_into_one_event():
    fa = one_wearer([0, 10, 29.9, 30, 61], [1, 1, 1, 1, 1], refractory=30)
    # 0 opens; 10 and 29.9 are inside; 30 is exactly 30 s later and opens; 61 is 31 s after 30
    assert fa.events(0.5).tolist() == [3]


def test_refractory_runs_from_the_event_not_the_last_detection():
    # a detection every 20 s: each is inside the previous detection's window but the event opened at 0
    fa = one_wearer([0, 20, 40, 60], [1, 1, 1, 1], refractory=30)
    assert fa.events(0.5).tolist() == [2]  # 0 and 40


def test_wearers_are_counted_separately():
    fa = FalseAlarms([np.array([0.0]), np.array([1.0])], [np.array([1.0]), np.array([1.0])], [12.0, 12.0])
    assert fa.events(0.5).tolist() == [1, 1]
    assert fa.per_day(0.5) == pytest.approx(2 / 24 * 24)


def test_threshold_selects_firing_detections():
    fa = one_wearer([0, 100, 200], [0.2, 0.6, 0.9])
    assert [fa.events(t)[0] for t in (0.1, 0.5, 0.95)] == [3, 2, 0]


def test_unsorted_input_is_handled():
    fa = one_wearer([60, 0, 20], [1, 1, 1], refractory=30)
    assert fa.events(0.5).tolist() == [2]


def test_window_scores_use_window_ends_and_stride_exposure():
    scores = np.zeros(100)
    scores[[10, 11, 12]] = 1.0  # consecutive windows, 2 s apart: one event
    scores[40] = 1.0  # 56 s after window 12: a second event
    fa = FalseAlarms.from_window_scores([scores])
    assert fa.hours[0] == pytest.approx(100 * 2 / 3600)
    assert fa.events(0.5).tolist() == [2]
    assert fa.per_day(0.5) == pytest.approx(2 / (200 / 3600) * 24)


def test_by_label_events_sum_to_total_and_belong_to_the_opening_detection():
    scores = np.zeros(50)
    scores[[5, 6, 7]] = 1.0
    labels = np.array(["sitting"] * 6 + ["walking"] * 44)
    fa = FalseAlarms.from_window_scores([scores], [labels])
    t = fa.by_label(0.5).set_index("label")
    assert t.events.sum() == fa.events(0.5).sum() == 1
    assert t.loc["sitting", "events"] == 1 and t.loc["walking", "events"] == 0
    assert t.loc["walking", "detections"] == 2
    assert t.loc["sitting", "hours"] == pytest.approx(6 * 2 / 3600)


def test_curve_rises_as_threshold_falls():
    rng = np.random.default_rng(0)
    fa = FalseAlarms.from_window_scores([rng.random(5000), rng.random(5000)])
    thr, per_day = fa.curve()
    assert np.all(np.diff(thr) <= 0)
    assert per_day[-1] >= per_day[0]


def test_sensitivity_at_budget_picks_lowest_threshold_within_budget():
    thr = np.array([0.99, 0.9, 0.5, 0.1])
    per_day = np.array([2, 8, 30, 200.0])
    falls = np.array([0.95, 0.6, 0.3, 0.05])
    assert [sensitivity_at_budget(thr, per_day, falls, b) for b in (1, 10, 50, 300)] == [0.0, 0.25, 0.5, 0.75]


def test_sensitivity_at_budget_ignores_dips_in_the_curve():
    # alarms cannot fall as the threshold drops; a noisy dip below budget must not be used
    thr = np.array([0.9, 0.5, 0.1])
    per_day = np.array([5, 40, 20.0])
    assert sensitivity_at_budget(thr, per_day, np.array([0.7, 0.2]), 25) == 0.0


def test_bootstrap_interval_brackets_the_pooled_rate():
    rng = np.random.default_rng(1)
    events, hours = rng.poisson(20, 40), np.full(40, 24.0)
    lo, hi = per_day_ci(events, hours, np.random.default_rng(0))
    assert lo < events.sum() / hours.sum() * 24 < hi


def test_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError):
        FalseAlarms([np.zeros(3)], [np.zeros(2)], [1.0])
    with pytest.raises(ValueError):
        one_wearer([0], [1]).by_label(0.5)
