from datetime import date, timedelta

import numpy as np
import pytest

from dk2_forecast.delivery_day import delivery_intervals_utc
from dk2_forecast.two_month import (
    gate_weights,
    origin_for,
    prior_mask,
    residual_bounds,
    retain_selection,
    rolling_margin,
    validate_grid,
)


def sample():
    origin = date(2026, 10, 2)
    days = np.asarray([origin + timedelta(days=i // 96 - 60) for i in range(68 * 96)])
    actual = np.full(len(days), 50.0)
    previous = np.full(len(days), 100.0)
    candidate = np.full(len(days), 50.0)
    return origin, days, actual, candidate, previous


def test_gate_and_intervals_ignore_origin_and_future_actuals():
    origin, days, actual, cat, previous = sample()
    expected = gate_weights(days, actual, cat, previous, origin)
    bounds = residual_bounds(days, actual, cat, origin)
    raw = np.column_stack((cat - 5, cat, cat + 5))
    margin = rolling_margin(days, actual, raw, origin)
    actual[days >= origin] = 999999
    assert gate_weights(days, actual, cat, previous, origin) == expected
    np.testing.assert_array_equal(
        residual_bounds(days, actual, cat, origin)[0], bounds[0]
    )
    assert rolling_margin(days, actual, raw, origin) == margin
    assert expected[1]["candidate_weight"] == 1
    assert expected[0]["fallback_half_blend"]
    assert expected[0]["candidate_weight"] == 0.5
    assert expected[1]["prior_intervals"] == 56 * 96


def test_gate_regimes_use_previous_prices_and_can_select_baseline():
    origin, days, actual, cat, previous = sample()
    previous[:] = -10
    actual[:] = -10
    weights = gate_weights(days, actual, cat, previous, origin)
    assert weights[0]["candidate_weight"] == 0
    assert weights[1]["fallback_half_blend"]
    assert weights[0]["last_eligible_delivery"] == str(origin - timedelta(days=1))


def test_calibration_fallback_and_signed_offsets():
    origin, days, actual, cat, previous = sample()
    offsets, count = residual_bounds(days, actual, previous, origin)
    np.testing.assert_array_equal(offsets, [-50, -50, -50])
    assert count == 56 * 96
    raw = np.column_stack((cat + 10, cat + 20, cat + 30))
    assert rolling_margin(days, actual, raw, origin) == (10.0, 28 * 96)
    raw[:] = np.nan
    assert rolling_margin(days, actual, raw, origin) == (0.0, 0)
    with pytest.raises(ValueError, match="Insufficient"):
        residual_bounds(days, actual, np.full(len(days), np.nan), origin)


def test_two_month_grid_contains_fall_clock_change_and_rejects_gaps():
    start, end = date(2026, 9, 26), date(2026, 11, 26)
    stamps = [
        t
        for n in range((end - start).days)
        for t in delivery_intervals_utc(start + timedelta(days=n))
    ]
    assert len(stamps) == 61 * 96 + 4 == 5860
    validate_grid(stamps, start, end)
    with pytest.raises(ValueError, match="grid"):
        validate_grid(stamps[:-1], start, end)
    with pytest.raises(ValueError, match="grid"):
        validate_grid(stamps[:-1] + [stamps[0]], start, end)
    assert origin_for(start, date(2026, 11, 20)) == date(2026, 9, 25)
    assert prior_mask(
        np.asarray([date(2026, 9, 24), date(2026, 9, 25)]), date(2026, 9, 25)
    ).tolist() == [True, False]


def test_residual_pipeline_fits_preprocessing_on_training_only():
    from dk2_forecast.research import fit_model, predict_model

    x = np.asarray([[1.0, np.nan], [2.0, 3.0], [3.0, 5.0]])
    model, _ = fit_model("ridge", x, np.asarray([10.0, 20.0, 30.0]), {"alpha": 100})
    statistics = model[0].statistics_.copy()
    predict_model(model, np.asarray([[99999.0, 99999.0]]))
    np.testing.assert_array_equal(model[0].statistics_, statistics)
    np.testing.assert_array_equal(statistics, [2.0, 4.0])


def test_interrupted_retry_preserves_selection_receipt(tmp_path):
    path = tmp_path / "selection.json"
    scores = [{"alpha": 1000, "mae": 31.86, "intervals": 5376}]
    first = retain_selection(path, scores, 1000)
    raw = path.read_bytes()
    assert retain_selection(path, scores, 1000) == first
    assert path.read_bytes() == raw
    with pytest.raises(ValueError, match="differs"):
        retain_selection(path, scores, 100)
    assert path.read_bytes() == raw
