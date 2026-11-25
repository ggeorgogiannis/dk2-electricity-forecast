from datetime import date, timedelta

import pytest

from dk2_forecast.delivery_day import delivery_intervals_utc
from scripts.replay_year import (
    sha,
    training_indices,
    training_origin,
    validate_model_cache,
    validate_rows,
)


def test_year_schedule_extends_the_same_frozen_anchor_backwards():
    anchor = date(2026, 11, 20)
    assert training_origin(date(2026, 1, 1), anchor, 7) == date(2025, 12, 26)
    assert training_origin(date(2026, 1, 2), anchor, 7) == date(2026, 1, 2)
    assert training_origin(date(2026, 11, 24), anchor, 7) == anchor
    with pytest.raises(ValueError, match="positive"):
        training_origin(anchor, anchor, 0)


def test_training_selection_never_reads_origin_or_future_labels():
    origin = date(2026, 3, 29)
    days = [origin + timedelta(days=i) for i in range(-60, 8)]
    selected = training_indices(days, origin)
    labels = [10 if day < origin else 99999 for day in days]
    assert len(selected) == 60
    assert max(days[i] for i in selected) == origin - timedelta(days=1)
    assert all(labels[i] == 10 for i in selected)
    rolling = training_indices(days, origin, weeks=8)
    assert len(rolling) == 56
    assert min(days[i] for i in rolling) == origin - timedelta(weeks=8)


@pytest.mark.parametrize(
    "day,count", [(date(2026, 3, 29), 92), (date(2026, 10, 25), 100)]
)
def test_daily_forecasts_preserve_clock_change_keys(day, count):
    expected = delivery_intervals_utc(day)
    rows = [
        {"delivery_start_utc": stamp.isoformat(), "price_area": "DK2"}
        for stamp in expected
    ]
    assert len(rows) == count
    validate_rows(rows, expected)
    duplicate = rows[:-1] + [rows[0]]
    with pytest.raises(ValueError, match="UTC"):
        validate_rows(duplicate, expected)
    with pytest.raises(ValueError, match="UTC"):
        validate_rows(rows[:-1], expected)
    wrong_area = [row | {"price_area": "DK1"} for row in rows]
    with pytest.raises(ValueError, match="area"):
        validate_rows(wrong_area, expected)


def test_resume_rejects_wrong_model_or_training_identity():
    raw = b"locally generated model fixture"
    origin = date(2026, 1, 2)
    metadata = {
        "sha256": sha(raw),
        "run_id": "run",
        "origin": str(origin),
        "training_sha256": "inputs",
    }
    validate_model_cache(metadata, raw, "run", origin, "inputs")
    for changed in [
        metadata | {"run_id": "other"},
        metadata | {"origin": "2026-01-09"},
        metadata | {"training_sha256": "other"},
        metadata | {"sha256": "bad"},
    ]:
        with pytest.raises(ValueError, match="mismatch"):
            validate_model_cache(changed, raw, "run", origin, "inputs")


def test_year_has_every_delivery_day_and_both_dst_changes():
    start, end = date(2026, 1, 1), date(2026, 11, 26)
    days = [start + timedelta(days=i) for i in range((end - start).days)]
    slots = [stamp for day in days for stamp in delivery_intervals_utc(day)]
    assert len(days) == 329
    assert len(slots) == len(set(slots)) == 31584
