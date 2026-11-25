from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

import pytest

from dk2_forecast.delivery_day import delivery_intervals_utc


@pytest.mark.parametrize(
    ("delivery_date", "expected_count", "first_utc", "last_utc"),
    [
        pytest.param(
            date(2026, 3, 29),
            92,
            "2026-03-28T23:00:00+00:00",
            "2026-03-29T21:45:00+00:00",
            id="spring-clock-change",
        ),
        pytest.param(
            date(2026, 10, 5),
            96,
            "2026-10-04T22:00:00+00:00",
            "2026-10-05T21:45:00+00:00",
            id="regular-day",
        ),
        pytest.param(
            date(2026, 10, 25),
            100,
            "2026-10-24T22:00:00+00:00",
            "2026-10-25T22:45:00+00:00",
            id="autumn-clock-change",
        ),
    ],
)
def test_delivery_intervals_utc(
    delivery_date: date, expected_count: int, first_utc: str, last_utc: str
) -> None:
    intervals = delivery_intervals_utc(delivery_date)

    assert len(intervals) == expected_count
    assert intervals[0].isoformat() == first_utc
    assert intervals[-1].isoformat() == last_utc
    assert all(timestamp.utcoffset() == timedelta(0) for timestamp in intervals)
    assert all(
        end - start == timedelta(minutes=15) for start, end in pairwise(intervals)
    )


def test_autumn_clock_change_keeps_both_local_0200_slots() -> None:
    intervals = delivery_intervals_utc(date(2026, 10, 25))
    copenhagen = ZoneInfo("Europe/Copenhagen")
    local_intervals = [timestamp.astimezone(copenhagen) for timestamp in intervals]
    local_two_am = [
        timestamp for timestamp in local_intervals if timestamp.time() == time(2)
    ]

    assert [timestamp.astimezone(UTC) for timestamp in local_two_am] == [
        datetime(2026, 10, 25, 0, 0, tzinfo=UTC),
        datetime(2026, 10, 25, 1, 0, tzinfo=UTC),
    ]
    assert [timestamp.isoformat() for timestamp in local_two_am] == [
        "2026-10-25T02:00:00+02:00",
        "2026-10-25T02:00:00+01:00",
    ]
