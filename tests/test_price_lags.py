from datetime import UTC, date, datetime, timedelta

import pytest

from dk2_forecast.delivery_day import delivery_intervals_utc
from dk2_forecast.price_lags import lag_price, local_day, previous_day_price


def test_spring_previous_day_missing_hour_falls_back_to_weekly():
    spring = date(2026, 3, 29)
    target = date(2026, 3, 30)
    prior = delivery_intervals_utc(spring)
    timestamps = delivery_intervals_utc(target)
    prices = {stamp: 10.0 for stamp in prior}
    prices.update({stamp - timedelta(days=7): -25.0 for stamp in timestamps})
    matches = [previous_day_price(prices, stamp, spring) for stamp in timestamps]
    assert [index for index, match in enumerate(matches) if match.weekly_fallback] == [
        8,
        9,
        10,
        11,
    ]
    assert matches[8] == (-25.0, timestamps[8] - timedelta(days=7), True)
    assert matches[12] == (10.0, datetime(2026, 3, 29, 1, tzinfo=UTC), False)
    del prices[timestamps[8] - timedelta(days=7)]
    assert previous_day_price(prices, timestamps[8], spring) == (None, None, False)


def test_autumn_previous_day_uses_earlier_occurrence():
    earlier = datetime(2026, 10, 25, 0, 15, tzinfo=UTC)
    later = earlier + timedelta(hours=1)
    stamp = datetime(2026, 10, 26, 1, 15, tzinfo=UTC)
    weekly = stamp - timedelta(days=7)
    prices = {earlier: -10.0, later: 20.0, weekly: 30.0}
    assert previous_day_price(prices, stamp, date(2026, 10, 25)) == (
        -10.0,
        earlier,
        False,
    )
    # The auction from the previous local date is unavailable at this earlier cutoff.
    assert previous_day_price(prices, stamp, date(2026, 10, 24)) == (30.0, weekly, True)


@pytest.mark.parametrize("offset", [4, 3, 2, 1])
def test_four_late_autumn_intervals_never_read_target_day(offset):
    day = date(2026, 10, 25)
    stamp = delivery_intervals_utc(day)[-offset]
    unsafe = stamp - timedelta(hours=24)
    prior = stamp - timedelta(hours=25)
    assert local_day(unsafe) == day
    prices = {unsafe: 9999.0, prior: -12.0}
    assert lag_price(prices, stamp, 1, date(2026, 10, 24)) is None
    assert previous_day_price(prices, stamp, date(2026, 10, 24)) == (
        -12.0,
        prior,
        False,
    )
