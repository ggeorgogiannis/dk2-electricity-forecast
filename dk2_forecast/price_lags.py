from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from typing import NamedTuple

from dk2_forecast.delivery_day import COPENHAGEN, delivery_intervals_utc

PREVIOUS_DAY_VERSION = "previous-local-day-earliest-utc-weekly-fallback-v1"


def local_day(timestamp: datetime) -> date:
    return timestamp.astimezone(COPENHAGEN).date()


def available_price(
    prices: dict[datetime, float],
    timestamp: datetime,
    known_through: date,
) -> float | None:
    # At the issue cutoff the full auction for the issue day's delivery is assumed known.
    return prices.get(timestamp) if local_day(timestamp) <= known_through else None


def lag_price(
    prices: dict[datetime, float], timestamp: datetime, days: int, known_through: date
) -> float | None:
    lag = timestamp.astimezone(UTC) - timedelta(days=days)
    return available_price(prices, lag, known_through)


class PreviousPrice(NamedTuple):
    price: float | None
    source_utc: datetime | None
    weekly_fallback: bool


@lru_cache(maxsize=512)
def _first_slots(day: date) -> dict[time, datetime]:
    slots = {}
    for timestamp in delivery_intervals_utc(day):
        slots.setdefault(timestamp.astimezone(COPENHAGEN).time(), timestamp)
    return slots


def previous_day_price(
    prices: dict[datetime, float], timestamp: datetime, known_through: date
) -> PreviousPrice:
    local = timestamp.astimezone(COPENHAGEN)
    source = _first_slots(local.date() - timedelta(days=1)).get(local.time())
    if source is not None:
        price = available_price(prices, source, known_through)
        if price is not None:
            return PreviousPrice(price, source, False)

    weekly = timestamp.astimezone(UTC) - timedelta(days=7)
    price = available_price(prices, weekly, known_through)
    return PreviousPrice(
        price, weekly if price is not None else None, price is not None
    )
