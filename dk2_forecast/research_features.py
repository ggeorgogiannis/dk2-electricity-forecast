"""Price curves available at the preparation snapshot, plus calendar effects."""

from datetime import date, timedelta

import numpy as np

from dk2_forecast.delivery_day import COPENHAGEN, delivery_intervals_utc
from dk2_forecast.experiment_models import daily_inputs
from dk2_forecast.price_lags import available_price

PRICE_FEATURE_NAMES = (
    "previous_day",
    "weekly_utc",
    "quarter_sin",
    "quarter_cos",
    *(f"weekday_{day}" for day in range(7)),
    "year_sin",
    "year_cos",
    "utc_offset",
    "recent_mean",
    "recent_std",
    "recent_min",
    "recent_max",
    *(f"day_curve_{q}" for q in range(96)),
    *(f"week_curve_{q}" for q in range(96)),
)


def curve(prices, day: date, known_through: date):
    values = np.full(96, np.nan)
    for timestamp in delivery_intervals_utc(day):
        local = timestamp.astimezone(COPENHAGEN)
        quarter = local.hour * 4 + local.minute // 15
        if np.isfinite(values[quarter]):
            continue
        price = available_price(prices, timestamp, known_through)
        if price is not None:
            values[quarter] = price
    return values


def daily_price_features(history, day: date):
    stamps, _, weekly, previous = daily_inputs(history, day)
    known = day - date.resolution
    day_curve = curve(history.prices, known, known)
    week_curve = curve(history.prices, day - timedelta(days=7), known)
    recent = day_curve[np.isfinite(day_curve)]
    statistics = (
        [recent.mean(), recent.std(), recent.min(), recent.max()]
        if len(recent)
        else [np.nan] * 4
    )
    matrix = []
    seasonal = []
    for timestamp, weekly_price, previous_price in zip(
        stamps, weekly, previous, strict=True
    ):
        local = timestamp.astimezone(COPENHAGEN)
        quarter = local.hour * 4 + local.minute // 15
        phase = 2 * np.pi * quarter / 96
        yearly = 2 * np.pi * (day.timetuple().tm_yday - 1) / 365.2425
        matrix.append(
            [
                previous_price.price,
                weekly_price,
                np.sin(phase),
                np.cos(phase),
                *(int(day.weekday() == value) for value in range(7)),
                np.sin(yearly),
                np.cos(yearly),
                local.utcoffset().total_seconds() / 3600,
                *statistics,
                *day_curve,
                *week_curve,
            ]
        )
        if day.weekday() in (1, 2, 3, 4):
            seasonal.append(previous_price.price)
        else:
            source = week_curve[quarter]
            seasonal.append(source if np.isfinite(source) else weekly_price)
    return (
        stamps,
        np.asarray(matrix, dtype=float),
        np.asarray(seasonal, dtype=float),
        previous,
    )


def all_price_features(history):
    days = sorted({stamp.astimezone(COPENHAGEN).date() for stamp in history.prices})
    timestamps, matrices, seasonal = [], [], []
    for day in days:
        stamps, matrix, baseline, _ = daily_price_features(history, day)
        timestamps.extend(stamps)
        matrices.append(matrix)
        seasonal.extend(baseline)
    return timestamps, np.concatenate(matrices), np.asarray(seasonal, dtype=float)
