"""Evaluation of point and interval forecasts on identical UTC delivery keys."""

import numpy as np

from dk2_forecast.delivery_day import COPENHAGEN
from dk2_forecast.experiment_metrics import evaluate_comparison
from dk2_forecast.price_lags import local_day


def point_metrics(stamps, actual, predictions, rules, *, resamples=1000):
    actual = np.asarray(actual, dtype=float)
    values = {
        name: np.asarray(prediction, dtype=float)
        for name, prediction in predictions.items()
    }
    report = evaluate_comparison(stamps, actual, values, rules, [])
    common = np.isfinite(actual) & np.logical_and.reduce(
        [np.isfinite(value) for value in values.values()]
    )
    errors = {name: value - actual for name, value in values.items()}
    absolute = {name: np.abs(value) for name, value in errors.items()}

    def scores(indices):
        mask = common & indices
        reference = absolute["previous_day"][mask].mean() if mask.any() else 0
        return {
            "intervals": int(mask.sum()),
            "methods": {
                name: {
                    "mae": float(np.abs(error[mask]).mean()),
                    "rmse": float(np.sqrt(np.square(error[mask]).mean())),
                    "median_absolute_error": float(np.median(np.abs(error[mask]))),
                    "p90_absolute_error": float(np.quantile(np.abs(error[mask]), 0.9)),
                    "p99_absolute_error": float(np.quantile(np.abs(error[mask]), 0.99)),
                    "mean_signed_error": float(error[mask].mean()),
                    "skill_against_previous_day": 1
                    - float(np.abs(error[mask]).mean() / reference)
                    if reference
                    else None,
                }
                for name, error in errors.items()
            }
            if mask.any()
            else {},
        }

    report["extended"] = scores(np.ones(len(stamps), dtype=bool))
    report["signed_error_convention"] = (
        "prediction minus actual, positive means overprediction"
    )
    for group, keys in {
        "month": [stamp.astimezone(COPENHAGEN).strftime("%Y-%m") for stamp in stamps],
        "hour": [stamp.astimezone(COPENHAGEN).hour for stamp in stamps],
        "weekday": [stamp.astimezone(COPENHAGEN).weekday() for stamp in stamps],
        "weekend": [stamp.astimezone(COPENHAGEN).weekday() >= 5 for stamp in stamps],
    }.items():
        report[group] = {
            str(key): scores(np.asarray([value == key for value in keys]))
            for key in sorted(set(keys))
        }
    days = sorted({local_day(stamp) for stamp in stamps})
    daily = {
        name: np.asarray(
            [
                loss[
                    [
                        local_day(stamp) == day and common[i]
                        for i, stamp in enumerate(stamps)
                    ]
                ].mean()
                for day in days
            ]
        )
        for name, loss in absolute.items()
    }
    report["daily_mean_mae"] = {
        name: float(np.nanmean(loss)) for name, loss in daily.items()
    }
    report["paired_daily_uncertainty"] = {}
    rng = np.random.default_rng(42)
    block_length = 7
    indices = np.concatenate(
        [
            (rng.integers(0, len(days), size=(resamples, 1)) + np.arange(block_length))
            % len(days)
            for _ in range((len(days) + block_length - 1) // block_length)
        ],
        axis=1,
    )[:, : len(days)]
    for name, loss in daily.items():
        if name == "previous_day":
            continue
        if len(days) < 14:
            report["paired_daily_uncertainty"][name] = {
                "status": "insufficient_days",
                "days": len(days),
            }
            continue
        difference = loss - daily["previous_day"]
        boot = np.nanmean(difference[indices], axis=1)
        report["paired_daily_uncertainty"][name] = {
            "mean_daily_mae_difference": float(np.nanmean(difference)),
            "interval_95_percent": list(map(float, np.quantile(boot, [0.025, 0.975]))),
            "seed": 42,
            "resamples": resamples,
            "block_days": block_length,
            "adjusted_for_multiple_comparisons": False,
        }
    report["weighting"] = (
        "Primary metrics weight each quarter-hour equally. Daily summaries and block bootstrap weight Danish days equally, including 92/100-slot days."
    )
    return report


def interval_metrics(stamps, actual, quantiles):
    quantiles = np.asarray(quantiles, dtype=float)
    actual = np.asarray(actual, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(quantiles).all(axis=1)
    raw_crossings = int((np.diff(quantiles, axis=1) < 0).any(axis=1)[mask].sum())
    # Sort only for the reported operational interval, retain raw crossings as a diagnostic.
    ordered = np.sort(quantiles, axis=1)
    losses = {}
    for index, alpha in enumerate((0.1, 0.5, 0.9)):
        error = actual - ordered[:, index]
        losses[str(alpha)] = float(
            np.maximum(alpha * error[mask], (alpha - 1) * error[mask]).mean()
        )
    covered = (actual >= ordered[:, 0]) & (actual <= ordered[:, 2])
    width = ordered[:, 2] - ordered[:, 0]
    interval_score = (
        width
        + 10 * np.maximum(ordered[:, 0] - actual, 0)
        + 10 * np.maximum(actual - ordered[:, 2], 0)
    )
    report = {
        "intervals": int(mask.sum()),
        "pinball_loss": losses,
        "coverage_80": float(covered[mask].mean()),
        "mean_width": float(width[mask].mean()),
        "mean_interval_score_80": float(interval_score[mask].mean()),
        "raw_pinball_loss": {
            str(alpha): float(
                np.maximum(
                    alpha * (actual - quantiles[:, i])[mask],
                    (alpha - 1) * (actual - quantiles[:, i])[mask],
                ).mean()
            )
            for i, alpha in enumerate((0.1, 0.5, 0.9))
        },
        "raw_crossings": raw_crossings,
        "crossing_policy": "Sort the three quantiles per interval",
    }
    groups = {
        "month": [stamp.astimezone(COPENHAGEN).strftime("%Y-%m") for stamp in stamps],
        "hour": [stamp.astimezone(COPENHAGEN).hour for stamp in stamps],
        "regime": [
            "negative" if price < 0 else "high" if price >= 200 else "regular"
            for price in actual
        ],
    }
    for name, keys in groups.items():
        report[name] = {}
        for key in sorted(set(keys)):
            selected = mask & np.asarray([value == key for value in keys])
            report[name][str(key)] = {
                "intervals": int(selected.sum()),
                "coverage_80": float(covered[selected].mean())
                if selected.any()
                else None,
                "mean_width": float(width[selected].mean()) if selected.any() else None,
                "mean_interval_score_80": float(interval_score[selected].mean())
                if selected.any()
                else None,
            }
    return report
