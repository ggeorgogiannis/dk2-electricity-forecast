import numpy as np

from dk2_forecast.delivery_day import delivery_intervals_utc
from dk2_forecast.price_lags import local_day

SCORING = {
    "absolute_error_bands": [25, 50],
    "spike_threshold_eur_per_mwh": 200,
    "extreme_quantile": 0.99,
    "extreme_quantile_method": "higher",
    "delivery_day_reference": "previous_day",
}


def evaluate_comparison(timestamps, actual, predictions, rules, fallback):
    actual = np.asarray(actual, dtype=float)
    observed = np.isfinite(actual)
    covered = {name: np.isfinite(values) for name, values in predictions.items()}
    paired = observed & np.logical_and.reduce(list(covered.values()))
    errors = {name: np.abs(actual - values) for name, values in predictions.items()}
    masks = {
        "all": observed,
        "spike": observed & (actual >= rules["spike_threshold_eur_per_mwh"]),
        "extreme": observed
        & (np.abs(actual) >= rules["extreme_threshold_eur_per_mwh"]),
        "negative": observed & (actual < 0),
    }
    slices = {}
    for label, mask in masks.items():
        common = paired & mask
        count = int(common.sum())
        slices[label] = {
            "observed_intervals": int(mask.sum()),
            "paired_intervals": count,
            "methods": {
                name: {
                    "mae": float(values[common].mean()) if count else None,
                    **{
                        f"within_{band}": float((values[common] <= band).mean())
                        if count
                        else None
                        for band in rules["absolute_error_bands"]
                    },
                }
                for name, values in errors.items()
            },
        }

    by_day = {}
    for index, timestamp in enumerate(timestamps):
        by_day.setdefault(local_day(timestamp), []).append(index)
    wins = {name: {"wins": 0, "ties": 0, "losses": 0} for name in predictions}
    eligible_days = 0
    reference = rules["delivery_day_reference"]
    for day, indices in by_day.items():
        if (
            len(indices) != len(delivery_intervals_utc(day))
            or not paired[indices].all()
        ):
            continue
        eligible_days += 1
        benchmark = errors[reference][indices].mean()
        for name, values in errors.items():
            delta = values[indices].mean() - benchmark
            result = "wins" if delta < -1e-9 else "losses" if delta > 1e-9 else "ties"
            wins[name][result] += 1
    return {
        "expected_intervals": len(actual),
        "coverage": {
            name: {
                "forecast_intervals": int(mask.sum()),
                "share": float(mask.mean()) if len(mask) else None,
            }
            for name, mask in covered.items()
        },
        "previous_day_weekly_fallbacks": sum(bool(used) for used in fallback),
        "slices": slices,
        "delivery_day_results": {
            "reference": reference,
            "complete_paired_days": eligible_days,
            "methods": wins,
        },
    }
