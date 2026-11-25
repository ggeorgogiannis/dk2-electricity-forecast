"""Prior-only price experiments, separate from immutable live execution."""

from datetime import UTC, datetime, timedelta
import json

import numpy as np


def retain_selection(path, scores, alpha):
    """An interrupted retry preserves the first selection receipt and timestamp."""
    if path.exists():
        saved = json.loads(path.read_bytes())
        if saved["inner_scores"] != scores or saved["selected_alpha"] != alpha:
            raise ValueError("Existing inner selection differs from recomputation")
        datetime.fromisoformat(saved["frozen_before_outer_scoring_at_utc"])
        return saved
    from dk2_forecast.storage import publish_bytes

    saved = {
        "inner_scores": scores,
        "selected_alpha": alpha,
        "frozen_before_outer_scoring_at_utc": datetime.now(UTC).isoformat(),
    }
    publish_bytes(path, (json.dumps(saved, indent=2, sort_keys=True) + "\n").encode())
    return saved


def origin_for(day, anchor, frequency=7):
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    return day - timedelta(days=(day - anchor).days % frequency)


def prior_mask(days, origin, window=56):
    days = np.asarray(days)
    return (days < origin) & (days >= origin - timedelta(days=window))


def gate_weights(days, actual, candidate, previous, origin, minimum=192):
    """Learn weights from past OOF errors; regime uses available previous prices."""
    days, actual, candidate, previous = map(
        np.asarray, (days, actual, candidate, previous)
    )
    regimes = np.digitize(previous, [0, 200])
    prior = prior_mask(days, origin)
    records = []
    for regime in range(3):
        mask = prior & (regimes == regime)
        mask &= np.isfinite(actual) & np.isfinite(candidate) & np.isfinite(previous)
        weight = 0.5
        if mask.sum() >= minimum:
            weight = min(
                (0, 0.25, 0.5, 0.75, 1),
                key=lambda w: np.abs(
                    actual[mask] - w * candidate[mask] - (1 - w) * previous[mask]
                ).mean(),
            )
        records.append(
            {
                "origin": str(origin),
                "previous_price_regime": regime,
                "candidate_weight": weight,
                "prior_intervals": int(mask.sum()),
                "fallback_half_blend": bool(mask.sum() < minimum),
                "last_eligible_delivery": str(max(days[mask])) if mask.any() else None,
            }
        )
    return records


def residual_bounds(days, actual, prediction, origin):
    mask = prior_mask(days, origin)
    residual = np.asarray(actual)[mask] - np.asarray(prediction)[mask]
    residual = residual[np.isfinite(residual)]
    if len(residual) < 14 * 92:
        raise ValueError("Insufficient prior residuals for interval benchmark")
    return np.quantile(residual, [0.1, 0.5, 0.9]), int(len(residual))


def rolling_margin(days, actual, raw, origin):
    mask = prior_mask(days, origin, window=28)
    ordered = np.sort(raw, axis=1)
    mask &= np.isfinite(ordered).all(axis=1) & np.isfinite(actual)
    if mask.sum() < 14 * 92:
        return 0.0, int(mask.sum())
    scores = np.maximum(
        ordered[mask, 0] - actual[mask], actual[mask] - ordered[mask, 2]
    )
    return max(0.0, float(np.quantile(scores, 0.8, method="higher"))), int(mask.sum())


def validate_grid(stamps, start, end):
    from dk2_forecast.delivery_day import delivery_intervals_utc

    expected = [
        t
        for i in range((end - start).days)
        for t in delivery_intervals_utc(start + timedelta(days=i))
    ]
    if stamps != expected:
        raise ValueError("Incomplete, duplicate or unordered UTC delivery grid")
