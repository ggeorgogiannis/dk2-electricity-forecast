"""Controlled development tail experiments and prior-only interval benchmarks."""

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np

from dk2_forecast.backtest import fold_ranges
from dk2_forecast.delivery_day import forecast_issue_utc
from dk2_forecast.evaluation import read_json
from dk2_forecast.experiment import json_bytes
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.price_lags import local_day
from dk2_forecast.provenance import save_code_snapshot
from dk2_forecast.research import ResearchData
from dk2_forecast.research_metrics import interval_metrics, point_metrics
from dk2_forecast.storage import connect, digest, publish_bytes, write_parquet


def prior_gate(days, actual, candidate, previous, *, window_days=56, minimum=192):
    output = np.full(len(actual), np.nan)
    records = []
    regimes = np.where(
        previous < 0,
        "negative_previous",
        np.where(previous >= 200, "high_previous", "regular_previous"),
    )
    for origin, stop in fold_ranges(min(days), max(days) + timedelta(days=1), 7):
        prior = (days < origin) & (days >= origin - timedelta(days=window_days))
        target = (days >= origin) & (days < stop)
        for regime in sorted(set(regimes)):
            train = prior & (regimes == regime)
            mask = target & (regimes == regime)
            weight = (
                min(
                    (0, 0.25, 0.5, 0.75, 1),
                    key=lambda w: np.abs(
                        actual[train]
                        - (w * candidate[train] + (1 - w) * previous[train])
                    ).mean(),
                )
                if train.sum() >= minimum
                else 0.5
            )
            output[mask] = weight * candidate[mask] + (1 - weight) * previous[mask]
            records.append(
                {
                    "origin": str(origin),
                    "regime_from_available_previous_day": regime,
                    "candidate_weight": weight,
                    "prior_intervals": int(train.sum()),
                    "used_default_weight": bool(train.sum() < minimum),
                }
            )
    return output, records


def run(report_path, prices, root):
    report = read_json(report_path)
    with connect() as db:
        query = db.execute(
            "SELECT * FROM read_parquet(?) ORDER BY delivery_start_utc",
            [str(report_path.with_name("predictions.parquet"))],
        )
        names = [c[0] for c in query.description]
        rows = [dict(zip(names, row, strict=True)) for row in query.fetchall()]
    stamps = [row["delivery_start_utc"] for row in rows]
    days = np.asarray([local_day(t) for t in stamps])
    actual = np.asarray([r["actual"] for r in rows])
    candidate = np.asarray([r["weekly_refit"] for r in rows])
    previous = np.asarray([r["previous_day"] for r in rows])
    history = PriceHistory.load(prices)
    if history.data_version != report["data_version"]:
        raise ValueError("Failure analysis requires the original development snapshot")
    data = ResearchData(history, root)
    residual, fit = data.evaluate(
        "lightgbm",
        report["fits"]["lightgbm_tuned"]["parameters"],
        frequency=7,
        correction=True,
    )
    gated, weights = prior_gate(days, actual, candidate, previous)
    predictions = {
        "previous_day": previous,
        "candidate": candidate,
        "fixed_half_blend": (candidate + previous) / 2,
        "residual_correction_weekly": residual,
        "prior_only_gate": gated,
    }
    point = point_metrics(stamps, actual, predictions, report["scoring"])
    extra = {}
    horizon = np.asarray(
        [(t - forecast_issue_utc(local_day(t))).total_seconds() / 3600 for t in stamps]
    )
    for label, values in {
        "forecast_horizon_6h_bin": np.floor(horizon / 6) * 6,
        "predicted_price_bin": np.digitize(candidate, [0, 100, 200]),
        "previous_day_price_bin": np.digitize(previous, [0, 100, 200]),
    }.items():
        extra[label] = {
            str(key): point_metrics(
                [t for i, t in enumerate(stamps) if values[i] == key],
                actual[values == key],
                {k: v[values == key] for k, v in predictions.items()},
                report["scoring"],
                resamples=100,
            )
            for key in sorted(set(values))
        }
    quantiles = np.asarray([[r[f"q{q}"] for q in (10, 50, 90)] for r in rows])
    raw_sorted = np.sort(quantiles, axis=1)
    rolling = raw_sorted.copy()
    historical = np.full_like(quantiles, np.nan)
    calibration = []
    for origin, stop in fold_ranges(min(days), max(days) + timedelta(days=1), 7):
        target = (days >= origin) & (days < stop)
        prior = (days < origin) & (days >= origin - timedelta(days=28))
        adjustment = 0.0
        if prior.sum() >= 14 * 92:
            scores = np.maximum(
                raw_sorted[prior, 0] - actual[prior],
                actual[prior] - raw_sorted[prior, 2],
            )
            adjustment = max(0, float(np.quantile(scores, 0.8, method="higher")))
        rolling[target, 0] -= adjustment
        rolling[target, 2] += adjustment
        eligible = (data.days < origin) & (data.days >= origin - timedelta(days=56))
        residuals = (data.actual - data.matrix[:, 0])[eligible]
        residuals = residuals[np.isfinite(residuals)]
        for i, alpha in enumerate((0.1, 0.5, 0.9)):
            historical[target, i] = previous[target] + np.quantile(residuals, alpha)
        calibration.append(
            {
                "origin": str(origin),
                "prior_intervals": int(prior.sum()),
                "margin": adjustment,
                "fallback_raw": bool(prior.sum() < 14 * 92),
            }
        )
    intervals = {
        "raw": interval_metrics(stamps, actual, quantiles),
        "existing_prior_calibration": interval_metrics(
            stamps,
            actual,
            [[r[f"calibrated_q{q}"] for q in (10, 50, 90)] for r in rows],
        ),
        "rolling_28day_calibration": interval_metrics(stamps, actual, rolling),
        "historical_previous_day_residual_56day": interval_metrics(
            stamps, actual, historical
        ),
    }
    worst = []
    for regime, mask in {
        "negative": actual < 0,
        "high": actual >= 200,
        "extreme": abs(actual) >= report["scoring"]["extreme_threshold_eur_per_mwh"],
        "all": np.ones(len(actual), dtype=bool),
    }.items():
        for index in np.flatnonzero(mask)[
            np.argsort(abs(candidate[mask] - actual[mask]))[-8:][::-1]
        ]:
            worst.append(
                {
                    "slice": regime,
                    "delivery_start_utc": stamps[index],
                    "actual": float(actual[index]),
                    "previous_day": float(previous[index]),
                    "candidate": float(candidate[index]),
                    "signed_error_prediction_minus_actual": float(
                        candidate[index] - actual[index]
                    ),
                    "horizon_hours": float(horizon[index]),
                }
            )
    result = {
        "status": "development_failure_analysis_selection_affected_not_live",
        "source_report_sha256": digest(report_path.read_bytes()),
        "source_predictions_sha256": digest(
            report_path.with_name("predictions.parquet").read_bytes()
        ),
        "code_snapshot_sha256": save_code_snapshot(root),
        "period": report["period"],
        "scoring": report["scoring"],
        "point": point,
        "conditional": extra,
        "large_errors": worst,
        "gate_policy": {
            "features": ["previous_day"],
            "window_days": 56,
            "refit_days": 7,
            "minimum_prior_intervals": 192,
            "default_candidate_weight": 0.5,
        },
        "gate_weights": weights,
        "residual_fit": fit,
        "intervals": intervals,
        "rolling_calibration_policy": {
            "prior_days": 28,
            "update_days": 7,
            "minimum_prior_days": 14,
            "fallback": "raw sorted quantiles",
            "margin_quantile": 0.8,
        },
        "calibration": calibration,
    }
    raw = json_bytes(result)
    folder = root / "failure_analysis" / digest(raw)
    publish_bytes(folder / "report.json", raw)
    forecast_rows = [
        {
            "delivery_start_utc": t.isoformat(),
            "actual": float(actual[i]),
            **{k: float(v[i]) for k, v in predictions.items()},
        }
        for i, t in enumerate(stamps)
    ]
    schema = [
        {
            "delivery_start_utc": "TIMESTAMPTZ",
            **{k: "DOUBLE" for k in forecast_rows[0] if k != "delivery_start_utc"},
        }
    ]
    with connect() as db:
        import json

        db.execute(
            "CREATE TABLE predictions AS SELECT unnest(from_json_strict(?, ?), recursive := true)",
            [json.dumps(forecast_rows), json.dumps(schema)],
        )
        write_parquet(
            db,
            "SELECT * FROM predictions ORDER BY delivery_start_utc",
            folder / "predictions.parquet",
        )
    return folder / "report.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("prices", type=Path)
    p.add_argument("--root", type=Path, default=Path("data"))
    a = p.parse_args()
    print(run(a.report, a.prices, a.root))


if __name__ == "__main__":
    main()
