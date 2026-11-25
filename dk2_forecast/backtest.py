import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

from dk2_forecast.forecast import (
    ForecastRow,
    PriceHistory,
    TrainingData,
    predict_day,
    save_forecasts,
    save_model,
)
from dk2_forecast.storage import publish_bytes, digest

FOLD_DAYS = 28
SPIKE_EUR_PER_MWH = 200
PREDICTIONS = ("baseline_eur_per_mwh", "lightgbm_eur_per_mwh")


def fold_ranges(start: date, end: date, days: int = FOLD_DAYS):
    while start < end:
        stop = min(start + timedelta(days=days), end)
        yield start, stop
        start = stop


def evaluate(rows: list[ForecastRow], actuals: dict) -> dict:
    records = [
        row._asdict() | {"actual_eur_per_mwh": actuals.get(row.delivery_start_utc)}
        for row in rows
    ]
    observed = [row for row in records if row["actual_eur_per_mwh"] is not None]
    paired = [
        row for row in observed if all(row[name] is not None for name in PREDICTIONS)
    ]

    def scores(subset):
        return {
            "intervals": len(subset),
            "mae_eur_per_mwh": {
                name: mean(abs(row[name] - row["actual_eur_per_mwh"]) for row in subset)
                if subset
                else None
                for name in PREDICTIONS
            },
        }

    worst = {}
    for name in PREDICTIONS:
        examples = []
        seen_days = set()
        for row in sorted(
            paired,
            key=lambda row: abs(row[name] - row["actual_eur_per_mwh"]),
            reverse=True,
        ):
            if row["delivery_date"] in seen_days:
                continue
            seen_days.add(row["delivery_date"])
            examples.append(
                {
                    "delivery_date": row["delivery_date"].isoformat(),
                    "delivery_start_utc": row["delivery_start_utc"].isoformat(),
                    "actual_eur_per_mwh": row["actual_eur_per_mwh"],
                    **{method: row[method] for method in PREDICTIONS},
                    "absolute_error_eur_per_mwh": abs(
                        row[name] - row["actual_eur_per_mwh"]
                    ),
                }
            )
            if len(examples) == 3:
                break
        worst[name] = examples

    return {
        "expected_intervals": len(rows),
        "observed_intervals": len(observed),
        "coverage": {
            name: sum(row[name] is not None for row in records) / len(rows)
            for name in PREDICTIONS
        },
        "all": scores(paired),
        "negative": scores([row for row in paired if row["actual_eur_per_mwh"] < 0]),
        "spike": scores(
            [row for row in paired if row["actual_eur_per_mwh"] >= SPIKE_EUR_PER_MWH]
        ),
        "worst_by_method": worst,
    }


def run_backtest(
    prices: Path, root: Path, start: date, test_start: date, end: date
) -> Path:
    if not start < test_start < end:
        raise ValueError("Require start < test_start < end (exclusive)")
    history = PriceHistory.load(prices)
    training = TrainingData(history, start)
    all_rows = []
    folds = []

    def run_fold(origin: date, stop: date) -> list[ForecastRow]:
        model = training.fit(origin)
        version = save_model(model, root)
        return [
            row
            for offset in range((stop - origin).days)
            for row in predict_day(
                history, model, version, origin, origin + timedelta(days=offset)
            )
        ]

    for origin, stop in fold_ranges(start, test_start):
        rows = run_fold(origin, stop)
        all_rows.extend(rows)
        folds.append(
            {
                "start": origin.isoformat(),
                "end_exclusive": stop.isoformat(),
                "model_version": rows[0].model_version,
                **evaluate(rows, history.prices),
            }
        )

    test_rows = run_fold(test_start, end)
    report = {
        "data_version": history.data_version,
        "spike_threshold_eur_per_mwh": SPIKE_EUR_PER_MWH,
        "development_folds": folds,
        "development_pooled": evaluate(all_rows, history.prices),
        "test": {
            "start": test_start.isoformat(),
            "end_exclusive": end.isoformat(),
            "model_version": test_rows[0].model_version,
            **evaluate(test_rows, history.prices),
        },
    }
    content = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    output = root / "backtests" / digest(content)
    save_forecasts(all_rows + test_rows, output / "forecasts.parquet")
    publish_bytes(output / "report.json", content)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an expanding-window DK2 price backtest."
    )
    parser.add_argument("prices", type=Path)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 1))
    parser.add_argument(
        "--test-start", type=date.fromisoformat, default=date(2026, 9, 21)
    )
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 11, 16))
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(run_backtest(args.prices, args.root, args.start, args.test_start, args.end))


if __name__ == "__main__":
    main()
