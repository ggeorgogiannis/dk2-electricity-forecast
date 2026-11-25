import argparse
import json
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import lightgbm as lgb
import numpy as np

from dk2_forecast.delivery_day import (
    COPENHAGEN,
    delivery_intervals_utc,
    forecast_issue_utc,
)
from dk2_forecast.ingestion import PRICE_AREA
from dk2_forecast.price_lags import lag_price, local_day
from dk2_forecast.storage import publish_bytes, connect, digest, write_parquet

BASELINE_VERSION = "lag-168h-utc-v1"
MIN_TRAIN_DAYS = 90
LAG_DAYS = (1, 2, 3, 7, 14, 28)
FEATURE_NAMES = (
    "local_quarter",
    "weekday",
    "month",
    "utc_offset_hours",
    *(f"price_lag_{days * 24}h" for days in LAG_DAYS),
)
MODEL_PARAMS = {
    "objective": "regression_l1",
    "device_type": "cpu",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "num_threads": 1,
    "deterministic": True,
    "force_col_wise": True,
    "seed": 42,
    "verbosity": -1,
}
BOOSTING_ROUNDS = 160


def features(
    prices: dict[datetime, float], intervals: list[datetime], delivery_date: date
) -> np.ndarray:
    known_through = local_day(forecast_issue_utc(delivery_date))
    rows = []
    for timestamp in intervals:
        local = timestamp.astimezone(COPENHAGEN)
        rows.append(
            (
                local.hour * 4 + local.minute // 15,
                local.weekday(),
                local.month,
                local.utcoffset().total_seconds() / 3600,
                *(
                    lag_price(prices, timestamp, days, known_through)
                    for days in LAG_DAYS
                ),
            )
        )
    return np.asarray(rows, dtype=np.float64)


@dataclass
class PriceHistory:
    prices: dict[datetime, float]
    data_version: str

    @classmethod
    def load(cls, path: Path):
        manifest = path.with_name("manifest.json").read_bytes()
        version = digest(manifest)
        if path.parent.name != version:
            raise ValueError("Price snapshot directory must match its manifest SHA-256")
        with connect() as connection:
            rows = connection.execute(
                """
                SELECT delivery_start_utc, price_eur_per_mwh
                FROM read_parquet(?) WHERE price_area = ? ORDER BY delivery_start_utc
                """,
                [str(path), PRICE_AREA],
            ).fetchall()
        if not rows or len(dict(rows)) != len(rows):
            raise ValueError("Price snapshot must contain unique DK2 intervals")
        return cls(dict(rows), version)


class TrainingData:
    def __init__(self, history: PriceHistory, first_origin: date):
        self.intervals = sorted(history.prices)
        first_day = local_day(self.intervals[0])
        cutoff = delivery_intervals_utc(first_origin)[0]
        if first_origin < first_day + timedelta(days=MIN_TRAIN_DAYS):
            raise ValueError(
                f"Training requires at least {MIN_TRAIN_DAYS} delivery days"
            )
        if cutoff > self.intervals[-1] + timedelta(minutes=15):
            raise ValueError(
                "Training snapshot ends before the last eligible delivery day"
            )
        self.labels = np.asarray(
            [history.prices[timestamp] for timestamp in self.intervals]
        )
        days = sorted({local_day(timestamp) for timestamp in self.intervals})
        self.matrix = np.concatenate(
            [
                features(
                    history.prices,
                    [
                        timestamp
                        for timestamp in delivery_intervals_utc(day)
                        if timestamp in history.prices
                    ],
                    day,
                )
                for day in days
            ]
        )

    def fit(self, train_for: date) -> lgb.Booster:
        count = self.eligible_count(train_for)
        training = lgb.Dataset(
            self.matrix[:count],
            label=self.labels[:count],
            feature_name=list(FEATURE_NAMES),
        )
        return lgb.train(MODEL_PARAMS, training, num_boost_round=BOOSTING_ROUNDS)

    def eligible_count(self, train_for: date) -> int:
        # At 10:00 only labels through the local issue date are eligible.
        cutoff = delivery_intervals_utc(train_for)[0]
        return bisect_left(self.intervals, cutoff)


class ForecastRow(NamedTuple):
    price_area: str
    delivery_date: date
    delivery_start_utc: datetime
    issue_time_utc: datetime
    model_training_origin: date
    model_version: str
    baseline_model_version: str
    data_version: str
    lightgbm_eur_per_mwh: float
    baseline_eur_per_mwh: float | None


def save_model(model: lgb.Booster, root: Path) -> str:
    content = model.model_to_string().encode()
    version = f"lightgbm-{digest(content)}"
    publish_bytes(root / "models" / f"{version}.txt", content)
    return version


def predict_day(
    history: PriceHistory,
    model: lgb.Booster,
    model_version: str,
    train_for: date,
    delivery_date: date,
) -> list[ForecastRow]:
    if train_for > delivery_date:
        raise ValueError("Model training origin must be at or before the delivery date")
    intervals = delivery_intervals_utc(delivery_date)
    issue = forecast_issue_utc(delivery_date)
    predictions = model.predict(
        features(history.prices, intervals, delivery_date), num_threads=1
    )
    return [
        ForecastRow(
            PRICE_AREA,
            delivery_date,
            timestamp,
            issue,
            train_for,
            model_version,
            BASELINE_VERSION,
            history.data_version,
            float(prediction),
            lag_price(history.prices, timestamp, 7, local_day(issue)),
        )
        for timestamp, prediction in zip(intervals, predictions, strict=True)
    ]


def save_forecasts(rows: list[ForecastRow], path: Path) -> None:
    schema = {
        "price_area": "VARCHAR",
        "delivery_date": "DATE",
        "delivery_start_utc": "TIMESTAMPTZ",
        "issue_time_utc": "TIMESTAMPTZ",
        "model_training_origin": "DATE",
        "model_version": "VARCHAR",
        "baseline_model_version": "VARCHAR",
        "data_version": "VARCHAR",
        "lightgbm_eur_per_mwh": "DOUBLE",
        "baseline_eur_per_mwh": "DOUBLE",
    }
    data = json.dumps(
        [row._asdict() for row in rows],
        default=lambda value: value.isoformat(),
        allow_nan=False,
    )
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE forecasts AS
            SELECT unnest(from_json_strict(?, ?), recursive := true)
            """,
            [data, json.dumps([schema])],
        )
        write_parquet(
            connection, "SELECT * FROM forecasts ORDER BY delivery_start_utc", path
        )


def run_forecast(
    prices: Path,
    root: Path,
    delivery_date: date,
    train_for: date | None = None,
    *,
    prepare: bool = False,
) -> Path:
    origin = train_for or delivery_date
    if origin > delivery_date:
        raise ValueError("Model training origin must be at or before the delivery date")
    history = PriceHistory.load(prices)
    model = TrainingData(history, origin).fit(origin)
    version = save_model(model, root)
    rows = predict_day(history, model, version, origin, delivery_date)
    folder = root / ("prepared" if prepare else "forecasts") / str(delivery_date)
    path = folder / f"{origin}_{version}_{history.data_version}.parquet"
    save_forecasts(rows, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a 10:00 Danish day-ahead forecast."
    )
    parser.add_argument("prices", type=Path, help="Versioned prices.parquet")
    parser.add_argument("delivery_date", type=date.fromisoformat)
    parser.add_argument(
        "--train-for", type=date.fromisoformat, help="Optional earlier fit origin"
    )
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(run_forecast(args.prices, args.root, args.delivery_date, args.train_for))


if __name__ == "__main__":
    main()
