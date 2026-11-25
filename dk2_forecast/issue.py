import argparse
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import httpx

from dk2_forecast.delivery_day import COPENHAGEN, forecast_issue_utc
from dk2_forecast.forecast import PriceHistory, run_forecast
from dk2_forecast.ingestion import FIRST_DAY, ingest_prices
from dk2_forecast.storage import digest, publish_bytes


def capture_inputs(root: Path, delivery_date: date):
    with httpx.Client() as client:
        path = ingest_prices(root, FIRST_DAY, delivery_date, client)
    history = PriceHistory.load(path)
    price_hash = digest(path.read_bytes())
    observed_at = datetime.now(UTC)
    if observed_at > forecast_issue_utc(delivery_date):
        raise ValueError("Input snapshot was observed after the 10:00 cutoff")
    return (
        path,
        history,
        {
            "data_version": history.data_version,
            "prices_sha256": price_hash,
            "observed_at_utc": observed_at.isoformat(),
        },
    )


def save_preparation(path: Path, delivery_date: date, inputs: dict, **extra) -> None:
    forecast_hash = digest(path.read_bytes())
    ready_at = datetime.now(UTC)
    if ready_at > forecast_issue_utc(delivery_date):
        raise ValueError("The forecast finished preparation after the 10:00 cutoff")
    metadata = (
        inputs
        | extra
        | {
            "forecast_sha256": forecast_hash,
            "prepared_at_utc": ready_at.isoformat(),
        }
    )
    receipt = path.with_suffix(".json")
    result = json.loads(publish_bytes(receipt, json.dumps(metadata, indent=2).encode()))
    if result["forecast_sha256"] != forecast_hash:
        raise ValueError(f"Existing preparation receipt differs at {receipt}")


def publish_prepared(root: Path, now: datetime) -> Path:
    now = now.astimezone(COPENHAGEN)
    if not time(10) <= now.time() < time(10, 1):
        raise ValueError(
            "Live issuance must start in the 10:00 Europe/Copenhagen minute"
        )
    day = now.date() + timedelta(days=1)
    candidates = list((root / "prepared" / str(day)).glob("*.parquet"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one prepared forecast for {day}, found {len(candidates)}"
        )
    staged = candidates[0]
    path = root / "forecasts" / str(day) / staged.name
    content = staged.read_bytes()
    receipt = json.loads(staged.with_suffix(".json").read_bytes())
    if digest(content) != receipt["forecast_sha256"] or not datetime.fromisoformat(
        receipt["observed_at_utc"]
    ) <= datetime.fromisoformat(receipt["prepared_at_utc"]) <= forecast_issue_utc(day):
        raise ValueError(f"Invalid pre-cutoff receipt for {staged}")
    publication = receipt | {"published_at_utc": now.astimezone(UTC).isoformat()}
    existing = json.loads(
        publish_bytes(
            path.with_suffix(".json"), json.dumps(publication, indent=2).encode()
        )
    )
    if existing["forecast_sha256"] != digest(content):
        raise ValueError(f"Stored publication receipt differs at {path}")
    if publish_bytes(path, content) != content:
        raise ValueError(
            f"Stored live forecast differs from the prepared forecast at {path}"
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or issue the next Danish delivery day."
    )
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--prepare", action="store_true", help="Prepare before the 10:00 issue"
    )
    args = parser.parse_args()
    now = datetime.now(COPENHAGEN)
    delivery_date = now.date() + timedelta(days=1)
    if args.prepare:
        if now.time() >= time(10):
            parser.error(
                "Forecast preparation must start before 10:00 Europe/Copenhagen"
            )
        prices, _, receipt = capture_inputs(args.root, delivery_date)
        stage = run_forecast(prices, args.root, delivery_date, prepare=True)
        save_preparation(stage, delivery_date, receipt)
        print(stage)
        return
    print(publish_prepared(args.root, now))


if __name__ == "__main__":
    main()
