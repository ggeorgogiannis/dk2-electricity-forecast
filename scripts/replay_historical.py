"""Replay the original LightGBM into an explicitly labelled historical artifact."""

import argparse
import sys
from datetime import UTC, date, datetime
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dk2_forecast.experiment import json_bytes
from dk2_forecast.forecast import run_forecast
from dk2_forecast.storage import digest, publish_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prices", type=Path)
    parser.add_argument("delivery_date", type=date.fromisoformat)
    parser.add_argument("--train-for", type=date.fromisoformat)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    now = datetime.now(UTC)
    output = args.root / "replays" / now.strftime("%Y%m%dT%H%M%S%fZ")
    path = run_forecast(args.prices, output, args.delivery_date, args.train_for)
    publish_bytes(
        path.with_suffix(".json"),
        json_bytes(
            {
                "status": "historical_replay",
                "replay": True,
                "created_at_utc": datetime.now(UTC),
                "forecast_sha256": digest(path.read_bytes()),
                "prices_sha256": digest(args.prices.read_bytes()),
                "units": "EUR/MWh",
                "availability_assumption": "Archived historical prices treated as known at cutoff; publication vintage unavailable",
            }
        ),
    )
    print(path)


if __name__ == "__main__":
    main()
