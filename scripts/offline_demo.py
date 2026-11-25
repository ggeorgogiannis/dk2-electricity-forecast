"""Generate synthetic quarter-hours and run the existing forecast and API inputs."""

import argparse
import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from dk2_forecast.delivery_day import delivery_intervals_utc
from dk2_forecast.experiment import json_bytes
from dk2_forecast.forecast import run_forecast
from dk2_forecast.ingestion import ingest_prices
from dk2_forecast.storage import digest, publish_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/demo"))
    args = parser.parse_args()
    start, end = date(2026, 1, 1), date(2026, 4, 1)

    def respond(request):
        first = date.fromisoformat(request.url.params["start"])
        stop = date.fromisoformat(request.url.params["end"])
        records = []
        for offset in range((stop - first).days):
            day = first + timedelta(days=offset)
            for quarter, stamp in enumerate(delivery_intervals_utc(day)):
                records.append(
                    {
                        "PriceArea": "DK2",
                        "TimeUTC": stamp.isoformat(),
                        "DayAheadPriceEUR": 30
                        + 45 * math.sin(quarter * math.pi / 48)
                        + 3 * day.weekday(),
                    }
                )
        return httpx.Response(200, json={"records": records})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        prices = ingest_prices(args.root, start, end, client)
    forecast = run_forecast(prices, args.root, end)
    publish_bytes(
        forecast.with_suffix(".json"),
        json_bytes(
            {
                "replay": True,
                "status": "synthetic_demo",
                "created_at_utc": datetime.now(UTC),
                "forecast_sha256": digest(forecast.read_bytes()),
                "units": "EUR/MWh",
                "availability_assumption": "Synthetic data generated for an offline integration check",
            }
        ),
    )
    print(f"Prices: {prices}")
    print(f"Forecast: {forecast}")


if __name__ == "__main__":
    main()
