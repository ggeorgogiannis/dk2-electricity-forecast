import argparse
import sys
from datetime import date
from pathlib import Path

import httpx


def main() -> None:
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dk2_forecast.ingestion import fetch_dk2_prices

    parser = argparse.ArgumentParser(description="Check a DK2 day of day-ahead prices.")
    parser.add_argument(
        "date", nargs="?", type=date.fromisoformat, default=date(2026, 10, 5)
    )
    args = parser.parse_args()

    with httpx.Client() as client:
        prices = fetch_dk2_prices(client, args.date)

    print(f"DK2 records: {len(prices)}")
    print(f"First UTC interval: {prices[0].time_utc.isoformat()}")
    print(f"Last UTC interval:  {prices[-1].time_utc.isoformat()}")
    print(f"Example price: {prices[0].price_eur_per_mwh} EUR/MWh")


if __name__ == "__main__":
    main()
