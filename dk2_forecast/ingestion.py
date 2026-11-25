import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from math import isfinite
from pathlib import Path
from time import sleep

import httpx

from dk2_forecast.delivery_day import delivery_intervals_utc
from dk2_forecast.storage import publish_bytes, connect, digest, write_parquet

DAY_AHEAD_PRICES_URL = "https://api.energidataservice.dk/dataset/DayAheadPrices"
PRICE_AREA = "DK2"
FIRST_DAY = date(2025, 10, 1)
MAX_CAPTURE_DAYS = timedelta(days=84)


class IncompleteDeliveryDay(ValueError):
    """The requested source interval grid is incomplete or misaligned."""


@dataclass(frozen=True)
class DayAheadPrice:
    time_utc: datetime
    price_eur_per_mwh: float


def request_prices(
    client: httpx.Client, start: date, end: date, *, archive_root: Path | None = None
) -> bytes:
    for attempt in range(4):
        try:
            response = client.get(
                DAY_AHEAD_PRICES_URL,
                params={
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "filter": '{"PriceArea":["DK2"]}',
                    "sort": "TimeUTC",
                    "limit": 0,
                },
                timeout=30.0,
            )
            if response.status_code >= 300 and archive_root is not None:
                error_hash = digest(response.content)
                publish_bytes(
                    archive_root / "raw" / "errors" / f"{error_hash}.body",
                    response.content,
                )
                event = {
                    "source": DAY_AHEAD_PRICES_URL,
                    "dataset": "DayAheadPrices",
                    "price_area": PRICE_AREA,
                    "start": str(start),
                    "end_exclusive": str(end),
                    "retrieved_at_utc": datetime.now(UTC).isoformat(),
                    "status_code": response.status_code,
                    "raw_sha256": error_hash,
                    "retry_after": response.headers.get("Retry-After"),
                }
                event_content = (
                    json.dumps(event, sort_keys=True, indent=2) + "\n"
                ).encode()
                publish_bytes(
                    archive_root / "raw" / "errors" / f"{digest(event_content)}.json",
                    event_content,
                )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as error:
            if attempt == 3 or error.response.status_code not in (
                429,
                500,
                502,
                503,
                504,
            ):
                raise
            retry_after = error.response.headers.get("Retry-After", "")
            wait = re.search(r"try again in (\d+) seconds?", error.response.text, re.I)
            delay = int(retry_after) if retry_after.isdecimal() else 0
            if retry_after and not retry_after.isdecimal():
                try:
                    delay = max(
                        0,
                        (
                            parsedate_to_datetime(retry_after) - datetime.now(UTC)
                        ).total_seconds(),
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
            if wait:
                delay = max(delay, int(wait[1]))
            sleep(max(2**attempt, delay))
        except httpx.TransportError:
            if attempt == 3:
                raise
            sleep(2**attempt)
    raise AssertionError("Retry loop exited without a result")


def validate_prices(raw: bytes, start: date, end: date) -> list[DayAheadPrice]:
    expected = [
        timestamp
        for offset in range((end - start).days)
        for timestamp in delivery_intervals_utc(start + timedelta(days=offset))
    ]
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise TypeError(f"Expected a list of DK2 records for {start} to {end}")
    if payload.get("dataset", "DayAheadPrices") != "DayAheadPrices":
        raise ValueError("Unexpected source dataset identifier")

    prices = {}
    for row in payload["records"]:
        if not isinstance(row, dict) or row.get("PriceArea") != PRICE_AREA:
            raise ValueError(f"Expected only DK2 records for {start} to {end}")
        try:
            timestamp = datetime.fromisoformat(row["TimeUTC"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid TimeUTC for {start} to {end}") from error
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        timestamp = timestamp.astimezone(UTC)

        price = row.get("DayAheadPriceEUR")
        if type(price) not in (int, float) or not isfinite(price):
            raise ValueError(f"Invalid DK2 price at {timestamp.isoformat()}")
        record = DayAheadPrice(timestamp, float(price) if price != 0 else 0.0)
        if prices.setdefault(timestamp, record) != record:
            raise ValueError(f"Conflicting DK2 prices at {timestamp.isoformat()}")

    expected_set = set(expected)
    if prices.keys() != expected_set:
        missing = len(expected_set - prices.keys())
        unexpected = len(prices.keys() - expected_set)
        raise IncompleteDeliveryDay(
            f"Incomplete DK2 range {start} to {end} (exclusive): "
            f"{missing} missing and {unexpected} unexpected intervals"
        )
    return [prices[timestamp] for timestamp in expected]


def fetch_dk2_prices(client: httpx.Client, delivery_date: date) -> list[DayAheadPrice]:
    end = delivery_date + timedelta(days=1)
    return validate_prices(
        request_prices(client, delivery_date, end), delivery_date, end
    )


def ingest_prices(
    root: Path,
    start: date,
    end: date,
    client: httpx.Client | None = None,
    *,
    refresh: bool = False,
) -> Path:
    """Assemble a versioned snapshot. With no client, read archived responses only."""
    if not FIRST_DAY <= start < end:
        raise ValueError(f"Require {FIRST_DAY} <= start < end (exclusive)")
    if refresh and client is None:
        raise ValueError("Refreshing source revisions requires an HTTP client")
    captures = []
    records = []
    lineage = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + MAX_CAPTURE_DAYS, end)
        archive = root / "raw" / f"{chunk_start}_{chunk_end}.json"
        if (archive.exists() and not refresh) or client is None:
            raw = archive.read_bytes()
        else:
            for attempt in range(3):
                raw = request_prices(client, chunk_start, chunk_end, archive_root=root)
                raw_hash = digest(raw)
                object_path = root / "raw" / "objects" / f"{raw_hash}.json"
                publish_bytes(object_path, raw)
                observed = datetime.now(UTC).isoformat()
                observation = {
                    "source": DAY_AHEAD_PRICES_URL,
                    "dataset": "DayAheadPrices",
                    "price_area": PRICE_AREA,
                    "variable": "DayAheadPriceEUR",
                    "units": "EUR/MWh",
                    "source_resolution": "PT15M",
                    "start": str(chunk_start),
                    "end_exclusive": str(chunk_end),
                    "retrieved_at_utc": observed,
                    "raw_sha256": raw_hash,
                    "publication_time": None,
                    "revision": None,
                    "historical_availability_assumption": "Issue-day delivery prices assumed known by 10:00 Copenhagen; source has no point-in-time revision history",
                }
                observation_content = (
                    json.dumps(observation, sort_keys=True, indent=2) + "\n"
                ).encode()
                publish_bytes(
                    root
                    / "raw"
                    / "observations"
                    / f"{digest(observation_content)}.json",
                    observation_content,
                )
                try:
                    validate_prices(raw, chunk_start, chunk_end)
                    break
                except (IncompleteDeliveryDay, json.JSONDecodeError):
                    if attempt == 2:
                        raise
                    sleep(2**attempt)
            if refresh:
                archive = object_path
            else:
                raw = publish_bytes(archive, raw)
            if digest(raw) != observation["raw_sha256"]:
                observation = observation | {
                    "raw_sha256": digest(raw),
                    "retrieved_at_utc": None,
                    "retrieval_note": "A concurrent capture was retained; consult raw/observations for its retrieval receipt",
                }
            lineage.append(observation)
        captures.append(
            {
                "file": archive.relative_to(root / "raw").as_posix(),
                "sha256": digest(raw),
            }
        )
        records.extend(validate_prices(raw, chunk_start, chunk_end))
        chunk_start = chunk_end

    manifest = {
        "dataset": "DayAheadPrices",
        "schema_version": 1,
        "price_area": PRICE_AREA,
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "rows": len(records),
        "raw": captures,
    }
    content = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    snapshot = root / "prices" / digest(content)
    publish_bytes(snapshot / "manifest.json", content)
    path = snapshot / "prices.parquet"
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE prices (
                price_area VARCHAR NOT NULL,
                delivery_start_utc TIMESTAMPTZ NOT NULL,
                price_eur_per_mwh DOUBLE NOT NULL,
                PRIMARY KEY (price_area, delivery_start_utc)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO prices
            SELECT ?, unnest(from_json_strict(?, '["TIMESTAMPTZ"]')),
                      unnest(from_json_strict(?, '["DOUBLE"]'))
            """,
            [
                PRICE_AREA,
                json.dumps([row.time_utc.isoformat() for row in records]),
                json.dumps([row.price_eur_per_mwh for row in records], allow_nan=False),
            ],
        )
        write_parquet(
            connection, "SELECT * FROM prices ORDER BY delivery_start_utc", path
        )
    if lineage:
        publish_bytes(
            snapshot / "availability.json",
            (json.dumps(lineage, indent=2) + "\n").encode(),
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an Energinet DK2 price snapshot."
    )
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument(
        "end", type=date.fromisoformat, help="Exclusive Danish delivery date"
    )
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--offline", action="store_true", help="Replay archived responses"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Capture source corrections as a new immutable snapshot",
    )
    args = parser.parse_args()
    if args.offline and args.refresh:
        parser.error("--offline and --refresh cannot be combined")
    if args.offline:
        path = ingest_prices(args.root, args.start, args.end)
    else:
        with httpx.Client() as client:
            path = ingest_prices(
                args.root, args.start, args.end, client, refresh=args.refresh
            )
    print(path)


if __name__ == "__main__":
    main()
