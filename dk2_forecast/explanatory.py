"""Capture individual weather runs and retain evidence of their availability."""

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from math import isfinite
from pathlib import Path
from time import sleep

import httpx
import numpy as np

from dk2_forecast.delivery_day import COPENHAGEN, delivery_intervals_utc
from dk2_forecast.storage import connect, digest, publish_bytes, write_parquet

MODEL = "ecmwf_ifs025"
LOCATION = {"latitude": 55.68, "longitude": 12.57, "name": "Copenhagen proxy"}
VARIABLES = {
    "temperature_2m": "°C",
    "wind_speed_100m": "m/s",
    "shortwave_radiation_instant": "W/m²",
}
RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
META_URL = f"https://api.open-meteo.com/data/{MODEL}/static/meta.json"


def snapshot_cutoff(day: date) -> datetime:
    issue_day = day - timedelta(days=1)
    return (
        datetime.combine(issue_day, datetime.min.time(), COPENHAGEN)
        .replace(hour=9, minute=50)
        .astimezone(UTC)
    )


def get_response(client, url, params=None):
    for attempt in range(3):
        try:
            response = client.get(url, params=params, timeout=45)
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as error:
            if (
                error.response.status_code not in (429, 500, 502, 503, 504)
                or attempt == 2
            ):
                raise
            wait = error.response.headers.get("Retry-After", "")
            delay = int(wait) if wait.isdecimal() else 0
            if wait and not wait.isdecimal():
                try:
                    delay = max(
                        0,
                        (
                            parsedate_to_datetime(wait) - datetime.now(UTC)
                        ).total_seconds(),
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
            if delay > 60:
                raise
            sleep(max(2**attempt, delay))
        except httpx.TransportError:
            if attempt == 2:
                raise
            sleep(2**attempt)
    raise AssertionError("Retry loop exited")


def capture(root: Path, day: date, client, *, historical_assumption=False) -> Path:
    cutoff = snapshot_cutoff(day)
    now = datetime.now(UTC)
    metadata_raw = get_response(client, META_URL)
    metadata = json.loads(metadata_raw)
    if historical_assumption:
        # Conservative cycle choice, with a declared unverifiable historical delay.
        run = datetime.combine(
            day - timedelta(days=2), datetime.min.time(), UTC
        ).replace(hour=18)
        available = run + timedelta(hours=8)
        assumption = "Historical availability assumed eight hours after initialization; no archived API availability metadata"
    else:
        run = datetime.fromtimestamp(metadata["last_run_initialisation_time"], UTC)
        available = datetime.fromtimestamp(metadata["last_run_availability_time"], UTC)
        if now > cutoff or available + timedelta(minutes=10) > cutoff:
            raise ValueError(
                "Weather snapshot or API availability plus 10-minute buffer exceeds 09:50 cutoff"
            )
        if now - run > timedelta(hours=24):
            raise ValueError("Weather initialization is more than 24 hours old")
        assumption = None
    if available > cutoff:
        raise ValueError("Run was unavailable at the data cutoff")
    params = {
        **{key: LOCATION[key] for key in ("latitude", "longitude")},
        "hourly": ",".join(VARIABLES),
        "models": MODEL,
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "forecast_hours": 84,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    raw = get_response(client, RUN_URL, params)
    retrieved = datetime.now(UTC)
    if not historical_assumption and retrieved > cutoff:
        raise ValueError("Weather retrieval completed after the data cutoff")
    for content in (raw, metadata_raw):
        publish_bytes(root / "weather" / "raw" / f"{digest(content)}.json", content)
    payload = json.loads(raw)
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Expected UTC weather timestamps")
    units = payload["hourly_units"]
    if any(units.get(name) != unit for name, unit in VARIABLES.items()):
        raise ValueError(f"Unexpected weather units: {units}")
    times = [
        datetime.fromisoformat(value).replace(tzinfo=UTC)
        for value in payload["hourly"]["time"]
    ]
    if any(stamp.minute or stamp.second for stamp in times) or len(set(times)) != len(
        times
    ):
        raise ValueError("Misaligned or duplicate hourly weather keys")
    series = {}
    for name in VARIABLES:
        values = payload["hourly"][name]
        if len(values) != len(times):
            raise ValueError(f"Incomplete weather variable {name}")
        series[name] = dict(zip(times, values, strict=True))
    rows = []
    for stamp in delivery_intervals_utc(day):
        hour = stamp.replace(minute=0)
        for name, values in series.items():
            if (
                hour not in values
                or type(values[hour]) not in (int, float)
                or not isfinite(values[hour])
            ):
                raise ValueError(f"Weather run does not cover {stamp}")
            rows.append(
                {
                    "delivery_start_utc": stamp.isoformat(),
                    "source_valid_utc": hour.isoformat(),
                    "variable": name,
                    "value": values[hour],
                }
            )
    manifest = {
        "source": "Open-Meteo",
        "dataset": "Single Runs API",
        "model": MODEL,
        "delivery_date": str(day),
        "location": LOCATION,
        "grid_location": {
            key: payload[key] for key in ("latitude", "longitude", "elevation")
        },
        "variables": VARIABLES,
        "source_resolution": "PT1H",
        "alignment": "Left-hold hourly instantaneous values across four quarter-hours; no interpolation",
        "run_initialization_utc": run.isoformat(),
        "available_at_utc": available.isoformat(),
        "retrieved_at_utc": retrieved.isoformat(),
        "cutoff_utc": cutoff.isoformat(),
        "revision": None,
        "raw_sha256": digest(raw),
        "metadata_sha256": digest(metadata_raw),
        "historical_availability_assumption": assumption,
        "status": "historical_assumption" if assumption else "verified_live",
        "data_license": "CC BY 4.0",
        "api_use": "Non-commercial free API",
    }
    content = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    folder = root / "weather" / "snapshots" / str(day) / digest(content)
    schema = [
        {
            "delivery_start_utc": "TIMESTAMPTZ",
            "source_valid_utc": "TIMESTAMPTZ",
            "variable": "VARCHAR",
            "value": "DOUBLE",
        }
    ]
    with connect() as db:
        db.execute(
            "CREATE TABLE weather AS SELECT unnest(from_json_strict(?, ?), recursive := true)",
            [json.dumps(rows), json.dumps(schema)],
        )
        write_parquet(
            db,
            "SELECT * FROM weather ORDER BY delivery_start_utc, variable",
            folder / "weather.parquet",
        )
    publish_bytes(folder / "manifest.json", content)
    return folder


def align_weather(
    root: Path, timestamps, *, allow_assumed=False, interpolate=False
) -> tuple[np.ndarray, dict]:
    matrix = np.full((len(timestamps), len(VARIABLES)), np.nan)
    accepted = {}
    for day in sorted({stamp.astimezone(COPENHAGEN).date() for stamp in timestamps}):
        candidates = []
        for path in (root / "weather" / "snapshots" / str(day)).glob("*/manifest.json"):
            raw = path.read_bytes()
            manifest = json.loads(raw)
            if digest(raw) != path.parent.name:
                raise ValueError("Weather manifest hash mismatch")
            if manifest["status"] != "verified_live" and not allow_assumed:
                continue
            if datetime.fromisoformat(manifest["available_at_utc"]) > snapshot_cutoff(
                day
            ):
                continue
            if manifest["status"] == "verified_live" and datetime.fromisoformat(
                manifest["retrieved_at_utc"]
            ) > snapshot_cutoff(day):
                continue
            if manifest["status"] == "verified_live" and (
                snapshot_cutoff(day)
                - datetime.fromisoformat(manifest["run_initialization_utc"])
            ) > timedelta(hours=24):
                continue
            if (
                digest(
                    (
                        root / "weather" / "raw" / f"{manifest['raw_sha256']}.json"
                    ).read_bytes()
                )
                != manifest["raw_sha256"]
            ):
                raise ValueError("Weather response hash mismatch")
            candidates.append(
                (
                    int(manifest["status"] == "verified_live"),
                    manifest["run_initialization_utc"],
                    manifest["available_at_utc"],
                    manifest["retrieved_at_utc"],
                    path,
                )
            )
        if candidates:
            accepted[day] = max(candidates)[-1]
    values = {}
    with connect() as db:
        for path in accepted.values():
            for stamp, variable, value in db.execute(
                "SELECT delivery_start_utc, variable, value FROM read_parquet(?)",
                [str(path.with_name("weather.parquet"))],
            ).fetchall():
                values[stamp, variable] = value
            if interpolate:
                manifest = json.loads(path.read_bytes())
                raw = json.loads(
                    (
                        root / "weather" / "raw" / f"{manifest['raw_sha256']}.json"
                    ).read_bytes()
                )
                hourly_times = [
                    datetime.fromisoformat(value).replace(tzinfo=UTC)
                    for value in raw["hourly"]["time"]
                ]
                lookup = {
                    name: dict(zip(hourly_times, raw["hourly"][name], strict=True))
                    for name in VARIABLES
                }
                day = date.fromisoformat(manifest["delivery_date"])
                for stamp in delivery_intervals_utc(day):
                    hour = stamp.replace(minute=0)
                    weight = stamp.minute / 60
                    for name, series in lookup.items():
                        first, second = (
                            series.get(hour),
                            series.get(hour + timedelta(hours=1)),
                        )
                        if first is not None and (second is not None or weight == 0):
                            values[stamp, name] = (
                                first
                                if weight == 0
                                else first * (1 - weight) + second * weight
                            )
    for i, stamp in enumerate(timestamps):
        matrix[i] = [values.get((stamp, name), np.nan) for name in VARIABLES]
    return matrix, {str(day): path.parent.name for day, path in accepted.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Capture Open-Meteo individual runs with availability evidence."
    )
    parser.add_argument("start", type=date.fromisoformat, nargs="?")
    parser.add_argument(
        "end", type=date.fromisoformat, nargs="?", help="Exclusive delivery date"
    )
    parser.add_argument(
        "--next-day",
        action="store_true",
        help="Capture tomorrow before its 09:50 cutoff",
    )
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--historical-assumption", action="store_true")
    args = parser.parse_args()
    if args.next_day:
        if args.start is not None or args.end is not None or args.historical_assumption:
            parser.error("--next-day requires live mode with no explicit dates")
        args.start = datetime.now(COPENHAGEN).date() + date.resolution
        args.end = args.start + date.resolution
    if args.start is None or args.end is None:
        parser.error("Supply start and end, or --next-day")
    if args.start >= args.end:
        parser.error("start must precede end")
    with httpx.Client(follow_redirects=True) as client:
        for offset in range((args.end - args.start).days):
            day = args.start + timedelta(days=offset)
            existing = list(
                (args.root / "weather" / "snapshots" / str(day)).glob("*/manifest.json")
            )
            if existing and not args.next_day:
                print(f"{day}: cached")
                continue
            try:
                print(
                    capture(
                        args.root,
                        day,
                        client,
                        historical_assumption=args.historical_assumption,
                    ),
                    flush=True,
                )
            except (httpx.HTTPError, ValueError) as error:
                failure = {
                    "delivery_date": str(day),
                    "retrieved_at_utc": datetime.now(UTC).isoformat(),
                    "status": "unavailable",
                    "error": str(error),
                }
                if isinstance(error, httpx.HTTPStatusError):
                    failure["response"] = error.response.text
                content = (json.dumps(failure, indent=2) + "\n").encode()
                publish_bytes(
                    args.root
                    / "weather"
                    / "failures"
                    / str(day)
                    / f"{digest(content)}.json",
                    content,
                )
                print(
                    f"{day}: unavailable, recorded {type(error).__name__}", flush=True
                )
                if args.next_day:
                    raise RuntimeError(
                        "Live weather capture failed; inspect data/weather/failures"
                    ) from error


if __name__ == "__main__":
    main()
