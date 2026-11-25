"""Optional ENTSO-E vintage capture; never substitutes realized target inputs."""

import argparse
from datetime import UTC, date, datetime, timedelta
import os
from pathlib import Path
from time import sleep
from xml.etree import ElementTree as ET

import httpx
import numpy as np

from dk2_forecast.delivery_day import delivery_intervals_utc
from dk2_forecast.experiment import json_bytes
from dk2_forecast.explanatory import snapshot_cutoff
from dk2_forecast.storage import digest, publish_bytes

ENDPOINT = "https://web-api.tp.entsoe.eu/api"
DOMAIN = "10YDK-2--------M"
PSR = {"B16": "solar", "B18": "wind_offshore", "B19": "wind_onshore"}


def text(node, name):
    value = node.findtext("{*}" + name)
    if value is None:
        raise ValueError(f"ENTSO-E field missing: {name}")
    return value


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Source timestamps must carry UTC offset")
    return result.astimezone(UTC)


def parse_vintage(raw, kind, day, retrieved):
    if (
        len(raw) > 10_000_000
        or b"<!DOCTYPE" in raw.upper()
        or b"<!ENTITY" in raw.upper()
    ):
        raise ValueError("Unsupported XML declarations or oversized response")
    document = ET.fromstring(raw)
    expected = "A65" if kind == "load" else "A69"
    if (
        text(document, "type") != expected
        or document.findtext("{*}process.processType") != "A01"
    ):
        raise ValueError("Expected a day-ahead forecast document, never actuals")
    created = timestamp(text(document, "createdDateTime"))
    if not created <= retrieved <= snapshot_cutoff(day):
        raise ValueError("Document creation/retrieval violates input cutoff")
    revision = int(text(document, "revisionNumber"))
    if revision < 1:
        raise ValueError("Invalid document revision")
    values = {}
    resolutions = set()
    for series in document.findall("{*}TimeSeries"):
        domains = [
            n.text
            for n in series
            if n.tag.split("}")[-1]
            in ("outBiddingZone_Domain.mRID", "inBiddingZone_Domain.mRID")
        ]
        if not domains or any(value != DOMAIN for value in domains):
            raise ValueError("Forecast geographic domain is not DK2")
        if (
            text(series, "quantity_Measure_Unit.name") != "MAW"
            or text(series, "curveType") != "A01"
        ):
            raise ValueError(
                "Only MW sequential fixed-resolution source curves are supported"
            )
        variable = (
            "load"
            if kind == "load"
            else PSR.get(series.findtext("{*}MktPSRType/{*}psrType"))
        )
        if variable is None:
            raise ValueError("Unrecognized renewable production type")
        for period in series.findall("{*}Period"):
            interval = period.find("{*}timeInterval")
            start = timestamp(text(interval, "start"))
            end = timestamp(text(interval, "end"))
            resolution = text(period, "resolution")
            if resolution not in ("PT15M", "PT30M", "PT60M"):
                raise ValueError("Unsupported source resolution")
            minutes = {"PT15M": 15, "PT30M": 30, "PT60M": 60}[resolution]
            resolutions.add(resolution)
            positions = set()
            for point in period.findall("{*}Point"):
                position = int(text(point, "position"))
                quantity = float(text(point, "quantity"))
                stamp = start + timedelta(minutes=(position - 1) * minutes)
                if (
                    position in positions
                    or position < 1
                    or stamp + timedelta(minutes=minutes) > end
                    or not np.isfinite(quantity)
                    or quantity < 0
                ):
                    raise ValueError("Duplicate, invalid or non-finite source point")
                positions.add(position)
                for quarter in range(minutes // 15):
                    key = (variable, stamp + timedelta(minutes=15 * quarter))
                    if key in values:
                        raise ValueError(
                            "Overlapping source series; resolve revisions explicitly"
                        )
                    values[key] = quantity
    slots = delivery_intervals_utc(day)
    variables = sorted({key[0] for key in values})
    if not variables or any(
        (variable, stamp) not in values for variable in variables for stamp in slots
    ):
        raise ValueError("Forecast does not cover the complete next delivery day")
    return {
        "document_id": text(document, "mRID"),
        "revision": revision,
        "document_created_at_utc": created.isoformat(),
        "model_initialization_utc": None,
        "publication_timestamp": None,
        "retrieved_at_utc": retrieved.isoformat(),
        "availability_evidence": "Successful live retrieval completed before cutoff; document creation is not publication time",
        "source": "ENTSO-E Transparency Platform",
        "dataset": expected + "/A01",
        "price_area": "DK2",
        "domain": DOMAIN,
        "units": "MW",
        "source_resolutions": sorted(resolutions),
        "alignment": "Left-hold source MW value over its PT15M/PT30M/PT60M period; no realized target inputs",
        "historical_availability_assumption": None,
        "raw_sha256": digest(raw),
        "rows": [
            {
                "variable": variable,
                "valid_start_utc": stamp.isoformat(),
                "value": values[variable, stamp],
            }
            for variable in variables
            for stamp in slots
        ],
    }


def capture(root, kind, day, client, *, token=None):
    token = token or os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise ValueError(
            "Optional integration requires ENTSOE_API_TOKEN with approved Transparency Platform API access"
        )
    slots = delivery_intervals_utc(day)
    params = {
        "documentType": "A65" if kind == "load" else "A69",
        "processType": "A01",
        "outBiddingZone_Domain" if kind == "load" else "in_Domain": DOMAIN,
        "periodStart": slots[0].strftime("%Y%m%d%H%M"),
        "periodEnd": (slots[-1] + timedelta(minutes=15)).strftime("%Y%m%d%H%M"),
    }
    for attempt in range(3):
        if datetime.now(UTC) > snapshot_cutoff(day):
            raise ValueError("Live fundamental capture is after cutoff")
        try:
            response = client.get(
                ENDPOINT, params=params | {"securityToken": token}, timeout=30
            )
        except httpx.HTTPError:
            if attempt == 2:
                raise RuntimeError(
                    "ENTSO-E transport failed after three attempts; credentials and request URLs omitted"
                ) from None
            sleep(2**attempt)
            continue
        raw = response.content
        observed = datetime.now(UTC)
        publish_bytes(root / "fundamentals" / "raw" / f"{digest(raw)}.xml", raw)
        receipt = {
            "retrieved_at_utc": observed,
            "status_code": response.status_code,
            "parameters": params,
            "raw_sha256": digest(raw),
        }
        publish_bytes(
            root
            / "fundamentals"
            / "observations"
            / f"{digest(json_bytes(receipt))}.json",
            json_bytes(receipt),
        )
        if response.status_code == 200:
            manifest = parse_vintage(raw, kind, day, observed)
            content = json_bytes(manifest)
            path = (
                root
                / "fundamentals"
                / "snapshots"
                / str(day)
                / f"{digest(content)}.json"
            )
            publish_bytes(path, content)
            return path
        if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
            raise ValueError(
                f"ENTSO-E returned HTTP {response.status_code}; inspect retained response"
            )
        wait = response.headers.get("Retry-After", str(2**attempt))
        if not wait.isdigit() or int(wait) > 60:
            raise ValueError(
                "Retry-After cannot be honored within bounded capture; retry a later scheduled capture"
            )
        sleep(int(wait))
    raise RuntimeError("ENTSO-E capture exhausted")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("load", "renewables"))
    p.add_argument("delivery_date", type=date.fromisoformat)
    p.add_argument("--root", type=Path, default=Path("data"))
    a = p.parse_args()
    with httpx.Client() as client:
        print(capture(a.root, a.kind, a.delivery_date, client))


if __name__ == "__main__":
    main()
