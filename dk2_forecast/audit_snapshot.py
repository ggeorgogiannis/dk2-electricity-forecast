"""Compare a validated Parquet snapshot with its original raw responses."""

import argparse
import json
from datetime import date
from pathlib import Path

from dk2_forecast.ingestion import MAX_CAPTURE_DAYS, validate_prices
from dk2_forecast.storage import connect, digest


def verify_snapshot(path: Path, root: Path):
    manifest_raw = path.with_name("manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if digest(manifest_raw) != path.parent.name:
        raise ValueError("Manifest SHA-256 mismatch")
    start, end = map(date.fromisoformat, (manifest["start"], manifest["end_exclusive"]))
    rows = []
    for entry in manifest["raw"]:
        raw_path = (root / "raw" / entry["file"]).resolve()
        if not raw_path.is_relative_to((root / "raw").resolve()):
            raise ValueError("Raw reference escapes the data root")
        raw = raw_path.read_bytes()
        if digest(raw) != entry["sha256"]:
            raise ValueError(f"Raw response SHA-256 mismatch at {raw_path}")
        stop = min(end, start + MAX_CAPTURE_DAYS)
        rows.extend(validate_prices(raw, start, stop))
        start = stop
    if start != end:
        raise ValueError("Manifest does not cover the full requested range")
    with connect() as db:
        actual = db.execute(
            "SELECT price_area, delivery_start_utc, price_eur_per_mwh FROM read_parquet(?) ORDER BY delivery_start_utc",
            [str(path)],
        ).fetchall()
    expected = [("DK2", row.time_utc, row.price_eur_per_mwh) for row in rows]
    if actual != expected or manifest["rows"] != len(rows):
        raise ValueError("Parquet data differs from validated raw responses")
    return {
        "data_version": path.parent.name,
        "rows": len(rows),
        "parquet_sha256": digest(path.read_bytes()),
        "raw_hashes_verified": len(manifest["raw"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify prices against their raw response hashes and values."
    )
    parser.add_argument("prices", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(verify_snapshot(args.prices, args.root), indent=2))


if __name__ == "__main__":
    main()
