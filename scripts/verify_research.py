"""Reproduce the first and last saved candidate folds using the release code."""

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dk2_forecast.experiment import json_bytes
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.provenance import save_code_snapshot
from dk2_forecast.research import ResearchData
from dk2_forecast.storage import connect, digest, publish_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("prices", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    if args.prices.parent.name != report["data_version"]:
        raise ValueError("Verification requires the original price snapshot")
    config = report["selection"]
    frequency = config["retraining_days"]
    start = date.fromisoformat(report["period"]["start"])
    end = date.fromisoformat(report["period"]["end_exclusive"])
    last = start + timedelta(days=((end - start).days - 1) // frequency * frequency)
    candidate = "weekly_refit" if frequency == 7 else config["model"]
    data = ResearchData(PriceHistory.load(args.prices), args.root)
    bundle = save_code_snapshot(args.root)
    checks = []
    with connect() as db:
        for origin, stop in (
            (start, min(start + timedelta(days=frequency), end)),
            (last, end),
        ):
            current, _ = data.evaluate(
                config["family"],
                config["parameters"],
                weeks=config["training_weeks"],
                frequency=frequency,
                correction=config["correction"],
                start=origin,
                end=stop,
            )
            previous = np.asarray(
                db.execute(
                    f"SELECT {candidate} FROM read_parquet(?) WHERE CAST(timezone('Europe/Copenhagen',delivery_start_utc) AS DATE) >= ? AND CAST(timezone('Europe/Copenhagen',delivery_start_utc) AS DATE) < ? ORDER BY delivery_start_utc",
                    [str(args.report.with_name("predictions.parquet")), origin, stop],
                ).fetchall()
            )[:, 0]
            np.testing.assert_allclose(current, previous, rtol=0, atol=1e-10)
            checks.append(
                {
                    "start": str(origin),
                    "end_exclusive": str(stop),
                    "intervals": len(current),
                    "max_absolute_difference": float(np.abs(current - previous).max()),
                }
            )
    content = json_bytes(
        {
            "created_at_utc": datetime.now(UTC),
            "source_bundle_sha256": bundle,
            "checks": checks,
        }
    )
    path = args.report.parent / "verification" / f"{digest(content)}.json"
    publish_bytes(path, content)
    print(path)


if __name__ == "__main__":
    main()
