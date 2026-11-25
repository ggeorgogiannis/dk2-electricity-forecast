"""Assess fixed research candidates on the already inspected diagnostic period."""

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

from dk2_forecast.experiment import json_bytes
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.research import ResearchData
from dk2_forecast.research_metrics import point_metrics
from dk2_forecast.storage import digest, publish_bytes


def run(report_path, prices, root):
    report = json.loads(report_path.read_bytes())
    data = ResearchData(PriceHistory.load(prices), root)
    start, end = date(2026, 9, 21), date(2026, 11, 16)
    mask = (data.days >= start) & (data.days < end)
    predictions = {
        "previous_day": data.matrix[mask, 0],
        "weekly_utc": data.matrix[mask, 1],
        "weekday_persistence": data.seasonal[mask],
    }
    fits = {}
    for name in (
        "ridge_tuned",
        "lightgbm_tuned",
        "catboost_tuned",
        "lightgbm_squared",
        "correction",
    ):
        config = report["fits"][name]
        predictions[name], fits[name] = data.evaluate(
            config["family"],
            config["parameters"],
            weeks=config["weeks"],
            frequency=report["selection"]["retraining_days"],
            correction=config["correction"],
            start=start,
            end=end,
        )
    timestamps = [
        stamp for stamp, selected in zip(data.timestamps, mask, strict=True) if selected
    ]
    content = json_bytes(
        {
            "status": "previously_observed_diagnostic",
            "created_at_utc": datetime.now(UTC),
            "research_report_sha256": digest(report_path.read_bytes()),
            "start": start,
            "end_exclusive": end,
            "fits": fits,
            "point": point_metrics(
                timestamps, data.actual[mask], predictions, report["scoring"]
            ),
            "data_version": prices.parent.name,
        }
    )
    output = report_path.parent / "diagnostics" / f"{digest(content)}.json"
    publish_bytes(output, content)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("prices", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(run(args.report, args.prices, args.root))


if __name__ == "__main__":
    main()
