"""Replay a frozen comparison with explicit creation time and diagnostic status."""

import argparse
import sys
from datetime import UTC, date, datetime
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dk2_forecast.experiment import (
    Evaluation,
    json_bytes,
    prediction_rows,
    save_predictions,
)
from dk2_forecast.experiment_models import ComparisonTraining
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.prospective import load_plan
from dk2_forecast.storage import digest, publish_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("prices", type=Path)
    parser.add_argument("day", type=date.fromisoformat)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    plan = load_plan(args.plan)
    history = PriceHistory.load(args.prices)
    training = ComparisonTraining(history, args.day)
    evaluation = Evaluation.build(history, [(args.day, args.day + date.resolution)])
    predictions, fits = {}, {}
    for name, config in plan["models"].items():
        predictions[name], fits[name] = evaluation.fit(training, config)
    created = datetime.now(UTC)
    folder = (
        args.root
        / "replays"
        / args.plan.parent.name
        / created.strftime("%Y%m%dT%H%M%S%fZ")
    )
    combined = evaluation.baselines | predictions
    path = folder / "forecast.parquet"
    save_predictions(
        path, prediction_rows(evaluation, combined, history.data_version), combined
    )
    report = {
        "status": "historical_replay",
        "replay": True,
        "created_at_utc": created,
        "delivery_date": args.day,
        "frozen_plan_sha256": args.plan.parent.name,
        "data_version": history.data_version,
        "prices_sha256": digest(args.prices.read_bytes()),
        "forecast_sha256": digest(path.read_bytes()),
        "fits": fits,
        "historical_assumption": "Archived delivery prices through the issue day assumed known at cutoff; captured today and excluded from prospective scores",
        "metrics": evaluation.score(predictions, plan["scoring"]),
    }
    publish_bytes(path.with_suffix(".json"), json_bytes(report))
    print(path.with_suffix(".json"))


if __name__ == "__main__":
    main()
