import argparse
import json
from datetime import UTC, date, datetime, time, timedelta
from importlib.metadata import version
from pathlib import Path

import numpy as np

from dk2_forecast.delivery_day import (
    COPENHAGEN,
    delivery_intervals_utc,
    forecast_issue_utc,
)
from dk2_forecast.experiment import (
    DIAGNOSTIC_END,
    FRESH_DAYS,
    Evaluation,
    date_range,
    json_bytes,
    prediction_rows,
    save_predictions,
)
from dk2_forecast.experiment_metrics import evaluate_comparison
from dk2_forecast.experiment_models import BASELINES, ComparisonTraining, fit_predict
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.issue import capture_inputs, publish_prepared, save_preparation
from dk2_forecast.price_lags import local_day
from dk2_forecast.storage import connect, digest, publish_bytes


def implementation_version():
    folder = Path(__file__).parent
    sources = (
        "delivery_day.py",
        "experiment.py",
        "experiment_metrics.py",
        "experiment_models.py",
        "forecast.py",
        "issue.py",
        "price_lags.py",
        "prospective.py",
    )
    return digest(
        b"\0".join(
            (folder / name).read_text(encoding="utf-8").encode() for name in sources
        )
    )


def freeze(report_path: Path, root: Path, start: date | None = None) -> Path:
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    now = datetime.now(COPENHAGEN)
    first_issue = now.date() + (
        timedelta(days=1) if now.time() >= time(9, 50) else timedelta(0)
    )
    start = start or max(DIAGNOSTIC_END, first_issue + timedelta(days=1))
    if start < DIAGNOSTIC_END or start <= first_issue:
        raise ValueError(
            "Choose a fresh start whose preparation time is still in the future"
        )
    plan = {
        "frozen_at_utc": now.astimezone(UTC).isoformat(),
        "start": start,
        "end_exclusive": start + timedelta(days=FRESH_DAYS),
        "fold_days": report["fold_days"],
        "development_data_version": report["data_version"],
        "development_report_sha256": digest(report_bytes),
        "baseline_versions": report["baseline_versions"],
        "features": report["features"],
        "models": report["frozen_models"],
        "development_choice": report["development_choice"],
        "versions": report["versions"],
        "scoring": report["scoring"],
        "implementation_sha256": implementation_version(),
    }
    content = json_bytes(plan)
    path = root / "experiments" / digest(content) / "freeze.json"
    publish_bytes(path, content)
    return path


def load_plan(path: Path) -> dict:
    content = path.read_bytes()
    if path.parent.name != digest(content):
        raise ValueError("Freeze directory must match the SHA-256 of freeze.json")
    plan = json.loads(content)
    if plan["implementation_sha256"] != implementation_version():
        raise ValueError("Use the original frozen experiment code")
    for package, expected in plan["versions"].items():
        if version(package) != expected:
            raise ValueError(f"Frozen experiment requires {package}=={expected}")
    return plan


def prepare(plan_path: Path, root: Path) -> Path:
    now = datetime.now(COPENHAGEN)
    day = now.date() + timedelta(days=1)
    if now.time() >= time(10):
        raise ValueError("Preparation must start before 10:00 Europe/Copenhagen")
    plan = load_plan(plan_path)
    start = date.fromisoformat(plan["start"])
    if not start <= day < date.fromisoformat(plan["end_exclusive"]):
        raise ValueError("Tomorrow is outside the frozen evaluation period")
    offset = (day - start).days
    origin = day - timedelta(days=offset % plan["fold_days"])
    out = plan_path.parent
    staged = out / "prepared" / str(day) / "forecast.parquet"
    if staged.is_file() and staged.with_suffix(".json").is_file():
        return staged
    training_lock = out / "training" / f"{origin}.json"
    if day != origin and not training_lock.is_file():
        raise ValueError(
            f"No pre-cutoff training snapshot was captured at block origin {origin}"
        )

    _, live_history, live_receipt = capture_inputs(root, day)
    if day == origin:
        publish_bytes(training_lock, json_bytes(live_receipt))
    training_receipt = json.loads(training_lock.read_bytes())
    if datetime.fromisoformat(training_receipt["observed_at_utc"]) > forecast_issue_utc(
        origin
    ):
        raise ValueError("Training inputs were observed after the training cutoff")
    train_path = root / "prices" / training_receipt["data_version"] / "prices.parquet"
    if digest(train_path.read_bytes()) != training_receipt["prices_sha256"]:
        raise ValueError("Training snapshot differs from the recorded block input")
    training = ComparisonTraining(PriceHistory.load(train_path), origin)
    x, y = training.arrays(origin)
    evaluation = Evaluation.build(live_history, [(day, day + date.resolution)])
    evaluation.blocks = [
        (origin, day + date.resolution, slice(0, len(evaluation.timestamps)))
    ]
    predictions = evaluation.baselines.copy()
    runtime = {}
    for name, config in plan["models"].items():
        predictions[name], runtime[name] = fit_predict(config, x, y, evaluation.matrix)
    save_predictions(
        staged,
        prediction_rows(evaluation, predictions, live_history.data_version),
        predictions,
    )
    save_preparation(
        staged,
        day,
        live_receipt,
        freeze_sha256=out.name,
        training_origin=str(origin),
        training_inputs=training_receipt,
        fit_predict_seconds=runtime,
    )
    return staged


def score(plan_path: Path, actuals: Path) -> Path:
    plan = load_plan(plan_path)
    history = PriceHistory.load(actuals)
    start = date.fromisoformat(plan["start"])
    end = date.fromisoformat(plan["end_exclusive"])
    stop = min(end, local_day(max(history.prices)) + date.resolution)
    if stop <= start:
        raise ValueError("The snapshot contains no actuals in the frozen period")
    methods = (*BASELINES, *plan["models"])
    values = {method: [] for method in methods}
    timestamps, fallback, issued = [], [], []
    with connect() as connection:
        for day in date_range(start, stop):
            expected = delivery_intervals_utc(day)
            issue = forecast_issue_utc(day)
            timestamps.extend(expected)
            path = plan_path.parent / "forecasts" / str(day) / "forecast.parquet"
            if not path.is_file():
                for series in values.values():
                    series.extend([None] * len(expected))
                fallback.extend([False] * len(expected))
                continue
            receipt_bytes = path.with_suffix(".json").read_bytes()
            receipt = json.loads(receipt_bytes)
            published = datetime.fromisoformat(receipt["published_at_utc"])
            if (
                receipt["freeze_sha256"] != plan_path.parent.name
                or receipt["forecast_sha256"] != digest(path.read_bytes())
                or not issue <= published < issue + timedelta(minutes=1)
                or not datetime.fromisoformat(plan["frozen_at_utc"])
                < datetime.fromisoformat(receipt["observed_at_utc"])
                <= datetime.fromisoformat(receipt["prepared_at_utc"])
                <= issue
            ):
                raise ValueError(f"Invalid prospective issuance receipt for {day}")
            rows = connection.execute(
                f"SELECT delivery_start_utc, issue_time_utc, weekly_fallback, {', '.join(methods)} "
                "FROM read_parquet(?) ORDER BY delivery_start_utc",
                [str(path)],
            ).fetchall()
            if [row[0] for row in rows] != expected or any(
                row[1] != issue for row in rows
            ):
                raise ValueError(
                    f"Forecast interval keys or issue time differ for {day}"
                )
            for _, _, used_fallback, *predictions in rows:
                fallback.append(used_fallback)
                for method, value in zip(methods, predictions, strict=True):
                    values[method].append(value)
            issued.append({"day": day, "receipt_sha256": digest(receipt_bytes)})
    if not issued:
        raise ValueError("No prospectively issued forecasts overlap these actuals")
    metrics = evaluate_comparison(
        timestamps,
        [history.prices.get(stamp) for stamp in timestamps],
        {name: np.asarray(series, dtype=float) for name, series in values.items()},
        plan["scoring"],
        fallback,
    )
    content = json_bytes(
        {
            "frozen_plan_sha256": plan_path.parent.name,
            "actuals_data_version": history.data_version,
            "start": start,
            "end_exclusive": stop,
            "full_period_actuals_available": stop == end,
            "issued_delivery_days": len(issued),
            "issued": issued,
            **metrics,
        }
    )
    path = plan_path.parent / "scores" / f"{digest(content)}.json"
    publish_bytes(path, content)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Freeze, issue, or score prospective DK2 forecasts."
    )
    parser.add_argument("--root", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="action", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("report", type=Path)
    freezing.add_argument("--start", type=date.fromisoformat)
    for action in ("prepare", "publish", "score"):
        command = commands.add_parser(action)
        command.add_argument("plan", type=Path)
        if action == "score":
            command.add_argument("actuals", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        path = freeze(args.report, args.root, args.start)
    elif args.action == "prepare":
        path = prepare(args.plan, args.root)
    elif args.action == "publish":
        load_plan(args.plan)
        path = publish_prepared(args.plan.parent, datetime.now(UTC))
    else:
        path = score(args.plan, args.actuals)
    print(path)


if __name__ == "__main__":
    main()
