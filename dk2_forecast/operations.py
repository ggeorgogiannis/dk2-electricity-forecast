"""Audit and log the frozen experiment without altering its implementation."""

import argparse
import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from dk2_forecast.delivery_day import delivery_intervals_utc, forecast_issue_utc
from dk2_forecast.experiment import json_bytes
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.prospective import load_plan, score
from dk2_forecast.provenance import save_code_snapshot
from dk2_forecast.storage import connect, digest, publish_bytes


@contextmanager
def exclusive_run(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"created_at_utc": datetime.now(UTC).isoformat()}))
    try:
        yield
    finally:
        path.unlink()


def inspect_day(plan_path: Path, day: date) -> dict:
    plan = load_plan(plan_path)
    issue = forecast_issue_utc(day)
    now = datetime.now(UTC)
    path = plan_path.parent / "forecasts" / str(day) / "forecast.parquet"
    result = {
        "checked_at_utc": now.isoformat(),
        "delivery_date": str(day),
        "scheduled_issue_utc": issue.isoformat(),
        "primary_benchmark": "previous_day",
        "expected_intervals": len(delivery_intervals_utc(day)),
        "state": "pending" if now < issue + timedelta(minutes=1) else "missed",
        "eligible": False,
        "coverage": {name: 0.0 for name in ("weekly", "previous_day", *plan["models"])},
    }
    if path.is_file():
        try:
            receipt = json.loads(path.with_suffix(".json").read_bytes())
            published = datetime.fromisoformat(receipt["published_at_utc"])
            observed = datetime.fromisoformat(receipt["observed_at_utc"])
            prepared = datetime.fromisoformat(receipt["prepared_at_utc"])
            if receipt["freeze_sha256"] != plan_path.parent.name:
                raise ValueError("Publication references another experiment")
            if digest(path.read_bytes()) != receipt["forecast_sha256"]:
                raise ValueError("Forecast hash mismatch")
            snapshot = (
                plan_path.parents[2]
                / "prices"
                / receipt["data_version"]
                / "prices.parquet"
            )
            if digest(snapshot.read_bytes()) != receipt["prices_sha256"]:
                raise ValueError("Input snapshot hash mismatch")
            PriceHistory.load(snapshot)
            methods = ("weekly", "previous_day", *plan["models"])
            with connect() as db:
                query = db.execute(
                    "SELECT * FROM read_parquet(?) ORDER BY delivery_start_utc",
                    [str(path)],
                )
                columns = [column[0] for column in query.description]
                rows = [
                    dict(zip(columns, row, strict=True)) for row in query.fetchall()
                ]
            if [row["delivery_start_utc"] for row in rows] != delivery_intervals_utc(
                day
            ):
                raise ValueError("Missing or duplicate UTC delivery keys")
            if any(
                row["price_area"] != "DK2" or row["issue_time_utc"] != issue
                for row in rows
            ):
                raise ValueError("Price area or scheduled issue mismatch")
            result["coverage"] = {
                name: sum(row[name] is not None for row in rows) / len(rows)
                for name in methods
            }
            result["receipt"] = receipt
            result["input_verification_age_at_issue_seconds"] = (
                issue - observed
            ).total_seconds()
            result["eligible"] = (
                datetime.fromisoformat(plan["frozen_at_utc"])
                < observed
                <= prepared
                <= issue
                and issue <= published < issue + timedelta(minutes=1)
            )
            result["state"] = "issued" if result["eligible"] else "late"
        except (OSError, ValueError, KeyError) as error:
            result.update(state="invalid", error=str(error))
    start = date.fromisoformat(plan["start"])
    origin = day - timedelta(days=(day - start).days % plan["fold_days"])
    if not (plan_path.parent / "training" / f"{origin}.json").is_file():
        result["training_block_status"] = "missing_origin_snapshot"
        result["next_block_origin"] = str(origin + timedelta(days=plan["fold_days"]))
    content = json_bytes(result)
    output = plan_path.parent / "operations" / str(day) / f"{digest(content)}.json"
    publish_bytes(output, content)
    return result


def run_action(plan: Path, action: str, root: Path) -> int:
    code_snapshot = save_code_snapshot(root)
    now = datetime.now(UTC)
    output = plan.parent / "logs" / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{action}"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "dk2_forecast.prospective",
        "--root",
        str(root),
        action,
        str(plan),
    ]
    with exclusive_run(plan.parent / "locks" / f"{action}.lock"):
        with output.with_suffix(".log").open("wb") as log:
            completed = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False
            )
    publish_bytes(
        output.with_suffix(".json"),
        json_bytes(
            {
                "started_at_utc": now,
                "finished_at_utc": datetime.now(UTC),
                "exit_code": completed.returncode,
                "code_snapshot_sha256": code_snapshot,
                "command": command,
                "input_policy": "Snapshot captured during preparation; inputs verified no later than issue cutoff",
            }
        ),
    )
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and log frozen live DK2 runs."
    )
    parser.add_argument("action", choices=("inspect", "prepare", "publish", "score"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("--day", type=date.fromisoformat, default=date(2026, 11, 18))
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--actuals", type=Path)
    args = parser.parse_args()
    if args.action == "inspect":
        print(json.dumps(inspect_day(args.plan, args.day), indent=2, default=str))
    elif args.action == "score":
        if args.actuals is None:
            parser.error("--actuals is required for scoring")
        print(score(args.plan, args.actuals))
    else:
        sys.exit(run_action(args.plan.resolve(), args.action, args.root.resolve()))


if __name__ == "__main__":
    main()
