"""Explicitly selected research evaluations and plan-derived scheduler definitions."""

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from dk2_forecast.delivery_day import COPENHAGEN
from dk2_forecast.evaluation import (
    aggregate,
    earliest_origin,
    freeze,
    load_evaluation,
    restart,
)
from dk2_forecast.ingestion import FIRST_DAY, ingest_prices
from dk2_forecast.research_forecast import forecast, publish


def schedule_spec(plan_path, project, *, weather=False):
    plan_path, project = Path(plan_path).resolve(), Path(project).resolve()
    plan = json.loads(plan_path.read_bytes())
    research = "report_sha256" in plan or plan.get("schema_version") == 2
    if research:
        plan = load_evaluation(plan_path)
    identity = plan.get("evaluation_id", plan_path.parent.name)
    start = date.fromisoformat(plan["start"]) - timedelta(days=1)
    last = date.fromisoformat(plan["end_exclusive"]) - timedelta(days=2)
    policy = plan.get("issue_policy", {})
    definitions = [
        ("prepare", policy.get("prepare", "09:40" if research else "09:50")),
        ("publish", policy.get("publish", "10:00")),
        ("collect-score", policy.get("actuals_score", "14:30")),
    ]
    if weather:
        definitions.append(("weather", "09:30"))
    tasks = []
    for action, clock in definitions:
        if research and plan.get("schema_version") == 2:
            arguments = f'-I "{plan_path.parent / "launch.py"}" {action} "{plan_path}" --root "{project / "data"}"'
        else:
            module = (
                "dk2_forecast.research_ops"
                if research or action in ("collect-score", "weather")
                else "dk2_forecast.operations"
            )
            arguments = (
                f'-m {module} {action} "{plan_path}" --root "{project / "data"}"'
            )
        tasks.append(
            {
                "name": f"DK2 {'Research' if research else 'Original'} {identity} {action}",
                "action": action,
                "execute": str(project / ".venv" / "python.exe"),
                "arguments": arguments,
                "directory": str(project),
                "start_boundary": f"{start}T{clock}:00",
                "end_boundary": f"{last + timedelta(days=1)}T00:00:00",
                "local_time": clock,
            }
        )
    return {
        "evaluation_id": identity,
        "type": "research" if research else "original",
        "timezone": "Europe/Copenhagen",
        "first_issue_date": str(start),
        "last_issue_date": str(last),
        "tasks": tasks,
    }


def worker(action, plan_path, root, *, day=None, actuals=None):
    plan_path, root = Path(plan_path), Path(root)
    raw = json.loads(plan_path.read_bytes())
    now = datetime.now(UTC)
    if "models" in raw:
        if action == "collect-score":
            from dk2_forecast.prospective import score

            stop = min(
                date.fromisoformat(raw["end_exclusive"]),
                now.astimezone(COPENHAGEN).date() + timedelta(days=2),
            )
            with httpx.Client() as client:
                prices = ingest_prices(root, FIRST_DAY, stop, client)
            return score(plan_path, prices)
        if action == "weather":
            from dk2_forecast.explanatory import capture

            with httpx.Client() as client:
                return capture(
                    root, now.astimezone(COPENHAGEN).date() + timedelta(days=1), client
                )
        raise ValueError("Use operations for original experiment actions")
    plan = load_evaluation(plan_path)
    report = plan_path.with_name("report.json")
    if plan.get("schema_version") != 2 and not report.is_file():
        report = root / "research" / plan["report_sha256"] / "report.json"
    day = day or now.astimezone(COPENHAGEN).date() + timedelta(days=1)
    if action == "prepare":
        return forecast(report, None, day, root, replay=False, evaluation=plan_path)
    if action == "publish":
        return publish(
            root,
            report,
            day,
            evaluation=plan_path if plan.get("schema_version") == 2 else None,
        )
    if action == "weather":
        from dk2_forecast.explanatory import capture

        with httpx.Client() as client:
            return capture(root, day, client)
    if action == "collect-score":
        stop = min(
            date.fromisoformat(plan["end_exclusive"]),
            now.astimezone(COPENHAGEN).date() + timedelta(days=2),
        )
        with httpx.Client() as client:
            actuals = ingest_prices(root, FIRST_DAY, stop, client)
    return aggregate(plan_path, root, actuals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "freeze",
            "restart",
            "prepare",
            "publish",
            "inspect",
            "score",
            "collect-score",
            "weather",
            "schedule-spec",
        ),
    )
    parser.add_argument(
        "plan",
        type=Path,
        help="Explicit freeze.json; for freeze, the existing research report",
    )
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--days", type=int, default=56)
    parser.add_argument("--evaluation-version", type=int, default=1)
    parser.add_argument("--day", type=date.fromisoformat)
    parser.add_argument("--actuals", type=Path)
    parser.add_argument("--weather", action="store_true")
    args = parser.parse_args()
    if args.action in ("freeze", "restart"):
        print(
            (freeze if args.action == "freeze" else restart)(
                args.plan,
                args.root,
                args.start or earliest_origin(),
                days=args.days,
                evaluation_version=args.evaluation_version,
            )
        )
    elif args.action == "schedule-spec":
        print(json.dumps(schedule_spec(args.plan, Path.cwd(), weather=args.weather)))
    else:
        if args.action == "score" and args.actuals is None:
            parser.error("score requires --actuals; use inspect without actuals")
        if (
            args.action in ("prepare", "publish", "weather", "collect-score")
            and json.loads(args.plan.read_bytes()).get("schema_version") == 2
        ):
            parser.error(
                "Run this evaluation's retained launch.py for operational actions"
            )
        print(
            json.dumps(
                worker(
                    args.action,
                    args.plan,
                    args.root,
                    day=args.day,
                    actuals=args.actuals,
                ),
                default=str,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
