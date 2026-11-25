"""Research candidate identities, immutable evaluation windows and full coverage."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from importlib.metadata import version
from math import lcm

import numpy as np
import duckdb

from dk2_forecast.delivery_day import (
    COPENHAGEN,
    delivery_intervals_utc,
    forecast_issue_utc,
)
from dk2_forecast.experiment import json_bytes
from dk2_forecast.explanatory import snapshot_cutoff
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.provenance import save_code_snapshot
from dk2_forecast.research_metrics import point_metrics, interval_metrics
from dk2_forecast.storage import digest, publish_bytes, connect

METHODS = ("previous_day", "weekly", "prediction")
IDENTITY_FIELDS = (
    "schema_version",
    "candidate_id",
    "start",
    "end_exclusive",
    "issue_policy",
    "scoring",
    "evaluation_version",
)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def earliest_origin(now=None):
    now = now or datetime.now(UTC)
    day = now.astimezone(COPENHAGEN).date() + timedelta(days=1)
    # Leave ten minutes before the scheduled 09:40 start to register and inspect tasks.
    if now >= snapshot_cutoff(day) - timedelta(minutes=20):
        day += timedelta(days=1)
    return day


def freeze(report_path, root, start=None, *, days=56, evaluation_version=1, now=None):
    now = now or datetime.now(UTC)
    start = start or earliest_origin(now)
    if now >= snapshot_cutoff(start) - timedelta(minutes=10):
        raise ValueError(
            "Freeze before the first 09:40 preparation; choose a future origin"
        )
    if not 1 <= days <= 366:
        raise ValueError("Evaluation length must be 1..366 days")
    report_raw = Path(report_path).read_bytes()
    report = json.loads(report_raw)
    from dk2_forecast.research_forecast import current_code_hash

    code = save_code_snapshot(root)
    archive = root / "code" / code / "source.zip"
    candidate = {
        "schema_version": 2,
        "selection": report["selection"],
        "features": report["features"],
        "implementation_sha256": current_code_hash(),
        "code_snapshot_sha256": code,
        "archive_sha256": digest(archive.read_bytes()),
        "launcher_sha256": digest(
            (Path(__file__).parents[1] / "scripts" / "run_evaluation.py").read_bytes()
        ),
        "versions": {
            line.split("==")[0]: version(line.split("==")[0])
            for line in (Path(__file__).parents[1] / "requirements-lock.txt")
            .read_text()
            .splitlines()
            if "==" in line and not line.startswith("#")
        },
        "quantile_parameters": report["fits"]["lightgbm_tuned"]["parameters"],
        "quantile_retraining_days": 28,
        "quantile_calibration": {
            "policy": "fixed_last_development_margin",
            "margin": report["calibration"][-1]["adjustment"]
            if report["calibration"]
            else 0.0,
        },
        "training_constants": {"seed": 42, "threads": 1, "iterations": 160},
        "previous_day_policy": "local-clock-earliest-UTC-weekly-UTC-fallback-v1",
    }
    candidate_id = digest(json_bytes(candidate))
    candidate_path = (
        root / "research_live" / "candidates" / candidate_id / "candidate.json"
    )
    if publish_bytes(candidate_path, json_bytes(candidate)) != json_bytes(candidate):
        raise ValueError("Candidate identity conflict")
    plan = {
        "schema_version": 2,
        "candidate_id": candidate_id,
        "start": str(start),
        "end_exclusive": str(start + timedelta(days=days)),
        "issue_policy": {
            "timezone": "Europe/Copenhagen",
            "prepare": "09:40",
            "input_cutoff": "09:50",
            "publish": "10:00",
            "window_seconds": 60,
            "actuals_score": "14:30",
        },
        "scoring": report["scoring"],
        "evaluation_version": evaluation_version,
    }
    evaluation_id = digest(json_bytes(plan))
    plan |= {
        "evaluation_id": evaluation_id,
        "created_at_utc": now.isoformat(),
        "report_sha256": digest(report_raw),
        "candidate": candidate,
        "primary_benchmark": "previous_day",
    }
    folder = root / "research_live" / "evaluations" / evaluation_id
    path = folder / "freeze.json"
    publish_bytes(path, json_bytes(plan))
    publish_bytes(folder / "report.json", report_raw)
    launcher = Path(__file__).parents[1] / "scripts" / "run_evaluation.py"
    publish_bytes(folder / "launch.py", launcher.read_bytes())
    load_evaluation(path)
    return path


def load_evaluation(path):
    path = Path(path)
    if path.is_dir():
        path /= "freeze.json"
    plan = read_json(path)
    if plan.get("schema_version") != 2:
        # Legacy plans remain read-only and keep their original directory identity.
        if "report_sha256" in plan and "selection" in plan:
            return plan
        raise ValueError("Select a research evaluation freeze.json explicitly")
    identity = digest(json_bytes({key: plan[key] for key in IDENTITY_FIELDS}))
    if plan["evaluation_id"] != identity or path.parent.name != identity:
        raise ValueError("Evaluation identity mismatch")
    if digest(json_bytes(plan["candidate"])) != plan["candidate_id"]:
        raise ValueError("Candidate identity mismatch")
    report = path.with_name("report.json")
    if digest(report.read_bytes()) != plan["report_sha256"]:
        raise ValueError("Frozen report hash mismatch")
    recorded = read_json(report)
    if (
        recorded["selection"] != plan["candidate"]["selection"]
        or recorded["features"] != plan["candidate"]["features"]
        or recorded["scoring"] != plan["scoring"]
    ):
        raise ValueError("Frozen report and candidate configuration disagree")
    if date.fromisoformat(plan["start"]) >= date.fromisoformat(plan["end_exclusive"]):
        raise ValueError("Empty evaluation window")
    return plan


def restart(plan_path, root, start=None, *, days=56, evaluation_version=1, now=None):
    """Start another period using exactly the retained candidate, even after checkout edits."""
    plan_path = Path(plan_path)
    previous = load_evaluation(plan_path)
    if previous.get("schema_version") != 2:
        raise ValueError(
            "Legacy evaluations remain read-only; freeze a schema-v2 candidate first"
        )
    now = now or datetime.now(UTC)
    start = start or earliest_origin(now)
    if now >= snapshot_cutoff(start) - timedelta(minutes=10) or not 1 <= days <= 366:
        raise ValueError(
            "Choose a future origin before 09:40 and a length of 1..366 days"
        )
    candidate = previous["candidate"]
    archive = root / "code" / candidate["code_snapshot_sha256"] / "source.zip"
    if digest(archive.read_bytes()) != candidate["archive_sha256"]:
        raise ValueError("Retained candidate archive missing or corrupt")
    plan = {key: previous[key] for key in IDENTITY_FIELDS}
    plan |= {
        "start": str(start),
        "end_exclusive": str(start + timedelta(days=days)),
        "evaluation_version": evaluation_version,
    }
    identity = digest(json_bytes(plan))
    plan |= {
        "evaluation_id": identity,
        "created_at_utc": now.isoformat(),
        "report_sha256": previous["report_sha256"],
        "candidate": candidate,
        "primary_benchmark": "previous_day",
    }
    folder = root / "research_live" / "evaluations" / identity
    path = folder / "freeze.json"
    publish_bytes(path, json_bytes(plan))
    publish_bytes(
        folder / "report.json", plan_path.with_name("report.json").read_bytes()
    )
    publish_bytes(folder / "launch.py", plan_path.with_name("launch.py").read_bytes())
    load_evaluation(path)
    return path


def origin_for(plan, day, *, quantile=False):
    start = date.fromisoformat(plan["start"])
    candidate = plan.get("candidate", plan)
    frequency = (
        candidate["quantile_retraining_days"]
        if quantile
        else candidate["selection"]["retraining_days"]
    )
    return day - timedelta(days=(day - start).days % frequency)


def input_history(root, receipt, cutoff):
    observed = datetime.fromisoformat(receipt["observed_at_utc"])
    if observed.tzinfo is None or observed > cutoff:
        raise ValueError("Input retrieval/verification finished after cutoff")
    data_version = receipt["data_version"]
    if len(data_version) != 64 or any(
        c not in "0123456789abcdef" for c in data_version
    ):
        raise ValueError("Invalid input version")
    path = root / "prices" / data_version / "prices.parquet"
    if digest(path.read_bytes()) != receipt["prices_sha256"]:
        raise ValueError("Input snapshot hash mismatch")
    return PriceHistory.load(path)


def validate_publication(plan_path, root, day, *, prepared=False):
    plan = load_evaluation(plan_path)
    folder = Path(plan_path).parent
    path = (
        folder
        / ("prepared" if prepared else "forecasts")
        / str(day)
        / "forecast.parquet"
    )
    receipt = read_json(path.with_suffix(".json"))
    candidate = plan.get("candidate", plan)
    if receipt["replay"] or receipt["status"] != ("prepared" if prepared else "issued"):
        raise ValueError("Replay or incorrect publication status")
    if plan.get("schema_version") == 2:
        if (
            receipt["evaluation_id"] != plan["evaluation_id"]
            or receipt["candidate_id"] != plan["candidate_id"]
        ):
            raise ValueError("Publication evaluation/candidate mismatch")
        if receipt["code_snapshot_sha256"] != candidate["code_snapshot_sha256"]:
            raise ValueError("Execution source bundle mismatch")
    if receipt["configuration_sha256"] != plan["report_sha256"] or receipt[
        "code_sha256"
    ] != candidate.get("implementation_sha256", candidate.get("code_sha256")):
        raise ValueError("Publication configuration/implementation mismatch")
    if digest(path.read_bytes()) != receipt["forecast_sha256"]:
        raise ValueError("Forecast hash mismatch")
    issue = forecast_issue_utc(day)
    created = datetime.fromisoformat(receipt["created_at_utc"])
    frozen = datetime.fromisoformat(plan["created_at_utc"])
    if (
        not frozen <= created <= issue
        or datetime.fromisoformat(receipt["scheduled_issue_utc"]) != issue
        or datetime.fromisoformat(receipt["data_cutoff_utc"]) != snapshot_cutoff(day)
    ):
        raise ValueError("Incorrect issue/cutoff/creation timing")
    if not prepared:
        published = datetime.fromisoformat(receipt["published_at_utc"])
        if not issue <= published < issue + timedelta(seconds=60):
            raise ValueError("Late publication")
    input_history(root, receipt["input"], snapshot_cutoff(day))
    for quantile, field in [
        (False, "training_origin"),
        (True, "quantile_training_origin"),
    ]:
        origin = origin_for(plan, day, quantile=quantile)
        if receipt[field] != str(origin):
            raise ValueError("Training origin mismatch")
        training = read_json(folder / "training" / f"{origin}.json")
        input_history(root, training, snapshot_cutoff(origin))
        expected_hash = digest(json_bytes(training))
        if (
            plan.get("schema_version") == 2
            and receipt[
                "quantile_training_receipt_sha256"
                if quantile
                else "training_receipt_sha256"
            ]
            != expected_hash
        ):
            raise ValueError("Training receipt mismatch")
    if receipt["units"] != "EUR/MWh" or receipt["stale_input"]:
        raise ValueError("Wrong units or stale input")
    if receipt["configuration"] != candidate["selection"]:
        raise ValueError("Model configuration mismatch")
    if datetime.fromisoformat(receipt["input"]["observed_at_utc"]) > created:
        raise ValueError("Inputs observed after forecast creation")
    if not isinstance(receipt["fallback"], bool) or (
        receipt["fallback"] and not receipt["fallback_reason"]
    ):
        raise ValueError("Fallback reason missing")
    if not isinstance(receipt["quantile_fallback"], bool) or (
        receipt["quantile_fallback"] and not receipt["quantile_fallback_reason"]
    ):
        raise ValueError("Quantile fallback reason missing")
    if not isinstance(
        receipt["previous_day_weekly_fallbacks"], int
    ) or not 0 <= receipt["previous_day_weekly_fallbacks"] <= len(
        delivery_intervals_utc(day)
    ):
        raise ValueError("Invalid baseline fallback count")
    with connect() as db:
        result = db.execute(
            "SELECT * FROM read_parquet(?) ORDER BY delivery_start_utc", [str(path)]
        )
        columns = [column[0] for column in result.description]
        rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
    if [row["delivery_start_utc"] for row in rows] != delivery_intervals_utc(day):
        raise ValueError("Missing, duplicate or wrong delivery timestamps")
    if any(
        row["price_area"] != "DK2"
        or row["delivery_date"] != day
        or row["issue_time_utc"] != issue
        or row["model_version"] != receipt["model_version"]
        or row["data_version"] != receipt["input"]["data_version"]
        for row in rows
    ):
        raise ValueError("Forecast row provenance mismatch")
    if not all(
        np.isfinite([row[name] for name in (*METHODS, "q10", "q50", "q90")]).all()
        and row["q10"] <= row["q50"] <= row["q90"]
        for row in rows
    ):
        raise ValueError("Non-finite values or crossing operational quantiles")
    if receipt["validation"] != {"complete": True, "unique_intervals": len(rows)}:
        raise ValueError("Completeness receipt mismatch")
    if plan.get("schema_version") == 2:
        raw = np.asarray(
            [[row[name] for name in ("raw_q10", "raw_q50", "raw_q90")] for row in rows]
        )
        if (
            not np.isfinite(raw).all()
            or int((np.diff(raw, axis=1) < 0).any(axis=1).sum())
            != receipt["raw_quantile_crossings"]
        ):
            raise ValueError("Raw quantile diagnostics mismatch")
    if not receipt["fallback"]:
        model = root / "models" / f"research-{receipt['model_version']}.pkl"
        if digest(model.read_bytes()) != receipt["model_version"]:
            raise ValueError("Point model artifact hash mismatch")
    for model_hash in receipt["quantile_models"].values():
        if (
            digest((root / "models" / f"quantile-{model_hash}.txt").read_bytes())
            != model_hash
        ):
            raise ValueError("Quantile model artifact hash mismatch")
    return rows, receipt


def aggregate(plan_path, root, actuals=None, *, now=None, save=True):
    plan_path = Path(plan_path)
    plan = load_evaluation(plan_path)
    now = now or datetime.now(UTC)
    start, end = (date.fromisoformat(plan[key]) for key in ("start", "end_exclusive"))
    history = PriceHistory.load(actuals) if actuals else None
    actual_values = history.prices if history else {}
    states = []
    stamps, actual, predictions, model_flags = (
        [],
        [],
        {name: [] for name in METHODS},
        [],
    )
    quantiles = []
    expected = due = available = eligible_count = 0
    fallback_reasons = {}
    quantile_fallback_reasons = {}
    previous_day_weekly_fallbacks = 0
    for index in range((end - start).days):
        day = start + timedelta(days=index)
        slots = delivery_intervals_utc(day)
        expected += len(slots)
        is_due = now >= forecast_issue_utc(day) + timedelta(minutes=1)
        due += len(slots) if is_due else 0
        observed = int(
            sum(np.isfinite(actual_values.get(stamp, np.nan)) for stamp in slots)
        )
        available += observed
        state = {
            "delivery_date": str(day),
            "expected_intervals": len(slots),
            "actual_intervals": observed,
            "state": "missed_publication" if is_due else "not_yet_due",
            "forecast_intervals": 0,
        }
        path = plan_path.parent / "forecasts" / str(day)
        origin = origin_for(plan, day)
        quantile_origin = origin_for(plan, day, quantile=True)
        candidate = plan.get("candidate", plan)
        missing_point = not (
            plan_path.parent / "training" / f"{origin}.json"
        ).is_file() and now >= snapshot_cutoff(origin)
        missing_quantile = not (
            plan_path.parent / "training" / f"{quantile_origin}.json"
        ).is_file() and now >= snapshot_cutoff(quantile_origin)
        if missing_quantile:
            shared_days = lcm(
                candidate["selection"]["retraining_days"],
                candidate["quantile_retraining_days"],
            )
            next_origin = start + timedelta(
                days=((day - start).days // shared_days + 1) * shared_days
            )
            state |= {
                "training_state": "missing_quantile_block_origin_receipt",
                "missing_quantile_origin": str(quantile_origin),
                "missing_point_origin": str(origin) if missing_point else None,
                "next_origin": str(next_origin),
                "recovery": "Create a new future evaluation of the retained candidate, or wait for the next shared point/quantile origin. Never backdate training.",
            }
        elif not (
            plan_path.parent / "training" / f"{origin}.json"
        ).is_file() and now >= snapshot_cutoff(origin):
            state |= {
                "training_state": "missing_block_origin_receipt",
                "recovery": "Create a new future evaluation of the same candidate, or wait for the next training origin; never edit this freeze.",
                "next_origin": str(
                    origin
                    + timedelta(
                        days=plan.get("candidate", plan)["selection"]["retraining_days"]
                    )
                ),
            }
        if path.exists():
            try:
                rows, receipt = validate_publication(plan_path, root, day)
                state |= {
                    "state": "scored_publication"
                    if observed == len(slots)
                    else "published_awaiting_complete_actuals",
                    "publication_state": "eligible_publication",
                    "forecast_intervals": len(slots),
                    "fallback": receipt["fallback"],
                    "fallback_reason": receipt["fallback_reason"],
                }
                eligible_count += len(slots)
                if receipt["fallback"]:
                    reason = receipt["fallback_reason"]
                    fallback_reasons[reason] = fallback_reasons.get(reason, 0) + len(
                        slots
                    )
                previous_day_weekly_fallbacks += receipt[
                    "previous_day_weekly_fallbacks"
                ]
                if receipt["quantile_fallback"]:
                    reason = receipt["quantile_fallback_reason"]
                    if not reason:
                        raise ValueError("Quantile fallback reason missing")
                    quantile_fallback_reasons[reason] = quantile_fallback_reasons.get(
                        reason, 0
                    ) + len(slots)
                if observed == len(slots):
                    stamps.extend(slots)
                    actual.extend(actual_values[stamp] for stamp in slots)
                    for name in METHODS:
                        predictions[name].extend(row[name] for row in rows)
                    quantiles.extend(
                        [[row[name] for name in ("q10", "q50", "q90")] for row in rows]
                    )
                    model_flags.extend([not receipt["fallback"]] * len(slots))
            except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as error:
                state |= {
                    "state": "invalid_or_late_publication",
                    "failure_reason": str(error),
                }
        states.append(state)
    metrics = (
        point_metrics(stamps, actual, predictions, plan["scoring"]) if stamps else None
    )
    model_indices = np.flatnonzero(model_flags)
    model_only = (
        point_metrics(
            [stamps[i] for i in model_indices],
            np.asarray(actual)[model_indices],
            {k: np.asarray(v)[model_indices] for k, v in predictions.items()},
            plan["scoring"],
        )
        if len(model_indices)
        else None
    )
    completed = (
        now >= forecast_issue_utc(end - timedelta(days=1)) + timedelta(minutes=1)
        and available == expected
        and not any(s["state"] == "published_awaiting_complete_actuals" for s in states)
    )
    result = {
        "schema_version": 2,
        "evaluation_id": plan.get("evaluation_id", plan_path.parent.name),
        "candidate_id": plan.get("candidate_id"),
        "created_at_utc": now.isoformat(),
        "status": "completed" if completed else "partial",
        "expected_intervals": expected,
        "due_intervals": due,
        "actual_price_intervals": available,
        "actual_price_coverage": available / expected,
        "coverage": {
            name: {
                "forecast_intervals": eligible_count,
                "full_window_share": eligible_count / expected,
                "due_share": eligible_count / due if due else None,
            }
            for name in METHODS
        },
        "common_paired_intervals": len(stamps),
        "common_paired_full_window_share": len(stamps) / expected,
        "days": states,
        "operational_point": metrics,
        "model_only_point": model_only,
        "model_only_intervals": len(model_indices),
        "fallback_intervals": sum(fallback_reasons.values()),
        "fallback_reasons": fallback_reasons,
        "quantile_fallback_reasons": quantile_fallback_reasons,
        "quantile_fallback_intervals": sum(quantile_fallback_reasons.values()),
        "previous_day_weekly_fallbacks": previous_day_weekly_fallbacks,
        "fallback_share_of_published": sum(fallback_reasons.values()) / eligible_count
        if eligible_count
        else None,
        "fallback_point": None,
        "intervals": interval_metrics(stamps, actual, quantiles) if stamps else None,
        "actuals_data_version": history.data_version if history else None,
        "actuals_sha256": digest(Path(actuals).read_bytes()) if actuals else None,
        "denominator_policy": "Forecast coverage includes missed/invalid publications; accuracy uses complete actual days and identical valid UTC intervals. Future days remain in full-window coverage, separately from due coverage.",
    }
    fallback_indices = np.flatnonzero(~np.asarray(model_flags, dtype=bool))
    if len(fallback_indices):
        result["fallback_point"] = point_metrics(
            [stamps[i] for i in fallback_indices],
            np.asarray(actual)[fallback_indices],
            {k: np.asarray(v)[fallback_indices] for k, v in predictions.items()},
            plan["scoring"],
        )
    if save:
        content = json_bytes(result)
        publish_bytes(plan_path.parent / "scores" / f"{digest(content)}.json", content)
    return result
