"""Prepare a research candidate or produce an explicitly labelled historical replay."""

import argparse
import json
import pickle
import shutil
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import lightgbm as lgb
from catboost import CatBoostError

from dk2_forecast.delivery_day import COPENHAGEN, forecast_issue_utc
from dk2_forecast.experiment import json_bytes
from dk2_forecast.explanatory import snapshot_cutoff
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.issue import capture_inputs
from dk2_forecast.operations import exclusive_run
from dk2_forecast.provenance import save_code_snapshot
from dk2_forecast.research import ResearchData, fit_model, predict_model
from dk2_forecast.research_features import daily_price_features
from dk2_forecast.research_metrics import interval_metrics, point_metrics
from dk2_forecast.storage import connect, digest, publish_bytes, write_parquet


def current_code_hash():
    return digest(
        b"\0".join(
            path.read_bytes() for path in sorted(Path(__file__).parent.glob("*.py"))
        )
    )


def freeze_candidate(report_path, root, start):
    from dk2_forecast.evaluation import freeze

    return freeze(report_path, root, start)


def forecast(report_path, prices, day, root, *, replay, evaluation=None):
    from dk2_forecast.evaluation import load_evaluation, input_history

    plan_path = Path(evaluation) if evaluation else None
    plan = load_evaluation(plan_path) if plan_path else None
    code_snapshot = save_code_snapshot(root)
    report = json.loads(report_path.read_bytes())
    config = report["selection"]
    report_hash = digest(report_path.read_bytes())
    now = datetime.now(UTC)
    base = plan_path.parent if plan_path else root / "research_live" / report_hash
    existing = base / "prepared" / str(day) / "forecast.parquet"
    if not replay and existing.is_file() and existing.with_suffix(".json").is_file():
        saved = json.loads(existing.with_suffix(".json").read_bytes())
        if (
            saved["forecast_sha256"] != digest(existing.read_bytes())
            or saved["configuration_sha256"] != report_hash
            or saved["code_sha256"] != current_code_hash()
        ):
            raise ValueError("Existing preparation differs from this candidate")
        if plan_path:
            from dk2_forecast.evaluation import validate_publication

            validate_publication(plan_path, root, day, prepared=True)
        return existing
    training_history = None
    if replay:
        history = PriceHistory.load(prices)
        input_receipt = {
            "data_version": history.data_version,
            "prices_sha256": digest(prices.read_bytes()),
            "historical_availability_assumption": report["historical_assumption"],
        }
    else:
        plan_path = plan_path or root / "research_live" / report_hash / "freeze.json"
        plan = load_evaluation(plan_path)
        candidate = plan.get("candidate", plan)
        if (
            candidate.get("implementation_sha256", candidate.get("code_sha256"))
            != current_code_hash()
        ):
            raise ValueError("Research implementation differs from its frozen plan")
        start = date.fromisoformat(plan["start"])
        if not start <= day < date.fromisoformat(plan["end_exclusive"]):
            raise ValueError("Delivery day is outside the research evaluation")
        if day != now.astimezone(COPENHAGEN).date() + date.resolution:
            raise ValueError("Live preparation forecasts tomorrow's delivery day")
        if now > snapshot_cutoff(day):
            raise ValueError(
                "Prepare research inputs before the 09:50 Copenhagen snapshot cutoff"
            )
        prices, history, input_receipt = capture_inputs(root, day)
        if datetime.fromisoformat(input_receipt["observed_at_utc"]) > snapshot_cutoff(
            day
        ):
            raise ValueError("Research snapshot completed after 09:50 Copenhagen")
        origin = day - timedelta(days=(day - start).days % config["retraining_days"])
        lock = plan_path.parent / "training" / f"{origin}.json"
        if day == origin:
            publish_bytes(lock, json_bytes(input_receipt))
        if not lock.is_file():
            raise ValueError(
                f"Missing block-origin receipt {origin}; create a new future evaluation of this candidate or wait for the next origin. Never backdate training."
            )
        training_receipt = json.loads(lock.read_bytes())
        if datetime.fromisoformat(
            training_receipt["observed_at_utc"]
        ) > snapshot_cutoff(origin):
            raise ValueError("Training snapshot was captured after its block cutoff")
        training_path = (
            root / "prices" / training_receipt["data_version"] / "prices.parquet"
        )
        if digest(training_path.read_bytes()) != training_receipt["prices_sha256"]:
            raise ValueError("Training snapshot hash mismatch")
        training_history = input_history(
            root, training_receipt, snapshot_cutoff(origin)
        )
    if replay:
        origin = day - timedelta(
            days=(day - date.fromisoformat(report["period"]["end_exclusive"])).days
            % config["retraining_days"]
        )
    data = ResearchData(training_history or history, root)
    eligible = (data.days < origin) & np.isfinite(data.actual)
    if config["training_weeks"] is not None:
        eligible &= data.days >= origin - timedelta(weeks=config["training_weeks"])
    train_x = data.matrix[eligible]
    train_y = data.actual[eligible]
    quantile_origin = day - timedelta(
        days=(
            day
            - (
                date.fromisoformat(report["period"]["end_exclusive"])
                if replay
                else start
            )
        ).days
        % 28
    )
    quantile_data = data
    quantile_receipt = training_receipt if not replay else input_receipt
    if not replay and quantile_origin != origin:
        quantile_lock = plan_path.parent / "training" / f"{quantile_origin}.json"
        if not quantile_lock.is_file():
            raise ValueError(
                f"Missing quantile block-origin receipt {quantile_origin}; create a new future evaluation of the retained candidate or wait for a shared point/quantile origin. Never backdate training."
            )
        quantile_receipt = json.loads(quantile_lock.read_bytes())
        if datetime.fromisoformat(
            quantile_receipt["observed_at_utc"]
        ) > snapshot_cutoff(quantile_origin):
            raise ValueError(
                "Quantile training inputs were observed after their cutoff"
            )
        quantile_path = (
            root / "prices" / quantile_receipt["data_version"] / "prices.parquet"
        )
        if digest(quantile_path.read_bytes()) != quantile_receipt["prices_sha256"]:
            raise ValueError("Quantile training snapshot hash mismatch")
        quantile_data = ResearchData(PriceHistory.load(quantile_path), root)
    quantile_eligible = (quantile_data.days < quantile_origin) & np.isfinite(
        quantile_data.actual
    )
    if config["training_weeks"] is not None:
        quantile_eligible &= quantile_data.days >= quantile_origin - timedelta(
            weeks=config["training_weeks"]
        )
    stamps, matrix, _, previous = daily_price_features(history, day)
    baseline = matrix[:, 0]
    if not np.isfinite(baseline).all():
        raise ValueError(
            "No complete previous-day benchmark, including permitted weekly fallback"
        )
    if config["correction"]:
        valid = np.isfinite(train_x[:, 0])
        train_y = train_y[valid] - train_x[valid, 0]
        train_x = train_x[valid]
    runtime = {}
    fallback = False
    error = None
    try:
        model, runtime["fit_seconds"], runtime["model_cache_hit"] = cached_model(
            base if not replay else root / "replay_model_cache",
            origin,
            config["family"],
            train_x,
            train_y,
            config["parameters"],
            {
                "code": current_code_hash(),
                "data": (training_history or history).data_version,
                "config": config,
            },
        )
        predictions = predict_model(model, matrix)
        if config["correction"]:
            predictions += baseline
        if not np.isfinite(predictions).all():
            raise ValueError("Model returned non-finite predictions")
        model_bytes = pickle.dumps(model)
        model_hash = digest(model_bytes)
        publish_bytes(root / "models" / f"research-{model_hash}.pkl", model_bytes)
    except (
        ValueError,
        RuntimeError,
        lgb.basic.LightGBMError,
        CatBoostError,
    ) as failure:
        predictions = baseline.copy()
        fallback = True
        error = str(failure)
        model_hash = "previous-day-fallback"
    quantiles = []
    quantile_fallback = False
    quantile_versions = {}
    quantile_error = None
    try:
        for alpha in (0.1, 0.5, 0.9):
            qmodel, _, _ = cached_model(
                base if not replay else root / "replay_model_cache",
                quantile_origin,
                "lightgbm",
                quantile_data.matrix[quantile_eligible],
                quantile_data.actual[quantile_eligible],
                report["fits"]["lightgbm_tuned"]["parameters"]
                | {"objective": "quantile", "alpha": alpha},
                {
                    "code": current_code_hash(),
                    "data": quantile_data.history.data_version,
                    "alpha": alpha,
                },
            )
            qbytes = qmodel.model_to_string().encode()
            quantile_versions[str(alpha)] = digest(qbytes)
            publish_bytes(root / "models" / f"quantile-{digest(qbytes)}.txt", qbytes)
            qprediction = predict_model(qmodel, matrix)
            if not np.isfinite(qprediction).all():
                raise ValueError("Quantile model returned non-finite values")
            quantiles.append(qprediction)
    except (ValueError, RuntimeError, lgb.basic.LightGBMError) as failure:
        quantile_fallback = True
        quantile_error = str(failure)
        quantile_versions = {}
        residuals = (
            quantile_data.actual[quantile_eligible]
            - quantile_data.matrix[quantile_eligible, 0]
        )
        residuals = residuals[np.isfinite(residuals)]
        if not len(residuals):
            raise ValueError(
                "No prior residuals available for quantile fallback"
            ) from failure
        quantiles = [
            baseline + np.quantile(residuals, alpha) for alpha in (0.1, 0.5, 0.9)
        ]
    raw_quantiles = np.column_stack(quantiles)
    quantile_crossings = int((np.diff(raw_quantiles, axis=1) < 0).any(axis=1).sum())
    quantiles = np.sort(raw_quantiles, axis=1)
    calibration = report.get("calibration", [])
    calibration_margin = (
        calibration[-1]["adjustment"] if calibration and not quantile_fallback else 0.0
    )
    quantiles[:, 0] -= calibration_margin
    quantiles[:, 2] += calibration_margin
    created = datetime.now(UTC)
    if not replay and created > forecast_issue_utc(day):
        raise ValueError("Forecast creation completed after scheduled issue")
    folder = root / "replays" / report_hash if replay else base
    if replay:
        folder = folder / created.strftime("%Y%m%dT%H%M%S%fZ")
    else:
        folder = folder / "prepared"
    folder = folder / str(day)
    rows = [
        {
            "price_area": "DK2",
            "delivery_date": str(day),
            "delivery_start_utc": stamp.isoformat(),
            "issue_time_utc": forecast_issue_utc(day).isoformat(),
            "model_version": model_hash,
            "data_version": history.data_version,
            "previous_day": float(baseline[i]),
            "weekly": float(matrix[i, 1]),
            "prediction": float(predictions[i]),
            "q10": float(quantiles[i, 0]),
            "q50": float(quantiles[i, 1]),
            "q90": float(quantiles[i, 2]),
            "raw_q10": float(raw_quantiles[i, 0]),
            "raw_q50": float(raw_quantiles[i, 1]),
            "raw_q90": float(raw_quantiles[i, 2]),
        }
        for i, stamp in enumerate(stamps)
    ]
    schema = [
        {
            "price_area": "VARCHAR",
            "delivery_date": "DATE",
            "delivery_start_utc": "TIMESTAMPTZ",
            "issue_time_utc": "TIMESTAMPTZ",
            "model_version": "VARCHAR",
            "data_version": "VARCHAR",
            **{
                name: "DOUBLE"
                for name in (
                    "previous_day",
                    "weekly",
                    "prediction",
                    "q10",
                    "q50",
                    "q90",
                    "raw_q10",
                    "raw_q50",
                    "raw_q90",
                )
            },
        }
    ]
    final_folder = folder
    with (
        exclusive_run(final_folder.parent / f"{day}.lock"),
        atomic_folder(final_folder, ".preparation-") as folder,
    ):
        path = folder / "forecast.parquet"
        with connect() as db:
            db.execute(
                "CREATE TABLE forecasts AS SELECT unnest(from_json_strict(?, ?), recursive := true)",
                [json.dumps(rows), json.dumps(schema)],
            )
            write_parquet(
                db, "SELECT * FROM forecasts ORDER BY delivery_start_utc", path
            )
        receipt = {
            "status": "replay" if replay else "prepared",
            "replay": replay,
            "created_at_utc": created,
            "scheduled_issue_utc": forecast_issue_utc(day),
            "data_cutoff_utc": snapshot_cutoff(day),
            "input": input_receipt,
            "configuration_sha256": report_hash,
            "model_version": model_hash,
            "training_origin": origin,
            **(
                {
                    "evaluation_id": plan["evaluation_id"],
                    "candidate_id": plan["candidate_id"],
                    "training_receipt_sha256": digest(json_bytes(training_receipt)),
                    "quantile_training_receipt_sha256": digest(
                        json_bytes(quantile_receipt)
                    ),
                }
                if plan and plan.get("schema_version") == 2
                else {}
            ),
            "git_revision": report["git_revision"],
            "code_sha256": current_code_hash(),
            "code_snapshot_sha256": code_snapshot,
            "configuration": config,
            "units": "EUR/MWh",
            "source_resolution": "PT15M",
            "forecast_sha256": digest(path.read_bytes()),
            "fallback": fallback,
            "fallback_reason": error,
            "quantile_fallback": quantile_fallback,
            "quantile_fallback_reason": quantile_error,
            "quantile_models": quantile_versions,
            "raw_quantile_crossings": quantile_crossings,
            "quantile_training_origin": quantile_origin,
            "calibration_margin_eur_per_mwh": calibration_margin,
            "calibration_source": calibration[-1] if calibration else None,
            "stale_input": False,
            "validation": {"unique_intervals": len(rows), "complete": True},
            "previous_day_weekly_fallbacks": sum(
                lag.weekly_fallback for lag in previous
            ),
            "runtime": runtime,
        }
        publish_bytes(path.with_suffix(".json"), json_bytes(receipt))

    return final_folder / "forecast.parquet"


@contextmanager
def atomic_folder(target, prefix):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix=prefix, dir=target.parent))
    try:
        yield temporary
        if target.exists():
            raise ValueError(
                "Incomplete or conflicting artifact exists; inspect and quarantine before retry"
            )
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def cached_model(base, origin, family, x, y, params, identity):
    key = digest(
        json_bytes(
            {
                "origin": origin,
                "family": family,
                "parameters": params,
                "identity": identity,
            }
        )
    )
    path = base / "model_cache" / key / "model.pkl"
    if path.is_file():
        meta = json.loads(path.with_suffix(".json").read_bytes())
        raw = path.read_bytes()
        if (
            digest(raw) != meta["sha256"]
            or meta["origin"] != str(origin)
            or meta["identity"] != identity
            or meta["cache_key"] != key
        ):
            raise ValueError("Cached model hash/origin/configuration mismatch")
        return pickle.loads(raw), 0.0, True
    model, seconds = fit_model(family, x, y, params)
    raw = pickle.dumps(model)
    # Only local generated model files are loaded. Receipt acts as cache commit marker.
    with atomic_folder(path.parent, ".model-") as temporary:
        (temporary / "model.pkl").write_bytes(raw)
        (temporary / "model.json").write_bytes(
            json_bytes(
                {
                    "sha256": digest(raw),
                    "origin": origin,
                    "identity": identity,
                    "cache_key": key,
                }
            )
        )
    return model, seconds, False


def publish_evaluation(root, plan_path, day):
    from dk2_forecast.evaluation import load_evaluation, validate_publication

    plan = load_evaluation(plan_path)
    if plan["candidate"]["implementation_sha256"] != current_code_hash():
        raise ValueError("Use this evaluation's retained launch.py execution bundle")
    target = plan_path.parent / "forecasts" / str(day) / "forecast.parquet"
    if target.parent.exists():
        validate_publication(plan_path, root, day)
        return target
    validate_publication(plan_path, root, day, prepared=True)
    now = datetime.now(UTC)
    issue = forecast_issue_utc(day)
    if not issue <= now < issue + timedelta(minutes=1):
        raise ValueError(
            "Publication must start during the scheduled 10:00 Copenhagen minute"
        )
    stage = plan_path.parent / "prepared" / str(day) / "forecast.parquet"
    receipt = json.loads(stage.with_suffix(".json").read_bytes())
    with exclusive_run(plan_path.parent / "locks" / "publication.lock"):
        target.parent.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(mkdtemp(prefix=".publication-", dir=target.parent.parent))
        try:
            (temporary / "forecast.parquet").write_bytes(stage.read_bytes())
            receipt |= {
                "status": "issued",
                "published_at_utc": datetime.now(UTC).isoformat(),
            }
            (temporary / "forecast.json").write_bytes(json_bytes(receipt))
            if datetime.now(UTC) >= issue + timedelta(minutes=1):
                raise ValueError("Publication exceeded issue window")
            temporary.rename(target.parent)
        finally:
            if temporary.exists():
                for file in temporary.iterdir():
                    file.unlink()
                temporary.rmdir()
    return target


def publish(root, report_path, day, *, evaluation=None):
    if evaluation:
        return publish_evaluation(root, Path(evaluation), day)
    report_hash = digest(report_path.read_bytes())
    base = root / "research_live" / report_hash
    path = base / "prepared" / str(day) / "forecast.parquet"
    target = base / "forecasts" / str(day) / "forecast.parquet"
    if target.is_file() and target.with_suffix(".json").is_file():
        receipt = json.loads(target.with_suffix(".json").read_bytes())
        if digest(target.read_bytes()) != receipt["forecast_sha256"]:
            raise ValueError("Published forecast hash mismatch")
        return target
    receipt = json.loads(path.with_suffix(".json").read_bytes())
    plan = json.loads((base / "freeze.json").read_bytes())
    if (
        plan["code_sha256"] != current_code_hash()
        or receipt["code_sha256"] != plan["code_sha256"]
    ):
        raise ValueError(
            "Research publication code differs from the frozen preparation"
        )
    now = datetime.now(UTC)
    issue = forecast_issue_utc(day)
    if not issue <= now < issue + timedelta(minutes=1):
        raise ValueError(
            "Publication must start during the scheduled 10:00 Copenhagen minute"
        )
    if receipt["replay"] or datetime.fromisoformat(receipt["created_at_utc"]) > issue:
        raise ValueError("Prepared artifact is ineligible for live publication")
    if datetime.fromisoformat(receipt["input"]["observed_at_utc"]) > snapshot_cutoff(
        day
    ):
        raise ValueError("Research inputs were verified after the data cutoff")
    if digest(path.read_bytes()) != receipt["forecast_sha256"]:
        raise ValueError("Prepared forecast hash mismatch")
    with exclusive_run(base / "locks" / f"{day}.lock"):
        target.parent.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(mkdtemp(prefix=".publication-", dir=target.parent.parent))
        try:
            (temporary / "forecast.parquet").write_bytes(path.read_bytes())
            receipt |= {
                "status": "issued",
                "published_at_utc": datetime.now(UTC).isoformat(),
            }
            (temporary / "forecast.json").write_bytes(json_bytes(receipt))
            if datetime.now(UTC) >= issue + timedelta(minutes=1):
                raise ValueError("Publication preparation exceeded the issue minute")
            temporary.rename(target.parent)
        finally:
            if temporary.exists():
                (temporary / "forecast.parquet").unlink(missing_ok=True)
                (temporary / "forecast.json").unlink(missing_ok=True)
                temporary.rmdir()
    return target


def score_forecast(report_path, forecast_path, actuals):
    report = json.loads(report_path.read_bytes())
    receipt = json.loads(forecast_path.with_suffix(".json").read_bytes())
    if receipt["configuration_sha256"] != digest(report_path.read_bytes()) or receipt[
        "forecast_sha256"
    ] != digest(forecast_path.read_bytes()):
        raise ValueError("Forecast configuration or artifact hash mismatch")
    if not receipt["replay"]:
        issue = datetime.fromisoformat(receipt["scheduled_issue_utc"])
        published = datetime.fromisoformat(receipt["published_at_utc"])
        observed = datetime.fromisoformat(receipt["input"]["observed_at_utc"])
        cutoff = datetime.fromisoformat(receipt["data_cutoff_utc"])
        if not observed <= cutoff <= issue <= published < issue + timedelta(minutes=1):
            raise ValueError(
                "Research issuance does not satisfy snapshot and publication timing"
            )
    history = PriceHistory.load(actuals)
    with connect() as db:
        query = db.execute(
            "SELECT * FROM read_parquet(?) ORDER BY delivery_start_utc",
            [str(forecast_path)],
        )
        names = [column[0] for column in query.description]
        rows = [dict(zip(names, row, strict=True)) for row in query.fetchall()]
    from dk2_forecast.delivery_day import delivery_intervals_utc

    day = rows[0]["delivery_date"]
    stamps = [row["delivery_start_utc"] for row in rows]
    if stamps != delivery_intervals_utc(day) or any(
        row["price_area"] != "DK2" for row in rows
    ):
        raise ValueError("Forecast has missing, duplicate or wrong-area delivery keys")
    actual = [history.prices.get(stamp, np.nan) for stamp in stamps]
    if not np.isfinite(actual).all():
        raise ValueError("Actual prices are incomplete for this delivery day")
    predictions = {
        name: [row[name] for row in rows]
        for name in ("previous_day", "weekly", "prediction")
    }
    quantiles = [[row[name] for name in ("q10", "q50", "q90")] for row in rows]
    content = json_bytes(
        {
            "status": "replay_diagnostic"
            if receipt["replay"]
            else "issued_research_evaluation",
            "created_at_utc": datetime.now(UTC),
            "delivery_date": day,
            "forecast_sha256": receipt["forecast_sha256"],
            "actuals_data_version": history.data_version,
            "point": point_metrics(stamps, actual, predictions, report["scoring"]),
            "intervals": interval_metrics(stamps, actual, quantiles),
            "fallback_intervals": len(rows) if receipt["fallback"] else 0,
            "quantile_fallback_intervals": len(rows)
            if receipt["quantile_fallback"]
            else 0,
        }
    )
    path = forecast_path.parent / "scores" / f"{digest(content)}.json"
    publish_bytes(path, content)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare, publish or replay a separately versioned research model."
    )
    parser.add_argument(
        "action", choices=("freeze", "prepare", "publish", "replay", "score")
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("delivery_date", type=date.fromisoformat)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--forecast", type=Path, help="Stored forecast to score")
    parser.add_argument(
        "--evaluation",
        type=Path,
        help="Explicit research evaluation freeze.json; operational runs should use its retained launch.py",
    )
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze_candidate(args.report, args.root, args.delivery_date)
    elif args.action == "score":
        if args.prices is None or args.forecast is None:
            parser.error("--prices and --forecast are required for scoring")
        result = score_forecast(args.report, args.forecast, args.prices)
    elif args.action == "publish":
        result = publish(
            args.root, args.report, args.delivery_date, evaluation=args.evaluation
        )
    else:
        if args.action == "replay" and args.prices is None:
            parser.error("--prices is required for a replay")
        result = forecast(
            args.report,
            args.prices,
            args.delivery_date,
            args.root,
            replay=args.action == "replay",
            evaluation=args.evaluation,
        )
    print(result)


if __name__ == "__main__":
    main()
