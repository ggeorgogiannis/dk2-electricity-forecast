r"""Replay daily DK2 forecasts with a retained candidate and no parameter tuning.

Run from the checkout with .venv\python.exe -I scripts\replay_year.py.
Generated models, daily receipts and checkpoints stay under ignored data.
"""

import argparse
from bisect import bisect_left
import csv
from datetime import UTC, date, datetime, timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import sys
from time import perf_counter
from zipfile import ZipFile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=str) + "\n"
    ).encode()


def training_origin(day, anchor, frequency):
    if frequency < 1:
        raise ValueError("Refit frequency must be positive")
    return day - timedelta(days=(day - anchor).days % frequency)


def training_indices(days, origin, weeks=None):
    lower = origin - timedelta(weeks=weeks) if weeks is not None else date.min
    return [i for i, day in enumerate(days) if lower <= day < origin]


def validate_rows(rows, expected):
    if [row["delivery_start_utc"] for row in rows] != [t.isoformat() for t in expected]:
        raise ValueError("Missing, duplicate or incorrect UTC delivery keys")
    if any(row["price_area"] != "DK2" for row in rows):
        raise ValueError("Wrong price area")


def validate_model_cache(metadata, raw, identity, origin, training_hash):
    if (
        metadata["sha256"] != sha(raw)
        or metadata["run_id"] != identity
        or metadata["origin"] != str(origin)
        or metadata["training_sha256"] != training_hash
    ):
        raise ValueError("Model cache identity, origin, training data or hash mismatch")


def load_frozen(project, plan):
    if "dk2_forecast" in sys.modules:
        raise RuntimeError("Start the replay in a fresh isolated Python process (-I)")
    candidate = plan["candidate"]
    for package, expected in candidate["versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Frozen dependency mismatch: {package}")
    archive = (
        project / "data" / "code" / candidate["code_snapshot_sha256"] / "source.zip"
    )
    if sha(archive.read_bytes()) != candidate["archive_sha256"]:
        raise ValueError("Source archive hash mismatch")
    with ZipFile(archive) as zipped:
        entries = [(name, zipped.read(name)) for name in zipped.namelist()]
    if (
        sha(b"\0".join(name.encode() + b"\0" + raw for name, raw in entries))
        != candidate["code_snapshot_sha256"]
    ):
        raise ValueError("Source content identity mismatch")
    source = (
        project / "data" / "year_replays" / "source" / candidate["code_snapshot_sha256"]
    )
    for name, raw in entries:
        target = source / name
        if not target.resolve().is_relative_to(source.resolve()):
            raise ValueError("Unsafe archive path")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != raw:
            raise ValueError("Retained source extraction was modified")
        if not target.exists():
            target.write_bytes(raw)
    sys.path.insert(0, str(source))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(project, prices, plan_path, start, end=None):
    project, prices, plan_path = (
        project.resolve(),
        prices.resolve(),
        plan_path.resolve(),
    )
    plan = json.loads(plan_path.read_bytes())
    load_frozen(project, plan)
    import numpy as np
    from dk2_forecast.audit_snapshot import verify_snapshot
    from dk2_forecast.delivery_day import (
        COPENHAGEN,
        delivery_intervals_utc,
        forecast_issue_utc,
    )
    from dk2_forecast.evaluation import load_evaluation
    from dk2_forecast.explanatory import snapshot_cutoff
    from dk2_forecast.forecast import PriceHistory
    from dk2_forecast.operations import exclusive_run
    from dk2_forecast.price_lags import local_day
    from dk2_forecast.research import ResearchData, fit_model, predict_model
    from dk2_forecast.research_features import daily_price_features
    from dk2_forecast.research_forecast import atomic_folder, current_code_hash
    from dk2_forecast.research_metrics import point_metrics
    from dk2_forecast.storage import connect, publish_bytes, write_parquet

    load_evaluation(plan_path)
    candidate = plan["candidate"]
    if current_code_hash() != candidate["implementation_sha256"]:
        raise ValueError("Loaded implementation differs from retained candidate")
    config = candidate["selection"]
    if config["family"] != "catboost" or config["correction"]:
        raise ValueError(
            "This replay supports the selected direct-price CatBoost candidate"
        )
    audit = verify_snapshot(prices, project / "data")
    history = PriceHistory.load(prices)
    latest = max(local_day(t) for t in history.prices)
    end = end or min(
        latest + timedelta(days=1), datetime.now(COPENHAGEN).date() + timedelta(days=1)
    )
    if not start < end <= latest + timedelta(days=1):
        raise ValueError("Choose a nonempty interval with complete actual-price data")
    anchor = date.fromisoformat(plan["start"])
    frequency = config["retraining_days"]
    contract = {
        "schema_version": 1,
        "status": "historical_replay_not_prospective",
        "candidate_id": plan["candidate_id"],
        "candidate": candidate,
        "evaluation_reference": plan["evaluation_id"],
        "start": str(start),
        "end_exclusive": str(end),
        "refit_anchor": str(anchor),
        "frequency_days": frequency,
        "scoring": plan["scoring"],
        "prices_sha256": audit["parquet_sha256"],
        "data_version": history.data_version,
        "driver_sha256": sha(Path(__file__).read_bytes()),
        "input_policy": "For each target D, use actual prices on delivery dates through D-1 only. Fit labels strictly before the weekly origin. Never feed predictions back as prices.",
        "historical_availability_assumption": "Archived issue-day delivery prices assumed available by 09:50 Copenhagen; historical publication/revision vintages are unavailable. Candidate parameters were selected after many evaluated dates; entire run is retrospective, not an untouched test.",
    }
    identity = sha(canonical(contract))
    folder = project / "data" / "year_replays" / identity
    folder.mkdir(parents=True, exist_ok=True)
    if publish_bytes(folder / "contract.json", canonical(contract)) != canonical(
        contract
    ):
        raise ValueError("Replay identity conflict")
    publish_bytes(folder / "runner.py", Path(__file__).read_bytes())
    publish_bytes(folder / "input-audit.json", canonical(audit))
    expected_days = [start + timedelta(days=i) for i in range((end - start).days)]
    expected = [t for day in expected_days for t in delivery_intervals_utc(day)]
    if not set(expected).issubset(history.prices):
        raise ValueError(
            "Actual prices have interior gaps; no delivery days may be silently omitted"
        )
    first_origin = training_origin(start, anchor, frequency)
    if min(local_day(t) for t in history.prices) > first_origin - timedelta(days=28):
        raise ValueError("Insufficient warm-up history for the first origin")
    print(
        f"Replay {start} through {end - timedelta(days=1)}: {len(expected_days)} days, {len(expected)} intervals",
        flush=True,
    )
    print(f"Output: {folder}", flush=True)
    rows, fits, receipt_hashes = [], [], []
    started = datetime.now(UTC)
    with exclusive_run(folder / "run.lock"):
        data = ResearchData(history, project / "data")
        all_stamps = list(history.prices)
        all_values = list(history.prices.values())
        model, model_origin, model_hash = None, None, None
        for day_number, day in enumerate(expected_days, 1):
            origin = training_origin(day, anchor, frequency)
            if origin != model_origin:
                indices = np.asarray(
                    training_indices(data.days, origin, config["training_weeks"]),
                    dtype=int,
                )
                indices = indices[np.isfinite(data.actual[indices])]
                assert all(data.days[i] < origin for i in indices)
                last_label = max(data.days[indices])
                assert last_label == origin - timedelta(days=1)
                train_x, train_y = data.matrix[indices], data.actual[indices]
                training_hash = sha(train_x.tobytes() + train_y.tobytes())
                cache = folder / "models" / str(origin)
                if cache.exists():
                    meta = json.loads((cache / "model.json").read_bytes())
                    raw = (cache / "model.pkl").read_bytes()
                    validate_model_cache(meta, raw, identity, origin, training_hash)
                    # Only verified, locally generated model caches are loaded.
                    model = pickle.loads(raw)
                    resumed = True
                else:
                    model, seconds = fit_model(
                        config["family"], train_x, train_y, config["parameters"]
                    )
                    raw = pickle.dumps(model)
                    meta = {
                        "run_id": identity,
                        "origin": str(origin),
                        "training_last_delivery": str(last_label),
                        "training_rows": len(indices),
                        "training_sha256": training_hash,
                        "sha256": sha(raw),
                        "fit_seconds": seconds,
                        "created_at_utc": datetime.now(UTC).isoformat(),
                    }
                    with atomic_folder(cache, ".model-") as temporary:
                        (temporary / "model.pkl").write_bytes(raw)
                        (temporary / "model.json").write_bytes(canonical(meta))
                    resumed = False
                model_origin, model_hash = origin, sha(raw)
                fits.append(meta)
                print(
                    f"Origin {origin}; labels through {last_label}; {len(indices)} rows; fit {meta['fit_seconds']:.2f}s; cache={resumed}",
                    flush=True,
                )
            stamps = delivery_intervals_utc(day)
            count = bisect_left(all_stamps, stamps[0])
            known = PriceHistory(
                dict(zip(all_stamps[:count], all_values[:count], strict=True)),
                history.data_version,
            )
            assert max(local_day(t) for t in known.prices) == day - timedelta(days=1)
            stamps_check, matrix, _, previous = daily_price_features(known, day)
            assert stamps_check == stamps
            # Test the information boundary every day, including both DST days.
            contaminated = PriceHistory(
                known.prices | {t: 99999.0 for t in stamps}, history.data_version
            )
            _, changed, _, _ = daily_price_features(contaminated, day)
            np.testing.assert_allclose(matrix, changed, equal_nan=True)
            assert all(
                lag.source_utc is not None and local_day(lag.source_utc) < day
                for lag in previous
            )
            prediction_started = perf_counter()
            predicted = np.asarray(predict_model(model, matrix))
            prediction_seconds = perf_counter() - prediction_started
            if not np.isfinite(predicted).all() or not np.isfinite(matrix[:, :2]).all():
                raise ValueError(f"Incomplete prediction or baseline for {day}")
            daily = [
                {
                    "price_area": "DK2",
                    "delivery_date": str(day),
                    "delivery_start_utc": t.isoformat(),
                    "issue_time_utc": forecast_issue_utc(day).isoformat(),
                    "model_version": model_hash,
                    "data_version": history.data_version,
                    "training_origin": str(origin),
                    "previous_source_utc": previous[i].source_utc.isoformat(),
                    "weekly_fallback": previous[i].weekly_fallback,
                    "previous_day": float(matrix[i, 0]),
                    "weekly_utc": float(matrix[i, 1]),
                    "prediction": float(predicted[i]),
                    "actual": float(history.prices[t]),
                }
                for i, t in enumerate(stamps)
            ]
            validate_rows(daily, stamps)
            path = folder / "forecasts" / str(day) / "forecast.parquet"
            metadata = {
                "status": "replay",
                "replay": True,
                "eligible_for_prospective_scoring": False,
                "run_id": identity,
                "candidate_id": plan["candidate_id"],
                "created_at_utc": datetime.now(UTC).isoformat(),
                "scheduled_issue_utc": forecast_issue_utc(day).isoformat(),
                "input_cutoff_utc": snapshot_cutoff(day).isoformat(),
                "input_last_delivery_date": str(day - timedelta(days=1)),
                "training_origin": str(origin),
                "training_last_delivery_date": str(last_label),
                "model_version": model_hash,
                "data_version": history.data_version,
                "prices_sha256": audit["parquet_sha256"],
                "code_snapshot_sha256": candidate["code_snapshot_sha256"],
                "driver_sha256": contract["driver_sha256"],
                "intervals": len(stamps),
                "previous_day_weekly_fallbacks": sum(
                    lag.weekly_fallback for lag in previous
                ),
                "prediction_seconds": prediction_seconds,
                "validation": {
                    "complete": True,
                    "target_contamination_test_passed": True,
                },
            }
            if path.parent.exists():
                saved = json.loads(path.with_suffix(".json").read_bytes())
                if (
                    saved["run_id"] != identity
                    or saved["model_version"] != model_hash
                    or saved["forecast_sha256"] != sha(path.read_bytes())
                ):
                    raise ValueError(
                        "Existing daily checkpoint differs from this replay"
                    )
            else:
                schema = [
                    {
                        "price_area": "VARCHAR",
                        "delivery_date": "DATE",
                        "delivery_start_utc": "TIMESTAMPTZ",
                        "issue_time_utc": "TIMESTAMPTZ",
                        "model_version": "VARCHAR",
                        "data_version": "VARCHAR",
                        "training_origin": "DATE",
                        "previous_source_utc": "TIMESTAMPTZ",
                        "weekly_fallback": "BOOLEAN",
                        **{
                            name: "DOUBLE"
                            for name in (
                                "previous_day",
                                "weekly_utc",
                                "prediction",
                                "actual",
                            )
                        },
                    }
                ]
                with atomic_folder(path.parent, ".day-") as temporary:
                    with connect() as db:
                        db.execute(
                            "CREATE TABLE forecasts AS SELECT unnest(from_json_strict(?, ?), recursive := true)",
                            [json.dumps(daily), json.dumps(schema)],
                        )
                        write_parquet(
                            db,
                            "SELECT * FROM forecasts ORDER BY delivery_start_utc",
                            temporary / "forecast.parquet",
                        )
                    metadata["forecast_sha256"] = sha(
                        (temporary / "forecast.parquet").read_bytes()
                    )
                    (temporary / "forecast.json").write_bytes(canonical(metadata))
                saved = metadata
            rows.extend(daily)
            receipt_hashes.append(
                {
                    "day": str(day),
                    "forecast_sha256": saved["forecast_sha256"],
                    "receipt_sha256": sha(path.with_suffix(".json").read_bytes()),
                }
            )
            progress = {
                "run_id": identity,
                "completed_days": day_number,
                "total_days": len(expected_days),
                "last_delivery": str(day),
                "updated_at_utc": datetime.now(UTC).isoformat(),
            }
            temporary_progress = folder / ".progress.tmp"
            temporary_progress.write_bytes(canonical(progress))
            temporary_progress.replace(folder / "progress.json")
        validate_rows(rows, expected)
        stamps = [datetime.fromisoformat(row["delivery_start_utc"]) for row in rows]
        actual = np.asarray([row["actual"] for row in rows])
        predictions = {
            "previous_day": np.asarray([row["previous_day"] for row in rows]),
            "weekly_utc": np.asarray([row["weekly_utc"] for row in rows]),
            "catboost": np.asarray([row["prediction"] for row in rows]),
        }
        print(
            "All daily forecasts complete; computing paired metrics and report",
            flush=True,
        )
        metrics = point_metrics(stamps, actual, predictions, plan["scoring"])
        result = {
            "run_id": identity,
            "status": "historical_year_replay",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "started_at_utc": started.isoformat(),
            "contract": contract,
            "input_audit": audit,
            "delivery_days": len(expected_days),
            "intervals": len(rows),
            "point": metrics,
            "fits": fits,
            "forecast_receipts": receipt_hashes,
            "fallback_intervals": sum(row["weekly_fallback"] for row in rows),
            "model_fallback_intervals": 0,
        }
        result.update(build_report(folder, rows, result, np))
        raw_report = canonical(result)
        report_path = folder / "reports" / f"{sha(raw_report)}.json"
        publish_bytes(report_path, raw_report)
        (folder / "latest-report.txt").write_text(str(report_path), encoding="utf-8")
    print(f"REPORT {report_path}", flush=True)
    return report_path


def build_report(folder, rows, result, np):
    """Human and machine summaries; no result-dependent candidate changes."""
    from dk2_forecast.delivery_day import COPENHAGEN
    from dk2_forecast.storage import connect, write_parquet

    days = np.asarray([row["delivery_date"] for row in rows])
    actual = np.asarray([row["actual"] for row in rows])
    values = {
        "previous_day": np.asarray([row["previous_day"] for row in rows]),
        "weekly_utc": np.asarray([row["weekly_utc"] for row in rows]),
        "catboost": np.asarray([row["prediction"] for row in rows]),
    }

    def stats(mask):
        n = int(mask.sum())
        return {
            "intervals": n,
            "methods": {
                name: {
                    "mae": float(np.abs(series[mask] - actual[mask]).mean()),
                    "rmse": float(
                        np.sqrt(np.square(series[mask] - actual[mask]).mean())
                    ),
                    "bias": float((series[mask] - actual[mask]).mean()),
                    "within_25": float((abs(series[mask] - actual[mask]) <= 25).mean()),
                    "within_50": float((abs(series[mask] - actual[mask]) <= 50).mean()),
                }
                for name, series in values.items()
            }
            if n
            else {},
        }

    daily = []
    for day in sorted(set(days)):
        mask = days == day
        score = stats(mask)
        daily.append(
            {
                "delivery_date": day,
                "intervals": score["intervals"],
                "training_origin": next(
                    row["training_origin"]
                    for row in rows
                    if row["delivery_date"] == day
                ),
                **{
                    f"{name}_mae": metrics["mae"]
                    for name, metrics in score["methods"].items()
                },
                "catboost_bias": score["methods"]["catboost"]["bias"],
                "catboost_within_25": score["methods"]["catboost"]["within_25"],
                "catboost_within_50": score["methods"]["catboost"]["within_50"],
                "actual_mean": float(actual[mask].mean()),
                "catboost_mean": float(values["catboost"][mask].mean()),
                "previous_day_mean": float(values["previous_day"][mask].mean()),
            }
        )
    monthly = []
    for month in sorted({day[:7] for day in days}):
        score = stats(np.asarray([day.startswith(month) for day in days]))
        month_days = [day for day in daily if day["delivery_date"].startswith(month)]
        model, baseline = score["methods"]["catboost"], score["methods"]["previous_day"]
        monthly.append(
            {
                "month": month,
                "days": len(month_days),
                "intervals": score["intervals"],
                "previous_day_mae": baseline["mae"],
                "weekly_utc_mae": score["methods"]["weekly_utc"]["mae"],
                "catboost_mae": model["mae"],
                "catboost_skill": 1 - model["mae"] / baseline["mae"]
                if baseline["mae"]
                else None,
                "catboost_day_wins": sum(
                    day["catboost_mae"] < day["previous_day_mae"] - 1e-9
                    for day in month_days
                ),
                "catboost_bias": model["bias"],
                "catboost_within_25": model["within_25"],
                "catboost_within_50": model["within_50"],
            }
        )
    periods = [
        ("Earlier development history / retrospective", "2026-01-01", "2026-04-02"),
        ("Current candidate selection data", "2026-04-02", "2026-09-21"),
        ("Previously observed diagnostic", "2026-09-21", "2026-11-16"),
        ("Transition before live window", "2026-11-16", "2026-11-20"),
        ("Post-freeze dates; already manually replayed", "2026-11-20", "2100-01-01"),
    ]
    segments = {
        label: {
            "start": start,
            "end_exclusive": min(end, result["contract"]["end_exclusive"]),
            **stats((days >= start) & (days < end)),
        }
        for label, start, end in periods
    }
    threshold = result["contract"]["scoring"]["extreme_threshold_eur_per_mwh"]
    regimes = {
        name: stats(mask)
        for name, mask in {
            "negative": actual < 0,
            "high_200_or_more": actual >= 200,
            "extreme_absolute": abs(actual) >= threshold,
        }.items()
    }
    best = min(daily, key=lambda row: row["catboost_mae"] - row["previous_day_mae"])
    worst = max(daily, key=lambda row: row["catboost_mae"])
    biggest_loss = max(
        daily, key=lambda row: row["catboost_mae"] - row["previous_day_mae"]
    )
    verification = {
        "daily_target_contamination_checks": len(daily),
        "training_origin_cutoff_checks": len(result["fits"]),
        "complete_day_keys": len(daily),
        "live_eligibility": False,
    }
    previous_check = (
        folder.parents[1]
        / "manual_runs"
        / "20261125T090426Z"
        / "forecasts"
        / "2026-11-24"
        / "forecast.parquet"
    )
    if previous_check.is_file() and "2026-11-24" in days:
        with connect() as db:
            saved = db.execute(
                "SELECT prediction FROM read_parquet(?) ORDER BY delivery_start_utc",
                [str(previous_check)],
            ).fetchnumpy()["prediction"]
        recomputed = values["catboost"][days == "2026-11-24"]
        np.testing.assert_allclose(saved, recomputed, rtol=0, atol=1e-10)
        verification["november_24_max_difference_from_prior_replay"] = float(
            np.max(abs(saved - recomputed))
        )
    write_csv(folder / "daily_scores.csv", daily)
    write_csv(folder / "monthly_scores.csv", monthly)
    export_rows = [
        row
        | {
            "delivery_start_copenhagen": datetime.fromisoformat(
                row["delivery_start_utc"]
            )
            .astimezone(COPENHAGEN)
            .isoformat()
        }
        for row in rows
    ]
    write_csv(folder / "all_interval_comparisons.csv", export_rows)
    combined = folder / "all_interval_comparisons.parquet"
    with connect() as db:
        paths = [
            str(folder / "forecasts" / day / "forecast.parquet")
            for day in sorted(set(days))
        ]
        db.execute(
            "CREATE TABLE all_forecasts AS SELECT * FROM read_parquet(?)", [paths]
        )
        write_parquet(
            db, "SELECT * FROM all_forecasts ORDER BY delivery_start_utc", combined
        )
    os.environ["MPLCONFIGDIR"] = str(folder / "plot-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    day_dates = [date.fromisoformat(row["delivery_date"]) for row in daily]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout="constrained")
    axes[0].plot(
        day_dates,
        [row["actual_mean"] for row in daily],
        label="Actual daily mean",
        color="black",
        lw=1,
    )
    axes[0].plot(
        day_dates,
        [row["catboost_mean"] for row in daily],
        label="CatBoost daily mean",
        lw=1,
        alpha=0.85,
    )
    axes[0].set_ylabel("EUR/MWh")
    axes[0].set_title("2026 daily forecasts: historical replay, not live performance")
    axes[0].legend(loc="upper left")
    for key, label in [
        ("previous_day_mae", "Previous Danish day"),
        ("catboost_mae", "CatBoost"),
    ]:
        losses = np.asarray([row[key] for row in daily])
        rolling = [losses[max(0, i - 6) : i + 1].mean() for i in range(len(losses))]
        axes[1].plot(day_dates, rolling, label=label)
    axes[1].set_ylabel("7-day mean daily MAE\n(EUR/MWh)")
    axes[1].legend(loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(folder / "year_overview.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    positions = np.arange(len(monthly))
    ax.bar(
        positions - 0.2,
        [row["previous_day_mae"] for row in monthly],
        width=0.4,
        label="Previous Danish day",
    )
    ax.bar(
        positions + 0.2,
        [row["catboost_mae"] for row in monthly],
        width=0.4,
        label="CatBoost",
    )
    ax.set_xticks(positions, [row["month"][5:] for row in monthly])
    ax.set_xlabel("2026 delivery month")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("All months on identical delivery intervals")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(folder / "monthly_mae.png", dpi=150)
    plt.close(fig)
    examples = list(
        dict.fromkeys(
            [
                worst["delivery_date"],
                best["delivery_date"],
                biggest_loss["delivery_date"],
            ]
        )
    )
    fig, axes = plt.subplots(
        len(examples),
        1,
        figsize=(12, 3.3 * len(examples)),
        layout="constrained",
        squeeze=False,
    )
    for ax, day in zip(axes.flat, examples, strict=True):
        selected = [row for row in rows if row["delivery_date"] == day]
        time = [datetime.fromisoformat(row["delivery_start_utc"]) for row in selected]
        ax.plot(
            time, [row["actual"] for row in selected], color="black", label="Actual"
        )
        ax.plot(time, [row["prediction"] for row in selected], label="CatBoost")
        ax.plot(
            time,
            [row["previous_day"] for row in selected],
            alpha=0.7,
            label="Previous day",
            ls="--",
        )
        ax.set_title(day)
        ax.set_ylabel("EUR/MWh")
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=COPENHAGEN))
        ax.legend(loc="upper left", ncols=3)
    fig.savefig(folder / "failure_and_success_examples.png", dpi=150)
    plt.close(fig)
    methods = result["point"]["extended"]["methods"]
    bands = result["point"]["slices"]["all"]["methods"]
    wins = result["point"]["delivery_day_results"]["methods"]["catboost"]
    uncertainty = result["point"]["paired_daily_uncertainty"]["catboost"]
    lines = [
        "# DK2 2026 daily forecast comparison",
        "",
        f"Period: {result['contract']['start']} through {day_dates[-1]}; {len(daily)} Danish delivery days and {len(rows):,} paired quarter-hours.",
        "",
        "## Question and method",
        "",
        "Can the fixed CatBoost candidate improve on simply copying the previous Danish delivery day's price curve? Every target day uses the preceding day's actual prices, eligible older history and calendar features. Predictions are refreshed daily; model fitting follows the unchanged seven-day Friday cycle anchored to 20 November. Training labels end before each origin. Prices remain in EUR/MWh and UTC identifies delivery intervals.",
        "",
        f"Warm-up data begin in October 2025; the first evaluated model origin is {result['fits'][0]['origin']}, with labels through {result['fits'][0]['training_last_delivery']}. Both clock-change days are included. Repeated local source slots use the earlier UTC occurrence; missing slots use the existing weekly fallback ({result['fallback_intervals']} intervals).",
        "",
        "This is a retrospective replay. Parameters were selected using some of these dates, and historical publication/revision vintages are unavailable. No forecast is represented as issued on time. No tuning or feature selection used the new annual results.",
        "",
        "## Overall result",
        "",
        "| Method | Coverage | MAE | RMSE | Bias | Within EUR25 | Within EUR50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("previous_day", "weekly_utc", "catboost"):
        row, band = methods[name], bands[name]
        lines.append(
            f"| {name} | 100% | {row['mae']:.2f} | {row['rmse']:.2f} | {row['mean_signed_error']:.2f} | {100 * band['within_25']:.1f}% | {100 * band['within_50']:.1f}% |"
        )
    skill = methods["catboost"]["skill_against_previous_day"]
    lines += [
        "",
        f"Relative MAE skill against persistence: {100 * skill:.1f}%. CatBoost day results: {wins['wins']} wins, {wins['losses']} losses, {wins['ties']} ties. Bias is prediction minus actual. Negative bias means underprediction. Main scores weight quarter-hours equally; daily wins weight delivery days equally, including 92/100-slot days.",
        "",
        f"Paired daily block-bootstrap result: {json.dumps(uncertainty)}. This uncertainty estimate is exploratory and unadjusted for parameter selection and multiple comparisons.",
        "",
        "## Monthly consistency",
        "",
        "| Month | Days | Intervals | Previous-day MAE | CatBoost MAE | Relative skill | Day wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in monthly:
        lines.append(
            f"| {row['month']} | {row['days']} | {row['intervals']:,} | {row['previous_day_mae']:.2f} | {row['catboost_mae']:.2f} | {100 * row['catboost_skill']:.1f}% | {row['catboost_day_wins']} |"
        )
    lines += [
        "",
        "## Price regimes",
        "",
        f"High-price threshold is at least EUR200/MWh. Extreme threshold is the unchanged development threshold |price| >= {threshold} EUR/MWh.",
        "",
        "| Slice | Intervals | Previous-day MAE | CatBoost MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, item in regimes.items():
        if item["intervals"]:
            lines.append(
                f"| {label} | {item['intervals']} | {item['methods']['previous_day']['mae']:.2f} | {item['methods']['catboost']['mae']:.2f} |"
            )
        else:
            lines.append(f"| {label} | 0 | unavailable | unavailable |")
    lines += [
        "",
        "## Interpretation of the dates",
        "",
        "| Evidence category | Intervals | Previous-day MAE | CatBoost MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, item in segments.items():
        if item["intervals"]:
            lines.append(
                f"| {label} | {item['intervals']} | {item['methods']['previous_day']['mae']:.2f} | {item['methods']['catboost']['mae']:.2f} |"
            )
    lines += [
        "",
        "Earlier dates were already part of earlier development, current candidate selection or inspected diagnostics. Post-freeze dates have also been manually inspected. There is no untouched annual test or live forecast performance in this report.",
        "",
        "## Failure and success cases",
        "",
        f"Highest CatBoost daily MAE: {worst['delivery_date']} ({worst['catboost_mae']:.2f}, baseline {worst['previous_day_mae']:.2f}). Largest relative daily improvement: {best['delivery_date']} ({best['catboost_mae']:.2f}, baseline {best['previous_day_mae']:.2f}). Largest daily loss to persistence: {biggest_loss['delivery_date']} ({biggest_loss['catboost_mae']:.2f}, baseline {biggest_loss['previous_day_mae']:.2f}). These examples are selected by fixed error criteria, not only favorable outcomes.",
        "",
        "## Limits and release assessment",
        "",
        "MAE improvement does not establish trading profitability. Transaction costs, positions, execution, imbalance and risk limits are not modelled. One calendar year does not establish multi-season robustness. Historical as-of assumptions and retrospective parameter selection limit claims. Report the negative/extreme slices and losing months alongside the average. GitHub publication requires user review; no repository, commit or upload was created by this run. All automatic tasks remain disabled.",
        "",
        "## Reproducibility",
        "",
        f"Candidate: {result['contract']['candidate_id']}. Data version: {result['contract']['data_version']}. Run: {result['run_id']}. Frozen source: {result['contract']['candidate']['code_snapshot_sha256']}. The contract records driver/source/data hashes, parameters, package versions, issue/cutoff policy and the historical assumption. Each date has an immutable replay receipt and model-origin reference. All target-contamination and training-date checks passed.",
        "",
        "The full interval CSV/Parquet, daily and monthly CSV files, model caches, receipts and hashed JSON reports are retained with this report. Never treat these historical receipts as prospective issuance.",
        "",
        "![Year overview](year_overview.png)",
        "",
        "![Monthly MAE](monthly_mae.png)",
        "",
        "![Failure and success cases](failure_and_success_examples.png)",
        "",
    ]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "daily": daily,
        "monthly": monthly,
        "regimes": regimes,
        "evidence_segments": segments,
        "verification": verification,
        "examples": {
            "highest_model_mae": worst,
            "largest_daily_improvement": best,
            "largest_daily_loss": biggest_loss,
        },
        "combined_parquet_sha256": sha(combined.read_bytes()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True, type=Path)
    parser.add_argument(
        "--plan", type=Path, default=Path("configs/active-evaluation.json")
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 1))
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        help="Exclusive date; default last complete day in the snapshot",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    plan_path = args.plan
    raw = json.loads(plan_path.read_bytes())
    if "active_evaluation_id" in raw:
        plan_path = project / raw["plan"]
    run(project, args.prices, plan_path, args.start, args.end)


if __name__ == "__main__":
    main()
