"""Run a bounded retrospective comparison; never publish prospective forecasts."""

import argparse
import csv
from datetime import UTC, date, datetime, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402

from dk2_forecast.audit_snapshot import verify_snapshot  # noqa: E402
from dk2_forecast.delivery_day import COPENHAGEN, forecast_issue_utc  # noqa: E402
from dk2_forecast.forecast import PriceHistory  # noqa: E402
from dk2_forecast.operations import exclusive_run  # noqa: E402
from dk2_forecast.research import fit_model, predict_model  # noqa: E402
from dk2_forecast.research_features import all_price_features, daily_price_features  # noqa: E402
from dk2_forecast.research_metrics import interval_metrics, point_metrics  # noqa: E402
from dk2_forecast.storage import connect, publish_bytes, write_parquet  # noqa: E402
from dk2_forecast.two_month import (  # noqa: E402
    gate_weights,
    origin_for,
    residual_bounds,
    retain_selection,
    rolling_margin,
    validate_grid,
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"
    ).encode()


def load_rows(path):
    with connect() as db:
        query = db.execute(
            "SELECT * FROM read_parquet(?) ORDER BY delivery_start_utc", [str(path)]
        )
        names = [col[0] for col in query.description]
        return [dict(zip(names, row, strict=True)) for row in query.fetchall()]


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    started = perf_counter()
    replay = args.replay.resolve()
    contract = json.loads((replay / "contract.json").read_bytes())
    source_report_path = Path((replay / "latest-report.txt").read_text().strip())
    raw_report = source_report_path.read_bytes()
    if sha(raw_report) != source_report_path.stem:
        raise ValueError("Annual report hash mismatch")
    source_report = json.loads(raw_report)
    if source_report["contract"] != contract or sha(encoded(contract)) != replay.name:
        raise ValueError("Annual replay identity mismatch")
    audit = verify_snapshot(args.prices, PROJECT / "data")
    if audit["parquet_sha256"] != contract["prices_sha256"]:
        raise ValueError(
            "Use the same audited actual-price snapshot as the reference replay"
        )
    history = PriceHistory.load(args.prices)
    start, end = args.start, args.end
    if start >= end:
        raise ValueError("start must precede exclusive end")
    # The candidate was selected through September 20; do not use it for earlier tests.
    if start < date(2026, 9, 26):
        raise ValueError(
            "This experiment requires dates after the original selection period"
        )
    anchor = date.fromisoformat(contract["refit_anchor"])
    first_origin = origin_for(start, anchor)
    inner_start = first_origin - timedelta(days=56)
    warm_start = min(inner_start, origin_for(start, anchor, 28) - timedelta(days=35))
    receipts = {r["day"]: r for r in source_report["forecast_receipts"]}
    rows = []
    receipt_hashes = []
    for n in range((end - warm_start).days):
        day = warm_start + timedelta(days=n)
        folder = replay / "forecasts" / str(day)
        raw = (folder / "forecast.json").read_bytes()
        meta = json.loads(raw)
        parquet = folder / "forecast.parquet"
        if (
            sha(raw) != receipts[str(day)]["receipt_sha256"]
            or sha(parquet.read_bytes()) != meta["forecast_sha256"]
        ):
            raise ValueError(f"Corrupted daily reference {day}")
        if (
            not meta["replay"]
            or meta["eligible_for_prospective_scoring"]
            or meta["run_id"] != replay.name
        ):
            raise ValueError("Reference must be the identified historical replay")
        if (
            meta["candidate_id"] != contract["candidate_id"]
            or meta["prices_sha256"] != contract["prices_sha256"]
        ):
            raise ValueError("Candidate or data mismatch")
        if date.fromisoformat(meta["training_origin"]) != origin_for(
            day, anchor
        ) or date.fromisoformat(
            meta["training_last_delivery_date"]
        ) >= date.fromisoformat(meta["training_origin"]):
            raise ValueError("Invalid training origin")
        daily = load_rows(parquet)
        stamps, x, _, lags = daily_price_features(history, day)
        validate_grid(
            [r["delivery_start_utc"] for r in daily], day, day + timedelta(days=1)
        )
        if [r["delivery_start_utc"] for r in daily] != stamps:
            raise ValueError("Reference keys differ from eligible feature keys")
        for i, row in enumerate(daily):
            if (
                row["actual"] != history.prices[row["delivery_start_utc"]]
                or row["previous_day"] != x[i, 0]
                or row["weekly_utc"] != x[i, 1]
            ):
                raise ValueError(
                    "Actual or benchmark differs from audited price inputs"
                )
            if (
                lags[i].source_utc is not None
                and lags[i].source_utc.astimezone(COPENHAGEN).date() >= day
            ):
                raise ValueError("Previous-day source reads target delivery day")
        # Current feature implementation must also be invariant to all target/future prices.
        altered = PriceHistory(
            {
                t: p if t.astimezone(COPENHAGEN).date() < day else 99999.0
                for t, p in history.prices.items()
            },
            history.data_version,
        )
        _, contaminated, _, _ = daily_price_features(altered, day)
        if not np.array_equal(x, contaminated, equal_nan=True):
            raise ValueError("Target-price contamination changed features")
        rows.extend(daily)
        receipt_hashes.append({"day": str(day), "sha256": sha(raw)})
    stamps = [r["delivery_start_utc"] for r in rows]
    validate_grid(stamps, warm_start, end)
    days = np.asarray([t.astimezone(COPENHAGEN).date() for t in stamps])
    actual = np.asarray([r["actual"] for r in rows])
    cat = np.asarray([r["prediction"] for r in rows])
    previous = np.asarray([r["previous_day"] for r in rows])
    source_files = [
        Path(__file__),
        PROJECT / "dk2_forecast" / "two_month.py",
        *[
            PROJECT / "dk2_forecast" / f"{name}.py"
            for name in (
                "research",
                "research_features",
                "research_metrics",
                "experiment_metrics",
                "price_lags",
                "experiment_models",
                "forecast",
                "delivery_day",
            )
        ],
    ]
    policy = {
        "schema": 1,
        "status": "retrospective_selection_affected_not_prospective",
        "replay": True,
        "eligible_for_prospective_scoring": False,
        "start": str(start),
        "end_exclusive": str(end),
        "source_report_sha256": sha(raw_report),
        "source_replay": replay.name,
        "source_contract_sha256": sha((replay / "contract.json").read_bytes()),
        "candidate_id": contract["candidate_id"],
        "prices_sha256": audit["parquet_sha256"],
        "data_version": history.data_version,
        "scoring": contract["scoring"],
        "source_files": {
            str(p.relative_to(PROJECT)): sha(p.read_bytes()) for p in source_files
        },
        "versions": {
            k: importlib.metadata.version(k)
            for k in ("numpy", "scikit-learn", "lightgbm", "catboost", "duckdb")
        },
        "weekly_anchor": str(anchor),
        "refit_days": 7,
        "training": "expanding labels strictly before origin; pipeline fitted per origin",
        "residual_model": "Ridge predicts actual minus previous-day price using common 210 features",
        "inner_validation": {
            "start": str(inner_start),
            "end_exclusive": str(first_origin),
            "alpha_grid": [100, 1000, 10000],
            "selection": "lowest pooled chronological OOF MAE; freeze alpha before outer period",
        },
        "fixed_blend_cat_weight": 0.5,
        "gate": {
            "prior_days": 56,
            "minimum_intervals_per_regime": 192,
            "weights": [0, 0.25, 0.5, 0.75, 1],
            "default": 0.5,
            "regimes_available_previous_price": [0, 200],
            "update_days": 7,
        },
        "intervals": {
            "levels": [0.1, 0.5, 0.9],
            "raw_lightgbm": "160 iterations, default frozen quantile settings, 28-day refit anchored 2026-11-20",
            "fixed_development_margin": 13.133146451383316,
            "rolling_margin_prior_days": 28,
            "update_days": 7,
            "minimum_prior_intervals": 14 * 92,
            "fallback": "sorted raw bounds",
            "residual_benchmark_prior_days": 56,
        },
        "availability": contract["historical_availability_assumption"],
        "excluded": {
            "fundamentals": "No verified pre-cutoff historical load/wind/solar forecast vintages supplied; no realized target-day inputs substituted",
            "trading_pnl": "No specified actionable strategy, volumes, execution prices or costs; MAE is not P&L",
            "live_operations": "Historical replay cannot establish on-time operation; tasks remain disabled",
        },
        "evaluation_policy": "All target dates already inspected. No outer outcome may select alpha, gate policy or candidate. Gate and calibration updates may use earlier outer outcomes only on declared weekly origins.",
    }
    identity = sha(encoded(policy))
    output = PROJECT / "data" / "two_month" / identity
    output.mkdir(parents=True, exist_ok=True)
    publish_bytes(output / "plan.json", encoded(policy))
    for p in source_files:
        publish_bytes(output / "source" / p.relative_to(PROJECT), p.read_bytes())
    print(f"OUTPUT {output}", flush=True)
    with exclusive_run(output / "run.lock"):
        if (output / "report.json").exists():
            saved_report = (output / "report.json").read_bytes()
            if sha(saved_report) != (output / "report.sha256").read_text().strip():
                raise ValueError("Completed report hash mismatch")
            result = json.loads(saved_report)
            if result["policy"] != policy or result["predictions_sha256"] != sha(
                (output / "predictions.parquet").read_bytes()
            ):
                raise ValueError("Completed run artifact mismatch")
            print("Verified completed run; no changes", flush=True)
            return output
        all_stamps, matrix, _ = all_price_features(history)
        all_days = np.asarray([t.astimezone(COPENHAGEN).date() for t in all_stamps])
        labels = np.asarray([history.prices.get(t, np.nan) for t in all_stamps])
        mapping = {t: i for i, t in enumerate(all_stamps)}
        row_indices = np.asarray([mapping[t] for t in stamps])
        fits = []

        def fit_at(origin, family, params, correction=False):
            eligible = (all_days < origin) & np.isfinite(labels)
            if correction:
                eligible &= np.isfinite(matrix[:, 0])
            x, y = matrix[eligible], labels[eligible].copy()
            if correction:
                y -= x[:, 0]
            key = sha(
                encoded(
                    {
                        "origin": origin,
                        "family": family,
                        "params": params,
                        "correction": correction,
                    }
                )
            )
            cached = output / "models" / key
            xhash = sha(x.tobytes() + y.tobytes())
            if cached.with_suffix(".json").exists():
                meta = json.loads(cached.with_suffix(".json").read_bytes())
                raw = cached.with_suffix(".pkl").read_bytes()
                if meta["training_sha256"] != xhash or meta["model_sha256"] != sha(raw):
                    raise ValueError("Model cache hash mismatch")
                model = pickle.loads(raw)
            else:
                model, seconds = fit_model(family, x, y, params)
                raw = pickle.dumps(model)
                meta = {
                    "origin": str(origin),
                    "family": family,
                    "parameters": params,
                    "correction": correction,
                    "training_rows": len(y),
                    "last_label": str(max(all_days[eligible])),
                    "training_sha256": xhash,
                    "model_sha256": sha(raw),
                    "fit_seconds": seconds,
                    "created_at_utc": datetime.now(UTC).isoformat(),
                }
                publish_bytes(cached.with_suffix(".pkl"), raw)
                publish_bytes(cached.with_suffix(".json"), encoded(meta))
            if date.fromisoformat(meta["last_label"]) >= origin:
                raise ValueError("Training label at or after origin")
            fits.append(meta)
            print(
                f"FIT {origin} {family} {params} {meta['fit_seconds']:.2f}s", flush=True
            )
            return model

        inner = (days >= inner_start) & (days < first_origin)
        inner_scores = []
        for alpha in policy["inner_validation"]["alpha_grid"]:
            pred = np.full(len(rows), np.nan)
            for origin in sorted({origin_for(day, anchor) for day in days[inner]}):
                mask = inner & (days >= origin) & (days < origin + timedelta(days=7))
                model = fit_at(origin, "ridge", {"alpha": alpha}, True)
                pred[mask] = previous[mask] + predict_model(
                    model, matrix[row_indices[mask]]
                )
            if not np.isfinite(pred[inner]).all():
                raise ValueError("Incomplete inner predictions")
            inner_scores.append(
                {
                    "alpha": alpha,
                    "mae": float(abs(actual[inner] - pred[inner]).mean()),
                    "intervals": int(inner.sum()),
                }
            )
        selected = min(inner_scores, key=lambda r: r["mae"])["alpha"]
        selection = retain_selection(output / "selection.json", inner_scores, selected)
        outer = (days >= start) & (days < end)
        residual = np.full(len(rows), np.nan)
        gate = np.full(len(rows), np.nan)
        weights, calibration = [], []
        residual_intervals = {
            name: np.full((len(rows), 3), np.nan)
            for name in ("previous_day_residual_56d", "catboost_residual_56d")
        }
        raw_q = np.full((len(rows), 3), np.nan)
        for origin in sorted({origin_for(day, anchor, 28) for day in days}):
            mask = (days >= origin) & (days < origin + timedelta(days=28))
            for j, alpha in enumerate((0.1, 0.5, 0.9)):
                model = fit_at(
                    origin, "lightgbm", {"objective": "quantile", "alpha": alpha}
                )
                raw_q[mask, j] = predict_model(model, matrix[row_indices[mask]])
        rolling_q = np.sort(raw_q, axis=1)
        for origin in sorted({origin_for(day, anchor) for day in days[outer]}):
            mask = outer & (days >= origin) & (days < origin + timedelta(days=7))
            model = fit_at(origin, "ridge", {"alpha": selected}, True)
            residual[mask] = previous[mask] + predict_model(
                model, matrix[row_indices[mask]]
            )
            records = gate_weights(days, actual, cat, previous, origin)
            weights.extend(records)
            w = np.asarray([r["candidate_weight"] for r in records])[
                np.digitize(previous[mask], [0, 200])
            ]
            gate[mask] = w * cat[mask] + (1 - w) * previous[mask]
            for name, center in (
                ("previous_day_residual_56d", previous),
                ("catboost_residual_56d", cat),
            ):
                offsets, count = residual_bounds(days, actual, center, origin)
                residual_intervals[name][mask] = center[mask, None] + offsets
                calibration.append(
                    {
                        "origin": str(origin),
                        "method": name,
                        "prior_intervals": count,
                        "offsets": offsets.tolist(),
                    }
                )
            margin, count = rolling_margin(days, actual, raw_q, origin)
            rolling_q[mask, 0] -= margin
            rolling_q[mask, 2] += margin
            calibration.append(
                {
                    "origin": str(origin),
                    "method": "rolling_quantile_28d",
                    "margin": margin,
                    "prior_intervals": count,
                    "fallback_raw": count < 14 * 92,
                }
            )
        target_stamps = [t for t, use in zip(stamps, outer, strict=True) if use]
        validate_grid(target_stamps, start, end)
        predictions = {
            "previous_day": previous[outer],
            "weekly_utc": np.asarray([r["weekly_utc"] for r in rows])[outer],
            "catboost": cat[outer],
            "fixed_half_blend": ((cat + previous) / 2)[outer],
            "residual_ridge": residual[outer],
            "prior_only_gate": gate[outer],
        }
        if not all(np.isfinite(v).all() for v in predictions.values()):
            raise ValueError(
                "Incomplete model predictions; do not silently drop intervals"
            )
        point = point_metrics(
            target_stamps, actual[outer], predictions, contract["scoring"]
        )
        fixed_q = np.sort(raw_q, axis=1)
        fixed_q[:, 0] -= policy["intervals"]["fixed_development_margin"]
        fixed_q[:, 2] += policy["intervals"]["fixed_development_margin"]
        qsets = {
            "raw_lightgbm": raw_q,
            "fixed_development_calibration": fixed_q,
            "rolling_quantile_28d": rolling_q,
            **residual_intervals,
        }
        intervals = {
            name: interval_metrics(target_stamps, actual[outer], q[outer])
            for name, q in qsets.items()
        }
        conditional = {}
        horizon = np.asarray(
            [
                (
                    t - forecast_issue_utc(t.astimezone(COPENHAGEN).date())
                ).total_seconds()
                / 3600
                for t in target_stamps
            ]
        )
        for group, keys in {
            "horizon_6h": np.floor(horizon / 6) * 6,
            "predicted_price_bin": np.digitize(cat[outer], [0, 100, 200]),
            "previous_price_bin": np.digitize(previous[outer], [0, 100, 200]),
        }.items():
            conditional[group] = {
                str(key): point_metrics(
                    [
                        t
                        for t, take in zip(target_stamps, keys == key, strict=True)
                        if take
                    ],
                    actual[outer][keys == key],
                    {name: v[keys == key] for name, v in predictions.items()},
                    contract["scoring"],
                    resamples=100,
                )
                for key in sorted(set(keys))
            }
        output_rows = []
        for i, t in enumerate(target_stamps):
            source_row = rows[np.flatnonzero(outer)[i]]
            output_rows.append(
                {
                    "price_area": "DK2",
                    "delivery_start_utc": t.isoformat(),
                    "delivery_date": str(t.astimezone(COPENHAGEN).date()),
                    "scheduled_issue_utc": forecast_issue_utc(
                        t.astimezone(COPENHAGEN).date()
                    ).isoformat(),
                    "replay": True,
                    "eligible_for_prospective_scoring": False,
                    "training_origin": source_row["training_origin"],
                    "previous_source_utc": source_row["previous_source_utc"],
                    "weekly_fallback": source_row["weekly_fallback"],
                    "actual": float(actual[outer][i]),
                    **{name: float(v[i]) for name, v in predictions.items()},
                    **{
                        f"{name}_q{level}": float(q[outer][i, j])
                        for name, q in qsets.items()
                        for j, level in enumerate((10, 50, 90))
                    },
                }
            )
        save_csv(output / "predictions.csv", output_rows)
        schema = [
            {
                key: "BOOLEAN"
                if isinstance(value, bool)
                else "DOUBLE"
                if isinstance(value, float)
                else "TIMESTAMPTZ"
                if key
                in ("delivery_start_utc", "scheduled_issue_utc", "previous_source_utc")
                else "VARCHAR"
                for key, value in output_rows[0].items()
            }
        ]
        with connect() as db:
            db.execute(
                "CREATE TABLE predictions AS SELECT unnest(from_json_strict(?, ?), recursive := true)",
                [json.dumps(output_rows, default=str), json.dumps(schema)],
            )
            write_parquet(
                db,
                "SELECT * FROM predictions ORDER BY delivery_start_utc",
                output / "predictions.parquet",
            )
        large = np.argsort(abs(predictions["catboost"] - actual[outer]))[-20:][::-1]
        daily = []
        target_days = days[outer]
        for day in sorted(set(target_days)):
            mask = target_days == day
            daily.append(
                {
                    "delivery_date": str(day),
                    "intervals": int(mask.sum()),
                    **{
                        f"{name}_mae": float(abs(v[mask] - actual[outer][mask]).mean())
                        for name, v in predictions.items()
                    },
                }
            )
        save_csv(output / "daily_scores.csv", daily)
        result = {
            "policy": policy,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "eligible_for_prospective_scoring": False,
            "selection": selection,
            "point": point,
            "conditional": conditional,
            "intervals": intervals,
            "gate_weights": weights,
            "calibration": calibration,
            "fits": fits,
            "new_fit_seconds": sum(r["fit_seconds"] for r in fits),
            "total_seconds": perf_counter() - started,
            "reference_receipts": receipt_hashes,
            "target_contamination_days_checked": len(receipt_hashes),
            "previous_day_weekly_fallbacks": sum(
                r["weekly_fallback"] for r in output_rows
            ),
            "model_fallback_intervals": 0,
            "days": len(daily),
            "paired_intervals": len(output_rows),
            "predictions_sha256": sha((output / "predictions.parquet").read_bytes()),
            "large_error_examples": [output_rows[i] for i in large],
            "actual_price_coverage": 1.0,
        }
        publish_bytes(output / "report.json", encoded(result))
        publish_bytes(
            output / "report.sha256",
            (sha((output / "report.json").read_bytes()) + "\n").encode(),
        )
        print(
            json.dumps(
                {
                    name: value["mae"]
                    for name, value in point["extended"]["methods"].items()
                },
                indent=2,
            ),
            flush=True,
        )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--prices", type=Path, required=True)
    p.add_argument("--start", type=date.fromisoformat, default=date(2026, 9, 26))
    p.add_argument("--end", type=date.fromisoformat, default=date(2026, 11, 26))
    print(run(p.parse_args()), flush=True)


if __name__ == "__main__":
    main()
