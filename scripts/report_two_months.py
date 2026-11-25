"""Verify a saved two-month comparison and build its public research report."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "data" / "plot-cache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dk2_forecast.storage import connect  # noqa: E402
from dk2_forecast.two_month import origin_for, validate_grid  # noqa: E402
from dk2_forecast.delivery_day import COPENHAGEN  # noqa: E402


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(folder):
    raw = (folder / "report.json").read_bytes()
    if sha(raw) != (folder / "report.sha256").read_text().strip():
        raise ValueError("Report hash mismatch")
    report = json.loads(raw)
    parquet = folder / "predictions.parquet"
    if sha(parquet.read_bytes()) != report["predictions_sha256"]:
        raise ValueError("Prediction hash mismatch")
    policy = report["policy"]
    if (policy["start"], policy["end_exclusive"]) != ("2026-09-26", "2026-11-26"):
        raise ValueError(
            "This dated public report builder requires the declared two-month window"
        )
    methods = report["point"]["extended"]["methods"]
    # Independent SQL aggregation verifies the saved scorer against actual saved rows.
    checked = {}
    with connect() as db:
        query = db.execute(
            "SELECT delivery_start_utc, actual, catboost, previous_day FROM read_parquet(?) ORDER BY delivery_start_utc",
            [str(parquet)],
        )
        rows = query.fetchall()
        validate_grid(
            [r[0] for r in rows],
            datetime.fromisoformat(policy["start"]).date(),
            datetime.fromisoformat(policy["end_exclusive"]).date(),
        )
        for method, expected in methods.items():
            mae, rmse, bias, within25, within50, count = db.execute(
                f'SELECT avg(abs("{method}" - actual)), sqrt(avg(pow("{method}" - actual, 2))), avg("{method}" - actual), avg(CAST(abs("{method}" - actual)<=25 AS INTEGER)), avg(CAST(abs("{method}" - actual)<=50 AS INTEGER)), count("{method}") FROM read_parquet(?)',
                [str(parquet)],
            ).fetchone()
            np.testing.assert_allclose(
                [mae, rmse, bias],
                [expected["mae"], expected["rmse"], expected["mean_signed_error"]],
                rtol=1e-12,
            )
            bands = report["point"]["slices"]["all"]["methods"][method]
            np.testing.assert_allclose(
                [within25, within50], [bands["within_25"], bands["within_50"]]
            )
            if count != report["paired_intervals"]:
                raise ValueError("Missing predictions in report")
            checked[method] = {
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
                "within_25": within25,
                "within_50": within50,
                "intervals": count,
            }
        flags = db.execute(
            "SELECT bool_and(replay), bool_or(eligible_for_prospective_scoring), count(DISTINCT delivery_start_utc) FROM read_parquet(?)",
            [str(parquet)],
        ).fetchone()
        assert flags == (True, False, report["paired_intervals"])
        daily_query = db.execute(
            "SELECT delivery_date, avg(actual), avg(catboost), avg(fixed_half_blend), avg(previous_day), avg(residual_ridge) FROM read_parquet(?) GROUP BY delivery_date ORDER BY delivery_date",
            [str(parquet)],
        )
        daily = daily_query.fetchall()
    verification = {
        "status": "passed",
        "report_sha256": sha(raw),
        "unique_utc_intervals": len(rows),
        "sql_metrics_match": checked,
        "replay_flags_verified": True,
    }
    (folder / "independent-verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    dates = [datetime.fromisoformat(row[0]) for row in daily]
    for index, label, color in [
        (1, "Actual", "black"),
        (2, "CatBoost", "tab:blue"),
        (3, "50/50 blend", "tab:green"),
        (4, "Previous day", "tab:orange"),
        (5, "Residual Ridge", "tab:purple"),
    ]:
        axes[0].plot(
            dates, [r[index] for r in daily], label=label, color=color, linewidth=1.2
        )
    axes[0].set(
        ylabel="Daily mean EUR/MWh",
        title="DK2: historical two-month comparison, 26 September–25 November 2026",
    )
    axes[0].legend(ncol=5)
    labels = list(methods)
    axes[1].barh(
        labels,
        [methods[name]["mae"] for name in labels],
        color=[
            {
                "previous_day": "tab:orange",
                "catboost": "tab:blue",
                "fixed_half_blend": "tab:green",
                "prior_only_gate": "tab:red",
                "residual_ridge": "tab:purple",
                "weekly_utc": "gray",
            }[name]
            for name in labels
        ],
    )
    axes[1].set(xlabel="Paired MAE, EUR/MWh (lower is better)")
    axes[1].invert_yaxis()
    figure = PROJECT / "reports" / "figures" / "two-month-2026-11-25.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    lines = [
        "# Two-month price-only improvement experiment",
        "",
        "26 September–25 November 2026 inclusive: 61 Danish delivery days, 5,860 unique UTC quarter-hours. All six methods and actuals have 100% coverage on the same intervals. October 25 has 100 intervals. This window was chosen before running these new comparisons to include the latest complete two months and November's known weakness.",
        "",
        "**Retrospective, selection-affected research.** These dates were already examined in the annual replay. No prospective evidence or trading-profit claim is produced. The frozen evaluation, candidate and receipts are unchanged; no scheduled tasks are registered or enabled.",
        "",
        "## Point results",
        "",
        "| Method | MAE EUR/MWh | RMSE | Bias (prediction − actual) | Within ±25 | Within ±50 | Day wins vs previous day |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in checked.items():
        wins = report["point"]["delivery_day_results"]["methods"][name]["wins"]
        lines.append(
            f"| {name} | {value['mae']:.2f} | {value['rmse']:.2f} | {value['bias']:+.2f} | {100 * value['within_25']:.1f}% | {100 * value['within_50']:.1f}% | {wins}/61 |"
        )
    lines += [
        "",
        "## Stress conditions",
        "",
        "Slices overlap. Extreme means absolute actual price ≥ the previously frozen development threshold of €276.019989/MWh. Realized regimes are used for scoring only.",
        "",
        "| Slice | Paired intervals | " + " | ".join(methods) + " |",
        "|---|---:|" + "---:|" * len(methods),
    ]
    for name in ("negative", "spike", "extreme"):
        values = report["point"]["slices"][name]
        lines.append(
            f"| {name} | {values['paired_intervals']} | "
            + " | ".join(
                f"{values['methods'][m]['mae']:.2f}"
                if values["paired_intervals"]
                else "n/a"
                for m in methods
            )
            + " |"
        )
    lines += [
        "",
        "## Monthly MAE",
        "",
        "| Month | Intervals | " + " | ".join(methods) + " |",
        "|---|---:|" + "---:|" * len(methods),
    ]
    for name, group in report["point"]["month"].items():
        lines.append(
            f"| {name} | {group['intervals']} | "
            + " | ".join(f"{group['methods'][m]['mae']:.2f}" for m in methods)
            + " |"
        )
    lines += [
        "",
        "## Uncertainty intervals",
        "",
        "Nominal coverage is 80%. Lower interval score is better; excessive width is penalized. These are empirical intervals without formal coverage guarantees under temporal dependence or drift. Raw quantiles and their crossing diagnostics remain in the full artifact.",
        "",
        "| Method | Intervals | Coverage | Mean width | Interval score | Raw crossings |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in report["intervals"].items():
        lines.append(
            f"| {name} | {value['intervals']} | {100 * value['coverage_80']:.1f}% | {value['mean_width']:.2f} | {value['mean_interval_score_80']:.2f} | {value['raw_crossings']} |"
        )
    baseline = methods["previous_day"]["mae"]
    original = methods["catboost"]["mae"]
    residual = methods["residual_ridge"]["mae"]
    gate_defaults = sum(w["fallback_half_blend"] for w in report["gate_weights"])
    operational = {
        "previous_day_weekly_fallback_intervals": report[
            "previous_day_weekly_fallbacks"
        ],
        "model_fallback_intervals": report["model_fallback_intervals"],
        "gate_default_regime_blocks": gate_defaults,
        "gate_total_regime_blocks": len(report["gate_weights"]),
        "gate_default_target_intervals": sum(
            next(
                w["fallback_half_blend"]
                for w in report["gate_weights"]
                if w["origin"]
                == str(
                    origin_for(
                        row[0].astimezone(COPENHAGEN).date(),
                        datetime.fromisoformat(policy["weekly_anchor"]).date(),
                    )
                )
                and w["previous_price_regime"] == int(np.digitize(row[3], [0, 200]))
            )
            for row in rows
        ),
    }
    lines += [
        "",
        "## What improved and what did not",
        "",
        f"Residual Ridge has the lowest overall MAE in this comparison: {100 * (1 - residual / original):.1f}% below the original CatBoost and {100 * (1 - residual / baseline):.1f}% below the previous-day benchmark. Its November MAE is €39.98/MWh versus CatBoost €45.19 and the benchmark €42.49. The fixed blend is better in November (€38.87) and on high-price intervals (€43.71). No method wins every slice.",
        "",
        "Extreme-price MAE remains worse than the previous-day benchmark for every challenger. Residual Ridge still underpredicts by €15.73/MWh on average. There are zero negative-price target intervals, so this run provides no evidence that the known negative-price weakness is fixed. The next test of that claim needs dates containing negative prices.",
        "",
        "Rolling quantile calibration raises nominal-80% interval coverage from 56.0% raw (66.0% with the retained fixed margin) to 78.1%, at the cost of wider intervals. The simple CatBoost historical-residual interval achieves 77.8% coverage and a better interval score with narrower bounds. Calibration remains conditional on this already observed period.",
        "",
        f"Operational and model-only point results coincide: {report['model_fallback_intervals']} model fallback intervals and {report['previous_day_weekly_fallbacks']} previous-day weekly fallback intervals. Gate default weights occur in {gate_defaults}/{len(report['gate_weights'])} regime/origin blocks; these counts include regimes that may have no target intervals. Per-block sample counts and reasons are retained in gate_weights. Baseline day-win count is zero because it is the reference; its 61 comparisons are ties.",
        "",
        "## Prior-only protocol and interpretation",
        "",
        "- Original CatBoost forecasts are reused from their hashed daily replay receipts, preserving their exact predictions and weekly Friday origins. Prior-day/weekly values and actuals are checked against the audited price snapshot; target/future price contamination is tested on every reference day.",
        "- The fixed blend gives CatBoost and the previous-day baseline equal weight. It has no fitted weights.",
        f"- Residual Ridge uses the same 210 price/calendar features, fitted imputation and scaling within each training origin. Alpha {report['selection']['selected_alpha']} was chosen from [100, 1000, 10000] on 31 July–24 September chronological weekly validation and saved before outer scoring. Training expands and excludes origin-day labels.",
        "- The gate chooses CatBoost weights from [0, .25, .5, .75, 1] separately for available previous-day prices <0, 0–200 and ≥200. It uses only prior 56-day out-of-fold errors, updates weekly and defaults to 50/50 with fewer than 192 eligible intervals. No actual target regime is an inference input.",
        "- Earlier outer outcomes may enter later weekly gate/calibration updates, as declared before execution. This evaluates a sequential updating policy; it does not evaluate one unchanging fitted gate over all 61 days.",
        "- LightGBM 0.1/0.5/0.9 quantiles use fixed default candidate settings, 160 rounds and expanding training, refitted every 28 days with the declared 20 November anchor. Raw models also run on earlier warm-up dates to provide prior forecast errors. Sorted bounds are compared with the retained development margin and a weekly 28-day prior-only expansion margin. The rolling method falls back to sorted raw bounds with fewer than 14×92 eligible prior intervals.",
        "- Historical-residual benchmarks add prior 56-day residual quantiles to the previous-day or CatBoost point forecast; parameters update weekly. Interval calibration by month, local hour and realized regime, pinball loss and interval scores are in the full JSON.",
        "- Point breakdowns include month, local hour, weekday/weekend, forecast horizon, available previous-day price and predicted price. Large-error examples, daily scores, denominators, gate weights, fallback counts and model/training hashes are retained under data/two_month.",
        "- Historical publication/revision vintages are unavailable: issue-day delivery prices are assumed known by the 09:50 Copenhagen cutoff. Features never use target-day realized prices. Scheduled issue is 10:00 on D−1; actual creation timestamps remain today's replay timestamps.",
        "- Load/wind/solar features are excluded because no verified pre-cutoff historical forecast vintages were supplied. Realized target-day fundamentals are never substituted. No strategy, exposure or execution-cost specification exists for a defensible trading P&L simulation. Live reliability remains untested by this replay.",
        "- The best observed method remains an experimental challenger. Selection among these new outcomes requires a subsequent untouched prospective evaluation; it does not replace the frozen candidate.",
        "",
        f"New fitting CPU wall time: {report['new_fit_seconds']:.1f} seconds across {len(report['fits'])} fits, one training thread. Total comparison wall time: {report['total_seconds']:.1f} seconds, including source validation and scoring. Existing annual CatBoost training time is excluded because those forecasts were reused.",
        "",
        f"Run identity: `{folder.name}`. Source replay: `{policy['source_replay']}`. Report SHA-256: `{sha(raw)}`. SQL verification independently reproduces MAE, RMSE, bias and accuracy bands for all methods, verifies the full UTC grid and replay flags.",
        "",
        "![Two-month comparison](figures/two-month-2026-11-25.png)",
        "",
    ]
    target = PROJECT / "reports" / "two-month-2026-11-25.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "run_id": folder.name,
        "source_report_sha256": sha(raw),
        "policy": policy,
        "days": report["days"],
        "paired_intervals": report["paired_intervals"],
        "metrics": checked,
        "stress_slices": report["point"]["slices"],
        "monthly": report["point"]["month"],
        "intervals": report["intervals"],
        "selection": report["selection"],
        "daily_wins": report["point"]["delivery_day_results"],
        "new_fit_seconds": report["new_fit_seconds"],
        "total_seconds": report["total_seconds"],
        "verification": verification,
        "operational": operational,
    }
    target.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(target)
    print(json.dumps(checked, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    run(parser.parse_args().folder)
