"""Build the public summary from retained research artifacts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--weather", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/research.md"))
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    point = report["point"]
    candidate = (
        "weekly_refit"
        if report["selection"]["retraining_days"] == 7
        else report["selection"]["model"]
    )
    lines = [
        "# DK2 price-curve experiment",
        "",
        f"Development used 2 April through 20 September 2026, with {point['expected_intervals']:,} quarter-hours. "
        "Every method is scored on identical paired intervals against the previous Danish day.",
        "",
        "| Method | Coverage | MAE | RMSE | Within €25 | Within €50 | Day wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    reference_names = ["previous_day", "weekly_utc", "weekday_persistence"]
    names = reference_names + [
        name for name in point["extended"]["methods"] if name not in reference_names
    ]
    for name in names:
        values = point["extended"]["methods"][name]
        bands = point["slices"]["all"]["methods"][name]
        wins = point["delivery_day_results"]["methods"][name]["wins"]
        lines.append(
            f"| {name} | {point['coverage'][name]['share']:.0%} | {values['mae']:.2f} | {values['rmse']:.2f} | "
            f"{bands['within_25']:.1%} | {bands['within_50']:.1%} | {wins} |"
        )
    selection = report["selection"]
    lines += [
        "",
        "MAE and RMSE are in EUR/MWh. This development result includes model selection and is optimistic.",
        f"The research proposal is {candidate}, using {selection['training_weeks'] or 'expanding'} history "
        f"and {selection['retraining_days']}-day point-model refitting. The original frozen live candidate remains CatBoost.",
        "",
        "## Window comparison",
        "",
        "| History | MAE | Fit seconds |",
        "| --- | ---: | ---: |",
    ]
    for name, values in report["fits"].items():
        if name.startswith("window_"):
            lines.append(
                f"| {name} | {values['mae']:.2f} | {values['fit_seconds']:.1f} |"
            )
    lines += [
        "",
        "## Search budgets",
        "",
        "| Tree | Completed | Failed | Seconds |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in report["searches"].items():
        completed = sum(trial["state"] == "COMPLETE" for trial in values["trials"])
        failed = sum(trial["state"] == "FAIL" for trial in values["trials"])
        lines.append(
            f"| {name} | {completed} | {failed} | {values['elapsed_seconds']:.1f} |"
        )
    lines += [
        "",
        "Each search had a ceiling of 30 sequential trials and a 180-second budget. "
        "A trial may finish beyond the budget. Stored studies retain trial parameters, fold dates and state.",
        "",
        "## Price regimes",
        "",
        "| Slice | Paired intervals | Previous day MAE | Research proposal MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in point["slices"].items():
        if name != "all":
            baseline = values["methods"]["previous_day"]["mae"]
            model = values["methods"][candidate]["mae"]
            lines.append(
                f"| {name} | {values['paired_intervals']} | {baseline:.2f} | {model:.2f} |"
            )
    confidence = point["paired_daily_uncertainty"][candidate]
    lines += [
        "",
        f"The development extreme threshold is |actual| ≥ {report['scoring']['extreme_threshold_eur_per_mwh']:.6f} EUR/MWh. "
        "The high-price threshold is 200 EUR/MWh.",
        f"Paired daily MAE difference was {confidence['mean_daily_mae_difference']:.2f} EUR/MWh, with a "
        f"seven-day block-bootstrap 95% interval [{confidence['interval_95_percent'][0]:.2f}, {confidence['interval_95_percent'][1]:.2f}]. "
        "This interval is unadjusted for model selection and multiple comparisons.",
        "",
        "## Quantile calibration",
        "",
        "| Interval | Observed coverage | Mean width EUR/MWh | Raw crossings |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("quantiles", "calibrated_quantiles"):
        values = report[name]
        lines.append(
            f"| {name} | {values['coverage_80']:.1%} | {values['mean_width']:.2f} | {values['raw_crossings']} |"
        )
    lines += [
        "",
        "The nominal coverage is 80%. Calibration uses prior validation folds. "
        "The aggregate improvement hides substantial monthly variation, including undercoverage in September. "
        "Temporal dependence limits exchangeability guarantees.",
        "",
        "![Representative delivery curves](figures/delivery_examples.png)",
        "",
        "![Rolling errors](figures/rolling_error.png)",
        "",
        "![Calibration by month](figures/interval_calibration.png)",
    ]
    weather = json.loads(args.weather.read_bytes()) if args.weather else None
    if weather:
        lines += [
            "",
            "## Conditional weather ablation",
            "",
            f"The common sample contains {weather['paired_intervals']:,} intervals "
            f"({weather['coverage']:.1%} of development). "
            "Historical run availability is assumed eight hours after initialization.",
            "",
            "| Features | MAE EUR/MWh |",
            "| --- | ---: |",
            *[
                f"| {name} | {weather['results'][name]['mae']:.2f} |"
                for name in (
                    "price_calendar_11",
                    "price_curves",
                    "plus_temperature",
                    "plus_temperature_wind",
                    "plus_temperature_wind_radiation",
                    "plus_weather_linear_interpolation",
                )
            ],
        ]
    if args.diagnostic:
        diagnostic = json.loads(args.diagnostic.read_bytes())
        lines += [
            "",
            "## Already observed diagnostic",
            "",
            "These 5,380 intervals cover 21 September through 15 November 2026 and were already inspected.",
            "",
            "| Method | MAE EUR/MWh |",
            "| --- | ---: |",
            *[
                f"| {name} | {values['mae']:.2f} |"
                for name, values in diagnostic["point"]["extended"]["methods"].items()
            ],
        ]
    lines += [
        "",
        "## Operational status and remaining work",
        "",
        "The original frozen comparison has no eligible publication for 18 or 19 November. "
        "Those days are recorded as missed. The next original training block begins with delivery 16 December.",
        "A separately labelled 18 November replay verified the five-method forecast and scoring path. "
        "Its full table appears in the [operational audit](../docs/audit.md).",
        "",
        "Historical source vintages remain assumptions. Eligible load and renewable forecasts require "
        "versioned collection and access credentials. Neighbour and interconnector integrations, seasonal "
        "coverage, confirmatory comparison tests, and neural models remain subsequent work.",
        "",
        f"Price data version {report['data_version']}.",
        f"Full experiment artifact {args.report.parent.name}.",
        "The saved JSON additionally contains signed errors, absolute-error percentiles, local calendar groups, "
        "pinball losses, hourly calibration, runtime and bootstrap details. See the README for reproduction commands.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "units": "EUR/MWh",
        "primary_benchmark": "previous_day",
        "period": report["period"],
        "selection": selection,
        "data_version": report["data_version"],
        "experiment": args.report.parent.name,
        "point": point["extended"],
        "coverage": point["coverage"],
        "slices": point["slices"],
        "quantiles": {
            name: {
                key: report[name][key]
                for key in (
                    "intervals",
                    "pinball_loss",
                    "coverage_80",
                    "mean_width",
                    "raw_crossings",
                )
            }
            for name in ("quantiles", "calibrated_quantiles")
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
