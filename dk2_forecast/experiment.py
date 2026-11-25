import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from importlib.metadata import version
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import optuna

from dk2_forecast.backtest import FOLD_DAYS, fold_ranges
from dk2_forecast.delivery_day import forecast_issue_utc
from dk2_forecast.experiment_metrics import SCORING, evaluate_comparison
from dk2_forecast.experiment_models import (
    COMMON_FEATURES,
    DEFAULTS,
    SEED,
    ComparisonTraining,
    daily_inputs,
    fit_predict,
)
from dk2_forecast.forecast import BASELINE_VERSION, PriceHistory
from dk2_forecast.ingestion import PRICE_AREA
from dk2_forecast.price_lags import PREVIOUS_DAY_VERSION, local_day
from dk2_forecast.storage import connect, digest, publish_bytes, write_parquet

DEVELOPMENT_START = date(2026, 1, 1)
DIAGNOSTIC_START = date(2026, 9, 21)
DIAGNOSTIC_END = date(2026, 11, 16)
DIAGNOSTIC_MAE = {"weekly": 83.74, "legacy_lightgbm": 64.71, "previous_day": 50.37}
FRESH_DAYS = 56


def date_range(start, end):
    return (start + timedelta(days=offset) for offset in range((end - start).days))


def json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=str) + "\n"
    ).encode()


@dataclass
class Evaluation:
    timestamps: list
    matrix: np.ndarray
    actual: np.ndarray
    baselines: dict
    previous: list
    blocks: list

    @classmethod
    def build(cls, history, ranges):
        timestamps, matrices, weekly, previous, blocks = [], [], [], [], []
        for start, end in ranges:
            begin = len(timestamps)
            for day in date_range(start, end):
                times, matrix, baseline, lags = daily_inputs(history, day)
                timestamps.extend(times)
                matrices.append(matrix)
                weekly.extend(baseline)
                previous.extend(lags)
            blocks.append((start, end, slice(begin, len(timestamps))))
        return cls(
            timestamps,
            np.concatenate(matrices),
            np.asarray(
                [history.prices.get(stamp) for stamp in timestamps], dtype=float
            ),
            {
                "weekly": np.asarray(weekly, dtype=float),
                "previous_day": np.asarray(
                    [lag.price for lag in previous], dtype=float
                ),
            },
            previous,
            blocks,
        )

    def fit(self, training, spec):
        predictions = np.empty(len(self.timestamps))
        results = []
        paired = np.isfinite(self.actual) & np.logical_and.reduce(
            [np.isfinite(values) for values in self.baselines.values()]
        )
        for origin, end, indices in self.blocks:
            x, y = training.arrays(origin)
            prediction, seconds = fit_predict(spec, x, y, self.matrix[indices])
            predictions[indices] = prediction
            mask = paired[indices]
            results.append(
                {
                    "start": origin,
                    "end_exclusive": end,
                    "training_rows": len(y),
                    "paired_intervals": int(mask.sum()),
                    "mae": float(
                        np.abs(self.actual[indices][mask] - prediction[mask]).mean()
                    )
                    if mask.any()
                    else None,
                    "seconds": seconds,
                }
            )
        return predictions, {
            "mae": float(np.abs(self.actual[paired] - predictions[paired]).mean()),
            "folds": results,
            "seconds": sum(fold["seconds"] for fold in results),
            "parameters": spec,
        }

    def score(self, predictions, rules):
        return evaluate_comparison(
            self.timestamps,
            self.actual,
            self.baselines | predictions,
            rules,
            [lag.weekly_fallback for lag in self.previous],
        )


def tree_parameters(family, trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.15, log=True)
    }
    if family == "lightgbm":
        return params | {
            "num_leaves": trial.suggest_int("num_leaves", 8, 48),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 120),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 20.0, log=True),
        }
    return params | {
        "depth": trial.suggest_int("depth", 4, 7),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 20.0, log=True),
    }


def tune(name, evaluation, training, untuned, trials, timeout):
    base = DEFAULTS[name]
    runs = []

    def run(parameters):
        spec = base | {"params": base["params"] | parameters}
        _, result = evaluation.fit(training, spec)
        runs.append(result)
        return result["mae"]

    started = perf_counter()
    if name == "elastic_net":
        for alpha, ratio in product((0.01, 0.1, 1.0), (0.1, 0.5, 0.9)):
            run({"alpha": alpha, "l1_ratio": ratio})
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
        )
        study.optimize(
            lambda trial: run(tree_parameters(name, trial)),
            n_trials=trials,
            timeout=timeout,
            n_jobs=1,
        )
    best = min(runs, key=lambda result: result["mae"], default=None)
    # Retain the default if none of the trials improves development MAE.
    selected = (
        min((untuned, best), key=lambda result: result["mae"]) if best else untuned
    )
    return selected["parameters"], {
        "untuned": untuned,
        "best_trial": best,
        "selected": selected,
        "trials": runs,
        "seconds": perf_counter() - started,
    }


def prediction_rows(evaluation, predictions, data_version):
    for origin, _, indices in evaluation.blocks:
        for i in range(indices.start, indices.stop):
            stamp = evaluation.timestamps[i]
            day = local_day(stamp)
            lag = evaluation.previous[i]
            yield {
                "price_area": PRICE_AREA,
                "delivery_date": day,
                "delivery_start_utc": stamp,
                "issue_time_utc": forecast_issue_utc(day),
                "model_training_origin": origin,
                "data_version": data_version,
                "previous_source_utc": lag.source_utc,
                "weekly_fallback": lag.weekly_fallback,
                **{
                    name: float(values[i]) if np.isfinite(values[i]) else None
                    for name, values in predictions.items()
                },
            }


def save_predictions(path, rows, methods):
    schema = {
        "price_area": "VARCHAR",
        "delivery_date": "DATE",
        "delivery_start_utc": "TIMESTAMPTZ",
        "issue_time_utc": "TIMESTAMPTZ",
        "model_training_origin": "DATE",
        "data_version": "VARCHAR",
        "previous_source_utc": "TIMESTAMPTZ",
        "weekly_fallback": "BOOLEAN",
        **{name: "DOUBLE" for name in methods},
    }
    with connect() as connection:
        connection.execute(
            "CREATE TABLE predictions AS SELECT unnest(from_json_strict(?, ?), recursive := true)",
            [json_bytes(list(rows)).decode(), json.dumps([schema])],
        )
        write_parquet(
            connection, "SELECT * FROM predictions ORDER BY delivery_start_utc", path
        )


def run_comparison(prices, root, trials, timeout):
    history = PriceHistory.load(prices)
    if local_day(max(history.prices)) < DIAGNOSTIC_END - date.resolution:
        raise ValueError("The snapshot must cover the full diagnostic period")
    training = ComparisonTraining(history, DEVELOPMENT_START)
    development = Evaluation.build(
        history, fold_ranges(DEVELOPMENT_START, DIAGNOSTIC_START)
    )
    diagnostic = Evaluation.build(history, [(DIAGNOSTIC_START, DIAGNOSTIC_END)])
    rules = SCORING | {
        "extreme_threshold_eur_per_mwh": float(
            np.quantile(
                np.abs(development.actual[np.isfinite(development.actual)]),
                SCORING["extreme_quantile"],
                method=SCORING["extreme_quantile_method"],
            )
        )
    }

    legacy, legacy_diagnostic = diagnostic.fit(training, DEFAULTS["legacy_lightgbm"])
    reproduced = diagnostic.score({"legacy_lightgbm": legacy}, rules)
    checks = {
        name: reproduced["slices"]["all"]["methods"][name]["mae"]
        for name in DIAGNOSTIC_MAE
    }
    if len(diagnostic.timestamps) != 5380 or any(
        abs(checks[name] - value) >= 0.005 for name, value in DIAGNOSTIC_MAE.items()
    ):
        raise ValueError(
            f"Diagnostic reproduction differs: {len(diagnostic.timestamps)} intervals, {checks}"
        )
    print(f"Reproduced diagnostic MAE: {checks}", file=sys.stderr, flush=True)

    development_predictions = {}
    models, untuned = {}, {}
    for name, spec in DEFAULTS.items():
        development_predictions[name], untuned[name] = development.fit(training, spec)
        print(
            f"Development {name}: {untuned[name]['mae']:.4f} EUR/MWh",
            file=sys.stderr,
            flush=True,
        )

    tuning = {}
    for name in ("elastic_net", "lightgbm", "catboost"):
        models[name], tuning[name] = tune(
            name, development, training, untuned[name], trials, timeout
        )
        development_predictions[f"{name}_tuned"], _ = development.fit(
            training, models[name]
        )
        result = tuning[name]["selected"]["mae"]
        print(
            f"Selected {name}: {result:.4f} EUR/MWh ({len(tuning[name]['trials'])} trials)",
            file=sys.stderr,
            flush=True,
        )

    diagnostic_predictions = {"legacy_lightgbm": legacy}
    diagnostic_runs = {"legacy_lightgbm": legacy_diagnostic}
    for name in models:
        for label, spec in ((name, DEFAULTS[name]), (f"{name}_tuned", models[name])):
            diagnostic_predictions[label], diagnostic_runs[label] = diagnostic.fit(
                training, spec
            )
    report = {
        "data_version": history.data_version,
        "features": COMMON_FEATURES,
        "baseline_versions": [BASELINE_VERSION, PREVIOUS_DAY_VERSION],
        "versions": {
            name: version(name)
            for name in ("lightgbm", "catboost", "scikit-learn", "optuna", "numpy")
        },
        "optuna": {
            "seed": SEED,
            "trials_per_tree_max": trials,
            "timeout_seconds_per_tree": timeout,
        },
        "fold_days": FOLD_DAYS,
        "scoring": rules,
        "diagnostic_reproduction": {
            "expected_rounded_mae": DIAGNOSTIC_MAE,
            "observed_mae": checks,
        },
        "development": development.score(development_predictions, rules),
        "diagnostic": diagnostic.score(diagnostic_predictions, rules),
        "development_untuned_fits": untuned,
        "diagnostic_fits": diagnostic_runs,
        "tuning": tuning,
        "frozen_models": models,
        "development_choice": min(
            models, key=lambda name: tuning[name]["selected"]["mae"]
        ),
    }
    content = json_bytes(report)
    output = root / "backtests" / digest(content)
    for name, evaluation, predictions in (
        ("development", development, development_predictions),
        ("diagnostic", diagnostic, diagnostic_predictions),
    ):
        combined = evaluation.baselines | predictions
        save_predictions(
            output / f"{name}.parquet",
            prediction_rows(evaluation, combined, history.data_version),
            combined,
        )
    publish_bytes(output / "experiment.json", content)
    return output / "experiment.json"


def main():
    parser = argparse.ArgumentParser(
        description="Compare DK2 models using fixed walk-forward folds."
    )
    parser.add_argument("prices", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--timeout", type=float, default=180, help="Seconds per tree search"
    )
    args = parser.parse_args()
    if not 1 <= args.trials <= 30 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Use 1–30 trials and a positive finite timeout")
    print(run_comparison(args.prices, args.root, args.trials, args.timeout))


if __name__ == "__main__":
    main()
