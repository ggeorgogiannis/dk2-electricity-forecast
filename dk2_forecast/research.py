"""Versioned CPU research using whole Danish delivery days and prior-only fits."""

import argparse
import json
import pickle
import platform
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dk2_forecast.backtest import fold_ranges
from dk2_forecast.delivery_day import delivery_intervals_utc
from dk2_forecast.experiment import json_bytes
from dk2_forecast.experiment_metrics import SCORING
from dk2_forecast.explanatory import align_weather
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.provenance import save_code_snapshot
from dk2_forecast.price_lags import local_day
from dk2_forecast.price_lags import previous_day_price
from dk2_forecast.research_features import PRICE_FEATURE_NAMES, all_price_features
from dk2_forecast.research_metrics import interval_metrics, point_metrics
from dk2_forecast.storage import connect, digest, publish_bytes, write_parquet

START = date(2026, 4, 2)
END = date(2026, 9, 21)
SEED = 42


def fit_model(family, x, y, params):
    started = perf_counter()
    if family == "ridge":
        with threadpool_limits(limits=1):
            model = make_pipeline(
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
                StandardScaler(),
                Ridge(**params),
            ).fit(x, y)
    elif family == "catboost":
        model = CatBoostRegressor(
            iterations=160,
            loss_function="MAE",
            thread_count=1,
            random_seed=SEED,
            task_type="CPU",
            allow_writing_files=False,
            verbose=False,
            **params,
        ).fit(x, y)
    else:
        model = lgb.train(
            {
                "objective": "regression_l1",
                "verbosity": -1,
                "num_threads": 1,
                "force_col_wise": True,
                "deterministic": True,
                "seed": SEED,
                "learning_rate": 0.05,
                "num_leaves": 15,
                **params,
            },
            lgb.Dataset(x, label=y),
            num_boost_round=160,
        )
    return model, perf_counter() - started


def predict_model(model, x):
    if isinstance(model, lgb.Booster):
        return model.predict(x, num_threads=1)
    with threadpool_limits(limits=1):
        return model.predict(x)


class ResearchData:
    def __init__(self, history, root):
        self.history = history
        self.timestamps, self.matrix, self.seasonal = all_price_features(history)
        self.days = np.asarray([local_day(stamp) for stamp in self.timestamps])
        self.actual = np.asarray(
            [history.prices.get(stamp, np.nan) for stamp in self.timestamps]
        )
        self.weather, self.weather_versions = align_weather(
            root, self.timestamps, allow_assumed=True
        )
        self.columns = {name: i for i, name in enumerate(PRICE_FEATURE_NAMES)}

    def evaluate(
        self,
        family,
        params,
        *,
        weeks=None,
        frequency=28,
        correction=False,
        weather=False,
        start=START,
        end=END,
    ):
        selected = (self.days >= start) & (self.days < end)
        predictions = np.full(selected.sum(), np.nan)
        all_indices = np.flatnonzero(selected)
        matrix = (
            np.column_stack((self.matrix, self.weather)) if weather else self.matrix
        )
        results = []
        for origin, stop in fold_ranges(start, end, frequency):
            train = (self.days < origin) & np.isfinite(self.actual)
            if weeks is not None:
                train &= self.days >= origin - timedelta(weeks=weeks)
            target_indices = np.flatnonzero((self.days >= origin) & (self.days < stop))
            labels = self.actual[train].copy()
            if correction:
                usable = np.isfinite(matrix[train, 0])
                train_indices = np.flatnonzero(train)[usable]
                labels = self.actual[train_indices] - matrix[train_indices, 0]
            else:
                train_indices = np.flatnonzero(train)
            model, fit_time = fit_model(family, matrix[train_indices], labels, params)
            started = perf_counter()
            forecast = predict_model(model, matrix[target_indices])
            if correction:
                forecast += matrix[target_indices, 0]
            prediction_time = perf_counter() - started
            predictions[np.searchsorted(all_indices, target_indices)] = forecast
            print(
                f"{family} window={weeks} {origin} fit={fit_time:.2f}s",
                file=sys.stderr,
                flush=True,
            )
            results.append(
                {
                    "origin": str(origin),
                    "end_exclusive": str(stop),
                    "training_rows": len(labels),
                    "training_last_delivery": str(max(self.days[train_indices])),
                    "fit_seconds": fit_time,
                    "prediction_seconds": prediction_time,
                }
            )
        actual = self.actual[selected]
        eligible = (
            np.isfinite(actual)
            & np.isfinite(self.matrix[selected, 0])
            & np.isfinite(predictions)
        )
        return predictions, {
            "family": family,
            "parameters": params,
            "weeks": weeks,
            "frequency_days": frequency,
            "correction": correction,
            "weather": weather,
            "folds": results,
            "mae": float(np.abs(actual[eligible] - predictions[eligible]).mean()),
            "fit_seconds": sum(row["fit_seconds"] for row in results),
            "prediction_seconds": sum(row["prediction_seconds"] for row in results),
        }


def tune_tree(
    data,
    family,
    root,
    identity,
    untuned,
    weeks,
    trials,
    timeout,
    *,
    startup_trials=5,
    guard=None,
):
    storage = f"sqlite:///{(root / 'studies.sqlite3').resolve().as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=startup_trials)
    study = optuna.create_study(
        study_name=f"{identity}-{family}",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
    )
    # Optuna persists the trials. Pickle preserves the TPE RNG state on sequential resumes.
    sampler_path = root / f"{identity}-{family}-sampler.pkl"
    if sampler_path.exists():
        study.sampler = pickle.loads(sampler_path.read_bytes())

    def objective(trial):
        finished = sum(
            t.state
            in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
            for t in study.trials
        )
        trial.set_user_attr(
            "sampling_phase",
            "random_startup" if finished < startup_trials else "adaptive_tpe",
        )
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.15, log=True)
        }
        if family == "catboost":
            params |= {
                "depth": trial.suggest_int("depth", 4, 7),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 20, log=True),
            }
        else:
            params |= {
                "num_leaves": trial.suggest_int("num_leaves", 8, 40),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 120),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 20, log=True),
            }
        _, result = data.evaluate(family, params, weeks=weeks)
        trial.set_user_attr("fit", result)
        return result["mae"]

    started = perf_counter()
    for _ in range(max(0, trials - len(study.trials))):
        if perf_counter() - started >= timeout:
            break
        if guard:
            guard()
        study.optimize(
            objective, n_trials=1, n_jobs=1, catch=(ValueError, lgb.basic.LightGBMError)
        )
        # Local, generated state only. Never load sampler files from untrusted sources.
        temporary = sampler_path.with_suffix(".tmp")
        temporary.write_bytes(pickle.dumps(study.sampler))
        temporary.replace(sampler_path)
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    best = min(
        (trial.user_attrs["fit"] for trial in complete),
        key=lambda row: row["mae"],
        default=untuned,
    )
    return min((untuned, best), key=lambda row: row["mae"]), {
        "sampler": "TPE",
        "startup_trials_planned": startup_trials,
        "random_startup_completed": sum(
            t.user_attrs.get("sampling_phase") == "random_startup" for t in complete
        ),
        "adaptive_tpe_completed": sum(
            t.user_attrs.get("sampling_phase") == "adaptive_tpe" for t in complete
        ),
        "failed_trials": sum(
            t.state == optuna.trial.TrialState.FAIL for t in study.trials
        ),
        "pruned_trials": sum(
            t.state == optuna.trial.TrialState.PRUNED for t in study.trials
        ),
        "completed_trials": len(complete),
        "budget_sufficient_for_adaptive": any(
            t.user_attrs.get("sampling_phase") == "adaptive_tpe" for t in complete
        ),
        "selected_parameters": min((untuned, best), key=lambda row: row["mae"])[
            "parameters"
        ],
        "seed": SEED,
        "sequential": True,
        "max_trials": trials,
        "timeout_seconds": timeout,
        "elapsed_seconds": perf_counter() - started,
        "trials": [
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "parameters": trial.params,
                "sampling_phase": trial.user_attrs.get(
                    "sampling_phase", "legacy_unrecorded"
                ),
                "fit": trial.user_attrs.get("fit"),
                "seconds": trial.duration.total_seconds() if trial.duration else None,
            }
            for trial in study.trials
        ],
    }


def research_run(prices, root, trials=30, timeout=180):
    code_snapshot = save_code_snapshot(root)
    history = PriceHistory.load(prices)
    required_start = START - timedelta(weeks=26)
    expected = {
        stamp
        for offset in range((END - required_start).days)
        for stamp in delivery_intervals_utc(required_start + timedelta(days=offset))
    }
    if not expected.issubset(history.prices):
        raise ValueError(
            "Research requires complete prices from 26 weeks before 2 April through 20 September 2026"
        )
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = ResearchData(history, root.parents[0] if root.name == "research" else root)
    selected = (data.days >= START) & (data.days < END)
    stamps = [
        stamp
        for stamp, include in zip(data.timestamps, selected, strict=True)
        if include
    ]
    actual = data.actual[selected]
    baseline = data.matrix[selected, 0]
    predictions = {
        "previous_day": baseline,
        "weekly_utc": data.matrix[selected, 1],
        "weekday_persistence": data.seasonal[selected],
    }
    fits = {}
    # Same outer dates for every window. 26 weeks are available by 2 April.
    for weeks in (None, 8, 12, 26):
        name = f"window_{weeks or 'expanding'}"
        _, fits[name] = data.evaluate("lightgbm", {}, weeks=weeks)
    window = min(fits.values(), key=lambda value: value["mae"])["weeks"]
    for family, params in (
        ("lightgbm", {}),
        ("catboost", {"depth": 4, "learning_rate": 0.07, "l2_leaf_reg": 3}),
        ("ridge", {"alpha": 10.0}),
    ):
        predictions[family], fits[family] = data.evaluate(family, params, weeks=window)
    linear_grid = []
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        _, result = data.evaluate("ridge", {"alpha": alpha}, weeks=window)
        linear_grid.append(result)
    best_linear = min(linear_grid, key=lambda row: row["mae"])
    predictions["ridge_tuned"], fits["ridge_tuned"] = data.evaluate(
        "ridge", best_linear["parameters"], weeks=window
    )
    study_identity = digest(
        json_bytes(
            {
                "data": history.data_version,
                "start": START,
                "end": END,
                "features": PRICE_FEATURE_NAMES,
                "window": window,
                "version": 2,
                "objective_code_sha256": digest(Path(__file__).read_bytes()),
            }
        )
    )
    searches = {}
    for family in ("lightgbm", "catboost"):
        best, searches[family] = tune_tree(
            data, family, root, study_identity, fits[family], window, trials, timeout
        )
        predictions[f"{family}_tuned"], fits[f"{family}_tuned"] = data.evaluate(
            family, best["parameters"], weeks=window
        )
    predictions["lightgbm_squared"], fits["lightgbm_squared"] = data.evaluate(
        "lightgbm", {"objective": "regression"}, weeks=window
    )
    predictions["correction"], fits["correction"] = data.evaluate(
        "lightgbm", fits["lightgbm_tuned"]["parameters"], weeks=window, correction=True
    )
    best_name = min(
        (
            "ridge_tuned",
            "lightgbm_tuned",
            "catboost_tuned",
            "lightgbm_squared",
            "correction",
        ),
        key=lambda name: fits[name]["mae"],
    )
    best = fits[best_name]
    predictions["weekly_refit"], fits["weekly_refit"] = data.evaluate(
        best["family"],
        best["parameters"],
        weeks=window,
        frequency=7,
        correction=best["correction"],
    )
    frequency = 7 if fits["weekly_refit"]["mae"] < best["mae"] else 28
    predictions["fixed_ensemble"] = (
        predictions["ridge_tuned"] + predictions["catboost_tuned"]
    ) / 2
    rolling_weights = []
    predictions["oof_ensemble"] = np.empty(len(stamps))
    for origin, stop in fold_ranges(START, END):
        prior = np.asarray([local_day(stamp) < origin for stamp in stamps])
        target = np.asarray([origin <= local_day(stamp) < stop for stamp in stamps])
        weight = (
            min(
                (0, 0.25, 0.5, 0.75, 1),
                key=lambda w: np.abs(
                    actual[prior]
                    - (
                        w * predictions["ridge_tuned"][prior]
                        + (1 - w) * predictions["catboost_tuned"][prior]
                    )
                ).mean(),
            )
            if prior.any()
            else 0.5
        )
        predictions["oof_ensemble"][target] = (
            weight * predictions["ridge_tuned"][target]
            + (1 - weight) * predictions["catboost_tuned"][target]
        )
        rolling_weights.append({"origin": str(origin), "ridge_weight": weight})
    quantiles = []
    for alpha in (0.1, 0.5, 0.9):
        values, fits[f"quantile_{alpha}"] = data.evaluate(
            "lightgbm",
            fits["lightgbm_tuned"]["parameters"]
            | {"objective": "quantile", "alpha": alpha},
            weeks=window,
        )
        quantiles.append(values)
    quantiles = np.column_stack(quantiles)
    calibrated = np.sort(quantiles.copy(), axis=1)
    calibration = []
    for origin, stop in fold_ranges(START, END):
        prior = np.asarray([local_day(stamp) < origin for stamp in stamps])
        target = np.asarray([origin <= local_day(stamp) < stop for stamp in stamps])
        if prior.sum() >= 28 * 92:
            residual = np.maximum(
                quantiles[prior, 0] - actual[prior], actual[prior] - quantiles[prior, 2]
            )
            adjustment = max(
                0,
                float(
                    np.quantile(
                        residual,
                        min(1, np.ceil((len(residual) + 1) * 0.8) / len(residual)),
                        method="higher",
                    )
                ),
            )
            calibrated[target, 0] -= adjustment
            calibrated[target, 2] += adjustment
            calibration.append(
                {
                    "origin": str(origin),
                    "prior_intervals": int(prior.sum()),
                    "adjustment": adjustment,
                }
            )
    rules = SCORING | {
        "extreme_threshold_eur_per_mwh": float(
            np.quantile(np.abs(actual), 0.99, method="higher")
        )
    }
    weather_report = {"status": "no eligible archived weather captures"}
    if data.weather_versions:
        weather_prediction, weather_fit = data.evaluate(
            "lightgbm", fits["lightgbm_tuned"]["parameters"], weeks=window, weather=True
        )
        eligible = np.isfinite(data.weather[selected]).all(axis=1)
        weather_report = {
            "status": "exploratory_historical_availability_assumption",
            "fit": weather_fit,
            "versions": data.weather_versions,
            "paired_intervals": int(eligible.sum()),
            "price_only_mae": float(
                np.abs(
                    actual[eligible] - predictions["lightgbm_tuned"][eligible]
                ).mean()
            ),
            "with_weather_mae": float(
                np.abs(actual[eligible] - weather_prediction[eligible]).mean()
            ),
            "coverage": float(eligible.mean()),
        }
    code_files = sorted(Path(__file__).parent.glob("*.py"))
    code_hash = digest(b"\0".join(path.read_bytes() for path in code_files))
    report = {
        "created_at_utc": datetime.now(UTC),
        "status": "development_research",
        "period": {"start": START, "end_exclusive": END},
        "information_policy": "Price labels and lags through issue-day delivery only; research live data cutoff 09:50 Copenhagen",
        "historical_assumption": "Archived price values treated as available on issue day; revision vintage unavailable",
        "features": PRICE_FEATURE_NAMES,
        "data_version": history.data_version,
        "code_sha256": code_hash,
        "code_snapshot_sha256": code_snapshot,
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "uncommitted",
        "versions": {
            name: version(name)
            for name in ("numpy", "lightgbm", "catboost", "scikit-learn", "optuna")
        },
        "python": platform.python_version(),
        "fits": fits,
        "linear_grid": linear_grid,
        "searches": searches,
        "selection": {
            "model": best_name,
            "training_weeks": window,
            "retraining_days": frequency,
            "parameters": best["parameters"],
            "family": best["family"],
            "correction": best["correction"],
        },
        "scoring": rules,
        "previous_day_weekly_fallbacks": sum(
            previous_day_price(
                history.prices, stamp, local_day(stamp) - date.resolution
            ).weekly_fallback
            for stamp in stamps
        ),
        "point": point_metrics(stamps, actual, predictions, rules),
        "quantiles": interval_metrics(stamps, actual, quantiles),
        "calibrated_quantiles": interval_metrics(stamps, actual, calibrated),
        "calibration": calibration,
        "ensemble_weights": rolling_weights,
        "weather_ablation": weather_report,
        "elapsed_seconds": perf_counter() - started,
    }
    content = json_bytes(report)
    folder = root / digest(content)
    rows = [
        {
            "price_area": "DK2",
            "delivery_start_utc": stamp.isoformat(),
            "actual": float(actual[i]),
            **{name: float(series[i]) for name, series in predictions.items()},
            **{
                f"q{level}": float(quantiles[i, index])
                for index, level in enumerate((10, 50, 90))
            },
            **{
                f"calibrated_q{level}": float(calibrated[i, index])
                for index, level in enumerate((10, 50, 90))
            },
        }
        for i, stamp in enumerate(stamps)
    ]
    schema = [
        {
            "price_area": "VARCHAR",
            "delivery_start_utc": "TIMESTAMPTZ",
            **{
                name: "DOUBLE"
                for name in rows[0]
                if name not in ("price_area", "delivery_start_utc")
            },
        }
    ]
    with connect() as db:
        db.execute(
            "CREATE TABLE predictions AS SELECT unnest(from_json_strict(?, ?), recursive := true)",
            [json.dumps(rows), json.dumps(schema)],
        )
        write_parquet(
            db,
            "SELECT * FROM predictions ORDER BY delivery_start_utc",
            folder / "predictions.parquet",
        )
    publish_bytes(folder / "report.json", content)
    return folder / "report.json"


def main():
    parser = argparse.ArgumentParser(
        description="Run the separately versioned CPU research comparison."
    )
    parser.add_argument("prices", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/research"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.trials <= 30 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Use 1â€“30 trials and a finite positive timeout")
    print(research_run(args.prices, args.root, args.trials, args.timeout))


if __name__ == "__main__":
    main()
