from datetime import date
from time import perf_counter

import lightgbm as lgb
import numpy as np
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dk2_forecast.delivery_day import delivery_intervals_utc
from dk2_forecast.forecast import (
    BOOSTING_ROUNDS,
    FEATURE_NAMES,
    MODEL_PARAMS,
    PriceHistory,
    TrainingData,
    features,
)
from dk2_forecast.price_lags import lag_price, local_day, previous_day_price

SEED = 42
COMMON_FEATURES = (*FEATURE_NAMES, "previous_danish_day_price")
BASELINES = ("weekly", "previous_day")
DEFAULTS = {
    "legacy_lightgbm": {
        "family": "lightgbm",
        "previous_day": False,
        "params": MODEL_PARAMS | {"num_boost_round": BOOSTING_ROUNDS},
    },
    "lightgbm": {
        "family": "lightgbm",
        "previous_day": True,
        "params": MODEL_PARAMS | {"num_boost_round": BOOSTING_ROUNDS},
    },
    "elastic_net": {
        "family": "elastic_net",
        "previous_day": True,
        "params": {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 10000},
    },
    "catboost": {
        "family": "catboost",
        "previous_day": True,
        "params": {
            "iterations": 160,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "loss_function": "MAE",
            "thread_count": 1,
            "random_seed": SEED,
            "task_type": "CPU",
            "verbose": False,
            "allow_writing_files": False,
        },
    },
}


def input_rows(history: PriceHistory, intervals):
    return [
        previous_day_price(
            history.prices, timestamp, local_day(timestamp) - date.resolution
        )
        for timestamp in intervals
    ]


class ComparisonTraining:
    def __init__(self, history: PriceHistory, first_origin: date):
        self.data = TrainingData(history, first_origin)
        previous = input_rows(history, self.data.intervals)
        self.matrix = np.column_stack(
            (self.data.matrix, np.asarray([lag.price for lag in previous], dtype=float))
        )

    def arrays(self, origin: date):
        count = self.data.eligible_count(origin)
        return self.matrix[:count], self.data.labels[:count]


def daily_inputs(history: PriceHistory, day: date):
    timestamps = delivery_intervals_utc(day)
    previous = input_rows(history, timestamps)
    matrix = np.column_stack(
        (features(history.prices, timestamps, day), [lag.price for lag in previous])
    ).astype(float)
    weekly = [
        lag_price(history.prices, timestamp, 7, day - date.resolution)
        for timestamp in timestamps
    ]
    return timestamps, matrix, weekly, previous


def fit_predict(spec: dict, train_x, train_y, evaluation_x):
    family = spec["family"]
    params = spec["params"].copy()
    names = COMMON_FEATURES
    if not spec["previous_day"]:
        names = FEATURE_NAMES
        train_x, evaluation_x = train_x[:, :-1], evaluation_x[:, :-1]
    start = perf_counter()
    if family == "lightgbm":
        rounds = params.pop("num_boost_round")
        dataset = lgb.Dataset(train_x, label=train_y, feature_name=list(names))
        model = lgb.train(params, dataset, num_boost_round=rounds)
        predictions = model.predict(evaluation_x, num_threads=1)
    elif family == "catboost":
        model = CatBoostRegressor(**params)
        model.fit(train_x, train_y)
        predictions = model.predict(evaluation_x, thread_count=1)
    else:
        if family != "elastic_net":
            raise ValueError(f"Unknown model family {family}")
        with threadpool_limits(limits=1):
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                ElasticNet(**params),
            )
            predictions = model.fit(train_x, train_y).predict(evaluation_x)
    if not np.isfinite(predictions).all():
        raise ValueError(f"{family} produced a non-finite forecast")
    return predictions, perf_counter() - start
