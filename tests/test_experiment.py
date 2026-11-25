import json
from datetime import date, timedelta

import numpy as np
import pytest

from dk2_forecast.delivery_day import delivery_intervals_utc, forecast_issue_utc
from dk2_forecast.experiment import json_bytes, save_predictions
from dk2_forecast.experiment_metrics import SCORING, evaluate_comparison
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.prospective import implementation_version, score
from dk2_forecast.storage import digest

RULES = SCORING | {"extreme_threshold_eur_per_mwh": 300}


def test_comparison_common_denominator_and_delivery_day_wins():
    day = date(2026, 3, 29)
    stamps = delivery_intervals_utc(day)
    actual = np.asarray([0] * 88 + [200, 300, -300, -1], dtype=float)
    baseline = actual + 50
    model = actual + 25
    methods = {"previous_day": baseline, "model": model}
    report = evaluate_comparison(stamps, actual, methods, RULES, [False] * 91 + [True])
    assert report["slices"]["spike"]["paired_intervals"] == 2
    assert report["slices"]["extreme"]["paired_intervals"] == 2
    assert report["slices"]["negative"]["paired_intervals"] == 2
    assert report["previous_day_weekly_fallbacks"] == 1
    assert report["slices"]["all"]["methods"]["model"] == {
        "mae": 25,
        "within_25": 1,
        "within_50": 1,
    }
    assert report["delivery_day_results"]["methods"]["model"]["wins"] == 1

    model[-1] = np.nan
    report = evaluate_comparison(stamps, actual, methods, RULES, [])
    assert report["slices"]["all"]["paired_intervals"] == 91
    assert report["coverage"]["previous_day"]["forecast_intervals"] == 92
    assert report["coverage"]["model"]["forecast_intervals"] == 91
    assert report["delivery_day_results"]["complete_paired_days"] == 0


def test_scores_only_prospectively_issued_intervals(tmp_path, monkeypatch):
    start = date(2026, 11, 18)
    end = start + timedelta(days=2)
    issue = forecast_issue_utc(start)
    rules = {
        "start": start,
        "end_exclusive": end,
        "frozen_at_utc": (issue - timedelta(days=1)).isoformat(),
        "implementation_sha256": implementation_version(),
        "versions": {},
        "scoring": RULES,
        "models": {"elastic_net": {}},
    }
    content = json_bytes(rules)
    plan = tmp_path / digest(content) / "freeze.json"
    plan.parent.mkdir()
    plan.write_bytes(content)
    stamps = delivery_intervals_utc(start)
    historical_prices = {
        stamp: 25.0
        for stamp in stamps + delivery_intervals_utc(start + date.resolution)
    }
    monkeypatch.setattr(
        PriceHistory, "load", lambda _: PriceHistory(historical_prices, "fixture")
    )
    with pytest.raises(ValueError, match="No prospectively issued"):
        score(plan, tmp_path / "unused")

    output = plan.parent / "forecasts" / str(start) / "forecast.parquet"
    methods = ("weekly", "previous_day", "elastic_net")
    rows = [
        {
            "price_area": "DK2",
            "delivery_date": start,
            "delivery_start_utc": stamp,
            "issue_time_utc": issue,
            "model_training_origin": start,
            "data_version": "fixture",
            "previous_source_utc": None,
            "weekly_fallback": False,
            "weekly": 75.0,
            "previous_day": 50.0,
            "elastic_net": 25.0,
        }
        for stamp in stamps
    ]
    save_predictions(output, rows, methods)
    receipt = {
        "freeze_sha256": plan.parent.name,
        "forecast_sha256": digest(output.read_bytes()),
        "observed_at_utc": (issue - timedelta(minutes=10)).isoformat(),
        "prepared_at_utc": (issue - timedelta(minutes=9)).isoformat(),
        "published_at_utc": issue.isoformat(),
    }
    output.with_suffix(".json").write_bytes(json_bytes(receipt))
    report = json.loads(score(plan, tmp_path / "unused").read_bytes())
    assert report["slices"]["all"]["paired_intervals"] == 96
    assert report["coverage"]["elastic_net"]["share"] == 0.5
    assert report["full_period_actuals_available"]
    receipt["published_at_utc"] = (issue + timedelta(minutes=1)).isoformat()
    output.with_suffix(".json").write_bytes(json_bytes(receipt))
    with pytest.raises(ValueError, match="Invalid prospective"):
        score(plan, tmp_path / "unused")
