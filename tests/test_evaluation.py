"""Offline operational integration: real research Parquet/receipt/score schemas."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from dk2_forecast.api import create_app
from dk2_forecast.delivery_day import delivery_intervals_utc, forecast_issue_utc
from dk2_forecast.evaluation import (
    aggregate,
    earliest_origin,
    freeze,
    load_evaluation,
    validate_publication,
    restart,
)
from dk2_forecast.experiment import json_bytes
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.ingestion import ingest_prices
from dk2_forecast.research_features import PRICE_FEATURE_NAMES
from dk2_forecast.research_forecast import forecast, publish
from dk2_forecast.research_ops import schedule_spec
from dk2_forecast.storage import digest


class Model:
    def predict(self, matrix):
        return matrix[:, 0] + 1

    def model_to_string(self):
        return "offline integration model"


@pytest.fixture
def case(tmp_path, monkeypatch):
    day = date(2026, 10, 25)

    def response(request):
        first = date.fromisoformat(request.url.params["start"])
        stop = date.fromisoformat(request.url.params["end"])
        return httpx.Response(
            200,
            json={
                "records": [
                    {
                        "PriceArea": "DK2",
                        "TimeUTC": stamp.isoformat(),
                        "DayAheadPriceEUR": float(quarter - 20),
                    }
                    for i in range((stop - first).days)
                    for quarter, stamp in enumerate(
                        delivery_intervals_utc(first + timedelta(days=i))
                    )
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        prices = ingest_prices(
            tmp_path, date(2026, 10, 1), day + timedelta(days=3), client
        )
    report = {
        "selection": {
            "family": "catboost",
            "parameters": {"depth": 7},
            "training_weeks": None,
            "retraining_days": 7,
            "correction": False,
        },
        "features": PRICE_FEATURE_NAMES,
        "scoring": {
            "absolute_error_bands": [25, 50],
            "spike_threshold_eur_per_mwh": 200,
            "extreme_threshold_eur_per_mwh": 276.019989,
            "delivery_day_reference": "previous_day",
        },
        "fits": {"lightgbm_tuned": {"parameters": {}}},
        "calibration": [{"adjustment": 2}],
        "git_revision": "offline-test",
        "historical_assumption": "fixture",
        "period": {"end_exclusive": "2026-09-21"},
    }
    report_path = tmp_path / "source-report.json"
    report_path.write_bytes(json_bytes(report))
    plan = freeze(
        report_path, tmp_path, day, days=3, now=datetime(2026, 10, 23, 7, tzinfo=UTC)
    )
    calls = []

    def fit(*args):
        calls.append(args[0])
        return Model(), 0.01

    monkeypatch.setattr("dk2_forecast.research_forecast.fit_model", fit)
    clock = {"now": datetime(2026, 10, 24, 7, 40, tzinfo=UTC)}

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"].astimezone(tz or UTC)

    monkeypatch.setattr("dk2_forecast.research_forecast.datetime", Clock)

    def capture(root, delivery):
        return (
            prices,
            PriceHistory.load(prices),
            {
                "data_version": prices.parent.name,
                "prices_sha256": digest(prices.read_bytes()),
                "observed_at_utc": clock["now"].isoformat(),
            },
        )

    monkeypatch.setattr("dk2_forecast.research_forecast.capture_inputs", capture)
    return tmp_path, report_path, plan, day, prices, clock, calls


def prepare(case):
    root, report, plan, day, _, _, _ = case
    return forecast(report, None, day, root, replay=False, evaluation=plan)


def issue(case):
    root, report, plan, day, _, clock, _ = case
    prepare(case)
    clock["now"] = forecast_issue_utc(day)
    return publish(root, report, day, evaluation=plan)


def test_candidate_can_have_multiple_evaluations_and_plan_tampering_fails(case):
    root, report, first, day, _, _, _ = case
    second = freeze(
        report, root, day + timedelta(days=5), now=datetime(2026, 10, 23, tzinfo=UTC)
    )
    assert first != second
    assert (
        load_evaluation(first)["candidate_id"]
        == load_evaluation(second)["candidate_id"]
    )
    assert (
        freeze(report, root, day, days=3, now=datetime(2026, 10, 23, tzinfo=UTC))
        == first
    )
    raw = json.loads(second.read_bytes())
    raw["start"] = str(day)
    second.write_bytes(json_bytes(raw))
    with pytest.raises(ValueError, match="identity"):
        load_evaluation(second)


def test_restart_reuses_retained_candidate_after_checkout_changes(case, monkeypatch):
    root, _, first, day, _, _, _ = case
    monkeypatch.setattr(
        "dk2_forecast.research_forecast.current_code_hash",
        lambda: "new-checkout-version",
    )
    second = restart(
        first, root, day + timedelta(days=8), now=datetime(2026, 10, 23, tzinfo=UTC)
    )
    assert second != first
    assert (
        load_evaluation(second)["candidate_id"]
        == load_evaluation(first)["candidate_id"]
    )
    assert (
        second.with_name("launch.py").read_bytes()
        == first.with_name("launch.py").read_bytes()
    )


def test_capture_finishing_late_and_late_publication_are_rejected(case, monkeypatch):
    root, report, plan, day, prices, clock, _ = case

    def late_capture(root, delivery):
        return (
            prices,
            PriceHistory.load(prices),
            {
                "data_version": prices.parent.name,
                "prices_sha256": digest(prices.read_bytes()),
                "observed_at_utc": "2026-10-24T07:51:00+00:00",
            },
        )

    with monkeypatch.context() as patch:
        patch.setattr("dk2_forecast.research_forecast.capture_inputs", late_capture)
        with pytest.raises(ValueError, match="completed after"):
            prepare(case)
    prepare(case)
    clock["now"] = forecast_issue_utc(day) + timedelta(seconds=61)
    with pytest.raises(ValueError, match="10:00"):
        publish(root, report, day, evaluation=plan)
    assert not (plan.parent / "forecasts" / str(day)).exists()


def test_new_evaluation_interrupted_publication_cleans_and_retries(case, monkeypatch):
    root, report, plan, day, _, clock, _ = case
    prepare(case)
    clock["now"] = forecast_issue_utc(day)
    real = Path.rename

    def interrupted(path, target):
        if path.name.startswith(".publication-"):
            raise OSError("interrupted publication")
        return real(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", interrupted)
        with pytest.raises(OSError, match="interrupted publication"):
            publish(root, report, day, evaluation=plan)
    assert not list((plan.parent / "forecasts").glob(".publication-*"))
    assert publish(root, report, day, evaluation=plan).is_file()


def test_point_and_quantile_training_never_receive_target_or_future_labels(
    case, monkeypatch
):
    from dk2_forecast.research import ResearchData

    day = case[3]

    def contaminated(history, root):
        data = ResearchData(history, root)
        data.actual[data.days >= day] = 99999.0
        return data

    calls = []

    def fit(family, matrix, labels, params):
        assert not (labels == 99999.0).any()
        calls.append(family)
        return Model(), 0.01

    monkeypatch.setattr("dk2_forecast.research_forecast.ResearchData", contaminated)
    monkeypatch.setattr("dk2_forecast.research_forecast.fit_model", fit)
    prepare(case)
    assert len(calls) == 4


def test_end_to_end_dst_publication_api_actuals_aggregate_and_cache(case):
    root, report, plan, day, prices, clock, calls = case
    issued = issue(case)
    original = issued.read_bytes(), issued.with_suffix(".json").read_bytes()
    assert len(validate_publication(plan, root, day)[0]) == 100
    clock["now"] += timedelta(hours=3)
    assert publish(root, report, day, evaluation=plan) == issued
    assert original == (issued.read_bytes(), issued.with_suffix(".json").read_bytes())
    pending = aggregate(plan, root, now=clock["now"])
    assert pending["days"][0]["state"] == "published_awaiting_complete_actuals"
    assert pending["common_paired_intervals"] == 0
    score = aggregate(plan, root, prices, now=datetime(2026, 10, 25, 10, tzinfo=UTC))
    assert score["expected_intervals"] == 292
    assert score["due_intervals"] == 196
    assert score["common_paired_intervals"] == 100
    assert score["coverage"]["prediction"]["forecast_intervals"] == 100
    assert [s["state"] for s in score["days"]] == [
        "scored_publication",
        "missed_publication",
        "not_yet_due",
    ]
    # A later coverage-only inspection must not erase the scored actuals view.
    aggregate(plan, root, now=datetime(2026, 10, 26, 12, tzinfo=UTC))
    with TestClient(create_app(prices, experiment=plan)) as client:
        assert client.get(f"/forecasts/{day}").status_code == 200
        evaluation = client.get("/evaluation").json()
        assert evaluation["evaluation_id"] == plan.parent.name
        assert evaluation["latest_aggregate_score"]["common_paired_intervals"] == 100
        assert evaluation["operations"][0]["state"] == "scored_publication"
        assert len(client.get(f"/forecasts/{day}").json()["intervals"]) == 100
    before = len(calls)
    clock["now"] = datetime(2026, 10, 25, 8, 40, tzinfo=UTC)
    forecast(report, None, day + timedelta(days=1), root, replay=False, evaluation=plan)
    assert len(calls) == before  # point and all three quantiles reused


@pytest.mark.parametrize(
    "failure", ["replay", "late", "corrupt", "identity", "training", "receipt"]
)
def test_aggregate_rejects_ineligible_artifacts_without_losing_coverage(case, failure):
    root, _, plan, day, prices, _, _ = case
    path = issue(case)
    receipt = json.loads(path.with_suffix(".json").read_bytes())
    if failure == "replay":
        receipt["replay"] = True
    if failure == "late":
        receipt["published_at_utc"] = (
            forecast_issue_utc(day) + timedelta(minutes=2)
        ).isoformat()
    if failure == "identity":
        receipt["evaluation_id"] = "other"
    if failure == "training":
        (plan.parent / "training" / f"{day}.json").unlink()
    if failure == "receipt":
        receipt.pop("units")
    if failure == "corrupt":
        path.write_bytes(b"corrupt")
    path.with_suffix(".json").write_bytes(json_bytes(receipt))
    result = aggregate(plan, root, prices, now=datetime(2026, 10, 27, 12, tzinfo=UTC))
    assert result["days"][0]["state"] == "invalid_or_late_publication"
    assert result["coverage"]["prediction"]["forecast_intervals"] == 0
    assert result["common_paired_intervals"] == 0
    assert result["operational_point"] is None


def test_late_capture_missing_origin_and_labelled_model_fallback(case, monkeypatch):
    root, report, plan, day, prices, clock, _ = case
    clock["now"] = datetime(2026, 10, 24, 7, 51, tzinfo=UTC)
    with pytest.raises(ValueError, match="09:50"):
        prepare(case)
    clock["now"] = datetime(2026, 10, 25, 8, 40, tzinfo=UTC)
    with pytest.raises(ValueError, match="Missing block-origin"):
        forecast(
            report, None, day + timedelta(days=1), root, replay=False, evaluation=plan
        )
    clock["now"] = datetime(2026, 10, 24, 7, 40, tzinfo=UTC)

    def failure(*args):
        raise ValueError("deliberate model failure")

    monkeypatch.setattr("dk2_forecast.research_forecast.fit_model", failure)
    issue(case)
    score = aggregate(plan, root, prices, now=datetime(2026, 10, 25, 10, tzinfo=UTC))
    assert score["fallback_intervals"] == 100
    assert score["model_only_point"] is None
    assert score["fallback_reasons"] == {"deliberate model failure": 100}
    assert (
        score["operational_point"]["extended"]["methods"]["prediction"]["mae"]
        == score["operational_point"]["extended"]["methods"]["previous_day"]["mae"]
    )


def test_interrupted_preparation_cleans_staging_and_can_retry(case, monkeypatch):
    real = Path.rename

    def fail(path, target):
        if path.name.startswith(".preparation-"):
            raise OSError("interrupted")
        return real(path, target)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="interrupted"):
        prepare(case)
    assert not list(case[2].parent.rglob(".preparation-*"))
    monkeypatch.setattr(Path, "rename", real)
    assert prepare(case).is_file()


def test_schedule_bounds_and_dst_issue_times(case):
    root, _, plan, day, _, _, _ = case
    spec = schedule_spec(plan, root)
    assert len(spec["tasks"]) == 3
    assert spec["first_issue_date"] == "2026-10-24"
    assert spec["last_issue_date"] == "2026-10-26"
    assert spec["tasks"][0]["local_time"] == "09:40"
    assert all(t["end_boundary"] == "2026-10-27T00:00:00" for t in spec["tasks"])
    assert forecast_issue_utc(day).hour == 8
    assert forecast_issue_utc(day + timedelta(days=1)).hour == 9
    assert earliest_origin(datetime(2026, 3, 28, 8, 29, tzinfo=UTC)) == date(
        2026, 3, 29
    )
    assert earliest_origin(datetime(2026, 3, 28, 8, 30, tzinfo=UTC)) == date(
        2026, 3, 30
    )


def test_retained_launcher_executes_saved_schema_and_rejects_corrupt_bundle(case):
    import subprocess
    import sys

    root, _, plan, _, _, _, _ = case
    issue(case)
    command = [
        sys.executable,
        "-I",
        str(plan.with_name("launch.py")),
        "inspect",
        str(plan),
        "--root",
        str(root),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    metadata_path = Path(completed.stdout.strip())
    metadata = json.loads(metadata_path.read_bytes())
    assert metadata["exit_code"] == 0
    result = json.loads(metadata_path.with_suffix(".stdout.log").read_bytes())
    assert result["coverage"]["prediction"]["forecast_intervals"] == 100
    bundle = (
        root
        / "code"
        / load_evaluation(plan)["candidate"]["code_snapshot_sha256"]
        / "source.zip"
    )
    bundle.write_bytes(b"corrupted bundle")
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    failure = json.loads(Path(failed.stdout.strip()).read_bytes())
    assert "archive hash mismatch" in failure["failure_reason"]


def test_missing_quantile_origin_reports_shared_recovery_date(case):
    root, report, _, day, _, clock, _ = case
    plan = freeze(report, root, day, days=35, now=datetime(2026, 10, 23, tzinfo=UTC))
    clock["now"] = datetime(2026, 10, 31, 8, 40, tzinfo=UTC)
    with pytest.raises(ValueError, match="Missing quantile block-origin"):
        forecast(
            report, None, day + timedelta(days=7), root, replay=False, evaluation=plan
        )
    state = aggregate(plan, root, now=datetime(2026, 11, 1, 10, tzinfo=UTC))["days"][7]
    assert state["training_state"] == "missing_quantile_block_origin_receipt"
    assert state["next_origin"] == "2026-11-22"


def test_model_cache_rejects_a_different_training_origin(case):
    from dk2_forecast.research_forecast import cached_model

    root, _, _, day, _, _, _ = case
    matrix = np.ones((4, 2))
    labels = np.ones(4)
    cached_model(root, day, "catboost", matrix, labels, {}, {"data": "fixture"})
    receipt = next((root / "model_cache").glob("*/model.json"))
    saved = json.loads(receipt.read_bytes())
    saved["origin"] = "2000-01-01"
    receipt.write_bytes(json_bytes(saved))
    with pytest.raises(ValueError, match="origin/configuration"):
        cached_model(root, day, "catboost", matrix, labels, {}, {"data": "fixture"})


def test_original_scheduler_collection_uses_original_dispatch(
    tmp_path, monkeypatch, capsys
):
    from dk2_forecast import research_ops

    plan = tmp_path / "original.json"
    plan.write_bytes(
        json_bytes({"models": {}, "start": "2026-03-28", "end_exclusive": "2026-04-02"})
    )
    spec = schedule_spec(plan, tmp_path)
    assert spec["type"] == "original"
    assert spec["first_issue_date"] == "2026-03-27"
    assert spec["last_issue_date"] == "2026-03-31"
    assert spec["tasks"][0]["local_time"] == "09:50"
    assert "dk2_forecast.operations prepare" in spec["tasks"][0]["arguments"]
    monkeypatch.setattr("sys.argv", ["research_ops", "collect-score", str(plan)])
    calls = []

    def worker(action, selected, root, **kwargs):
        calls.append((action, selected))
        return {"original_dispatch": True}

    monkeypatch.setattr(research_ops, "worker", worker)
    research_ops.main()
    assert calls == [("collect-score", plan)]
    assert json.loads(capsys.readouterr().out)["original_dispatch"] is True
