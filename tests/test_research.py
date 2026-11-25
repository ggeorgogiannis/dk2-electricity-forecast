from datetime import UTC, date, datetime, timedelta
from pathlib import Path
import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from dk2_forecast.api import create_app
from dk2_forecast.delivery_day import delivery_intervals_utc
from dk2_forecast.experiment import json_bytes
from dk2_forecast.explanatory import VARIABLES, align_weather, capture, snapshot_cutoff
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.operations import exclusive_run
from dk2_forecast.research_features import daily_price_features
from dk2_forecast.research_metrics import interval_metrics
from dk2_forecast.research import fit_model, predict_model
from dk2_forecast.storage import publish_bytes
from dk2_forecast.storage import digest
from dk2_forecast.storage import connect, write_parquet
from dk2_forecast.research_forecast import current_code_hash, publish


def test_price_curves_and_labels_obey_cutoff():
    day = date(2026, 10, 25)
    prices = {stamp: -20.0 for stamp in delivery_intervals_utc(day - date.resolution)}
    prices.update(
        {stamp: 80.0 for stamp in delivery_intervals_utc(day - timedelta(days=7))}
    )
    history = PriceHistory(prices, "fixture")
    stamps, original, _, _ = daily_price_features(history, day)
    history.prices.update({stamp: 99999.0 for stamp in stamps})
    _, revised, _, _ = daily_price_features(history, day)
    np.testing.assert_allclose(original, revised, equal_nan=True)
    assert len(original) == 100
    assert (original[-4:, 0] == -20).all()


def test_weather_assumption_is_excluded_from_strict_historical_features(tmp_path):
    day = date(2026, 6, 1)
    times = [
        datetime(2026, 5, 30, 18, tzinfo=UTC) + timedelta(hours=i) for i in range(84)
    ]
    payload = {
        "utc_offset_seconds": 0,
        "latitude": 55.75,
        "longitude": 12.5,
        "elevation": 17,
        "hourly_units": VARIABLES,
        "hourly": {
            "time": [stamp.replace(tzinfo=None).isoformat() for stamp in times],
            **{name: [12.0] * len(times) for name in VARIABLES},
        },
    }
    metadata = {
        "last_run_initialisation_time": times[0].timestamp(),
        "last_run_availability_time": (times[0] + timedelta(hours=8)).timestamp(),
    }
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json=metadata if request.url.path.endswith("meta.json") else payload
        )
    )
    with httpx.Client(transport=transport) as client:
        capture(tmp_path, day, client, historical_assumption=True)
    stamps = delivery_intervals_utc(day)
    strict, versions = align_weather(tmp_path, stamps)
    assert np.isnan(strict).all()
    assert not versions
    assumed, versions = align_weather(tmp_path, stamps, allow_assumed=True)
    assert (assumed == 12).all()
    assert len(versions) == 1
    original_manifest_path = next(
        (tmp_path / "weather" / "snapshots" / str(day)).glob("*/manifest.json")
    )
    original_manifest = json.loads(original_manifest_path.read_bytes())
    early_manifest = original_manifest | {
        "status": "verified_live",
        "historical_availability_assumption": None,
        "retrieved_at_utc": "2026-05-31T07:00:00+00:00",
    }
    early_content = json_bytes(early_manifest)
    early_folder = original_manifest_path.parents[1] / digest(early_content)
    early_folder.mkdir()
    (early_folder / "manifest.json").write_bytes(early_content)
    (early_folder / "weather.parquet").write_bytes(
        original_manifest_path.with_name("weather.parquet").read_bytes()
    )
    before, _ = align_weather(tmp_path, stamps)
    late_manifest = early_manifest | {
        "run_initialization_utc": "2026-05-31T06:00:00+00:00",
        "available_at_utc": "2026-05-31T09:00:00+00:00",
        "retrieved_at_utc": "2026-05-31T09:01:00+00:00",
    }
    late_content = json_bytes(late_manifest)
    late_folder = original_manifest_path.parents[1] / digest(late_content)
    late_folder.mkdir()
    (late_folder / "manifest.json").write_bytes(late_content)
    with connect() as db:
        db.execute(
            "CREATE TABLE late_weather AS SELECT * REPLACE (99999.0 AS value) FROM read_parquet(?)",
            [str(early_folder / "weather.parquet")],
        )
        write_parquet(db, "SELECT * FROM late_weather", late_folder / "weather.parquet")
    after, _ = align_weather(tmp_path, stamps)
    np.testing.assert_allclose(before, after)
    assert (after == 12).all()
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(ValueError, match="09:50"),
    ):
        capture(tmp_path, day, client)
    assert snapshot_cutoff(day).hour == 7


def test_publication_is_immutable_and_locks_prevent_overlap(tmp_path):
    path = tmp_path / "artifact.json"
    assert publish_bytes(path, b"first") == b"first"
    assert publish_bytes(path, b"changed") == b"first"
    lock = tmp_path / "run.lock"
    with pytest.raises(RuntimeError):
        with exclusive_run(lock):
            with pytest.raises(FileExistsError):
                with exclusive_run(lock):
                    pass
            raise RuntimeError("Interrupted preparation")
    assert not lock.exists()


def test_interval_scoring_records_raw_crossings():
    stamps = delivery_intervals_utc(date(2026, 10, 25))
    quantiles = np.tile([20, 10, 30], (100, 1))
    result = interval_metrics(stamps, [15] * 100, quantiles)
    assert result["raw_crossings"] == 100
    assert result["coverage_80"] == 1
    assert result["mean_width"] == 20


def test_research_preprocessing_fits_only_training_values():
    train = np.asarray([[1.0, np.nan], [3.0, 10.0], [5.0, 20.0]])
    target = np.asarray([1.0, 2.0, 3.0])
    model, _ = fit_model("ridge", train, target, {"alpha": 1.0})
    before = model.named_steps["simpleimputer"].statistics_.copy()
    predict_model(model, np.asarray([[10000.0, 50000.0]]))
    np.testing.assert_allclose(before, [3.0, 15.0])
    np.testing.assert_allclose(model.named_steps["simpleimputer"].statistics_, before)


def test_publication_commits_forecast_and_receipt_together_and_retries(
    tmp_path, monkeypatch
):
    report = tmp_path / "report.json"
    report.write_bytes(b"{}")
    day = date(2026, 11, 20)
    base = tmp_path / "research_live" / digest(report.read_bytes())
    stage = base / "prepared" / str(day) / "forecast.parquet"
    stage.parent.mkdir(parents=True)
    stage.write_bytes(b"staged fixture")
    code = current_code_hash()
    (base / "freeze.json").write_bytes(json_bytes({"code_sha256": code}))
    stage.with_suffix(".json").write_bytes(
        json_bytes(
            {
                "replay": False,
                "created_at_utc": "2026-11-19T08:45:00+00:00",
                "forecast_sha256": digest(stage.read_bytes()),
                "code_sha256": code,
                "input": {"observed_at_utc": "2026-11-19T08:44:00+00:00"},
            }
        )
    )

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 19, 9, tzinfo=UTC)

    monkeypatch.setattr("dk2_forecast.research_forecast.datetime", Clock)
    original_rename = Path.rename

    def interrupted_rename(path, target):
        if path.name.startswith(".publication-"):
            raise OSError("Simulated interruption before commit")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", interrupted_rename)
    with pytest.raises(OSError, match="Simulated interruption"):
        publish(tmp_path, report, day)
    assert not (base / "forecasts" / str(day)).exists()
    assert not list((base / "forecasts").glob(".publication-*"))
    monkeypatch.setattr(Path, "rename", original_rename)
    result = publish(tmp_path, report, day)
    assert result.read_bytes() == stage.read_bytes()
    assert result.with_suffix(".json").is_file()
    assert publish(tmp_path, report, day) == result


def test_api_serves_new_forecast_schema_and_evaluation(tmp_path):
    from dk2_forecast.experiment import save_predictions

    day = date(2026, 11, 18)
    stamps = delivery_intervals_utc(day)
    path = tmp_path / "forecasts" / str(day) / "forecast.parquet"
    rows = [
        {
            "price_area": "DK2",
            "delivery_date": day,
            "delivery_start_utc": stamp,
            "issue_time_utc": datetime(2026, 11, 17, 9, tzinfo=UTC),
            "model_training_origin": day,
            "data_version": "fixture",
            "previous_source_utc": stamp - timedelta(days=1),
            "weekly_fallback": False,
            "previous_day": 10.0,
            "catboost": 12.0,
        }
        for stamp in stamps
    ]
    save_predictions(path, rows, ("previous_day", "catboost"))
    (tmp_path / "freeze.json").write_bytes(json_bytes({"start": str(day)}))
    with TestClient(
        create_app(Path("absent.parquet"), tmp_path / "forecasts", tmp_path)
    ) as client:
        response = client.get(f"/forecasts/{day}")
        assert response.status_code == 200
        assert len(response.json()["intervals"]) == 96
        assert response.json()["units"] == "EUR/MWh"
        assert client.get("/evaluation").status_code == 200
        assert client.get("/prices/9999-12-31").status_code == 422
        assert client.get("/prices/invalid").status_code == 422
        assert client.get(f"/prices/{day}").status_code == 503
