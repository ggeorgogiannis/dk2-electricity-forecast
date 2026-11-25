import json
from datetime import UTC, date, datetime, timedelta

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from dk2_forecast.api import create_app
from dk2_forecast.backtest import run_backtest
from dk2_forecast.delivery_day import delivery_intervals_utc, forecast_issue_utc
from dk2_forecast.forecast import features, run_forecast
from dk2_forecast.ingestion import ingest_prices
from dk2_forecast.issue import publish_prepared
from dk2_forecast.storage import digest


def test_autumn_lag_cannot_use_early_prices_from_target_day():
    day = date(2026, 10, 25)
    timestamp = delivery_intervals_utc(day)[-1]
    prices = {
        timestamp - timedelta(days=1): 9000.0,
        timestamp - timedelta(days=7): -12.0,
    }
    matrix = features(prices, [timestamp], day)
    assert np.isnan(matrix[0, 4])
    assert matrix[0, 7] == -12.0
    assert forecast_issue_utc(day) == datetime(2026, 10, 24, 8, tzinfo=UTC)
    assert forecast_issue_utc(day + timedelta(days=1)) == datetime(
        2026, 10, 25, 9, tzinfo=UTC
    )


@pytest.fixture
def history_response():
    start = date(2025, 12, 20)
    end = date(2026, 3, 31)

    def create_snapshot(root, target_delta=0):
        def respond(request):
            first = date.fromisoformat(request.url.params["start"])
            stop = date.fromisoformat(request.url.params["end"])
            records = []
            for offset in range((stop - first).days):
                day = first + timedelta(days=offset)
                for quarter, timestamp in enumerate(delivery_intervals_utc(day)):
                    price = quarter - 40 + day.weekday() * 2
                    if day >= date(2026, 3, 29):
                        price += target_delta
                    records.append(
                        {
                            "PriceArea": "DK2",
                            "TimeUTC": timestamp.isoformat(),
                            "DayAheadPriceEUR": price,
                        }
                    )
            return httpx.Response(200, json={"records": records})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            return ingest_prices(root, start, end, client)

    return create_snapshot


def test_replay_backtest_and_serve_spring_forecast(tmp_path, history_response):
    day = date(2026, 3, 29)
    prices = history_response(tmp_path)
    replay = run_forecast(prices, tmp_path, day)
    assert run_forecast(prices, tmp_path, day) == replay
    report_path = run_backtest(
        prices, tmp_path, date(2026, 3, 23), day, date(2026, 3, 31)
    )
    report = json.loads((report_path / "report.json").read_text())
    assert report["test"]["all"]["intervals"] == 188
    assert set(report["test"]["coverage"].values()) == {1.0}
    with TestClient(create_app(prices, tmp_path / "forecasts")) as client:
        actual = client.get(f"/prices/{day}")
        forecast = client.get(f"/forecasts/{day}")
        assert client.get(f"/forecasts/{day}?model_version=missing").status_code == 404
        assert client.get("/prices/2025-01-01").status_code == 404
    assert actual.status_code == forecast.status_code == 200
    rows = forecast.json()["intervals"]
    assert len(rows) == len(actual.json()["intervals"]) == 92
    assert rows[0]["issue_time_utc"] == "2026-03-28T09:00:00+00:00"
    assert rows[0]["data_version"] == actual.json()["data_version"]
    assert rows[0]["model_version"] == report["test"]["model_version"]

    changed = history_response(tmp_path / "changed", target_delta=10000)
    revised = run_forecast(changed, tmp_path / "changed", day)
    with TestClient(create_app(changed, revised)) as client:
        revised_rows = client.get(f"/forecasts/{day}").json()["intervals"]
    for original, updated in zip(rows, revised_rows, strict=True):
        assert original["model_version"] == updated["model_version"]
        assert original["lightgbm_eur_per_mwh"] == updated["lightgbm_eur_per_mwh"]
        assert original["baseline_eur_per_mwh"] == updated["baseline_eur_per_mwh"]


def test_live_publisher_respects_issue_minute(tmp_path):
    day = "2026-10-25"
    stage = tmp_path / "prepared" / day / "fixture.parquet"
    stage.parent.mkdir(parents=True)
    stage.write_bytes(b"prepared fixture")
    with pytest.raises(ValueError, match="10:00"):
        publish_prepared(tmp_path, datetime(2026, 10, 24, 7, 59, tzinfo=UTC))
    receipt = {
        "forecast_sha256": digest(stage.read_bytes()),
        "observed_at_utc": "2026-10-24T07:50:00+00:00",
        "prepared_at_utc": "2026-10-24T08:00:01+00:00",
    }
    stage.with_suffix(".json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="pre-cutoff"):
        publish_prepared(tmp_path, datetime(2026, 10, 24, 8, tzinfo=UTC))
    receipt["prepared_at_utc"] = "2026-10-24T07:51:00+00:00"
    stage.with_suffix(".json").write_text(json.dumps(receipt))
    published = publish_prepared(tmp_path, datetime(2026, 10, 24, 8, tzinfo=UTC))
    assert published == tmp_path / "forecasts" / day / stage.name
    assert published.read_bytes() == stage.read_bytes()
    assert json.loads(published.with_suffix(".json").read_text())[
        "published_at_utc"
    ] == ("2026-10-24T08:00:00+00:00")
