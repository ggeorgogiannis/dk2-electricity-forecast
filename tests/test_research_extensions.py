from datetime import UTC, date, datetime, timedelta

import httpx
import numpy as np
import pytest

from dk2_forecast.delivery_day import delivery_intervals_utc
from dk2_forecast.failure_analysis import prior_gate
from dk2_forecast.fundamentals import capture, parse_vintage
from dk2_forecast.research import tune_tree
from dk2_forecast.research_metrics import interval_metrics


def test_prior_gate_uses_only_prior_validation_labels():
    days = np.asarray(
        [date(2026, 1, 1) + timedelta(days=i // 96) for i in range(96 * 70)]
    )
    previous = np.linspace(-30, 250, len(days))
    model = previous + 20
    actual = previous + 15
    original, _ = prior_gate(days, actual, model, previous)
    changed = actual.copy()
    changed[days >= date(2026, 2, 12)] = 99999
    updated, _ = prior_gate(days, changed, model, previous)
    np.testing.assert_allclose(
        original[days < date(2026, 2, 19)], updated[days < date(2026, 2, 19)]
    )


def test_interval_score_penalizes_misses_and_reports_sample_counts():
    stamps = delivery_intervals_utc(date(2026, 3, 29))
    result = interval_metrics(stamps, [30.0] * 92, np.tile([0.0, 10.0, 20.0], (92, 1)))
    assert result["mean_interval_score_80"] == 120
    assert result["regime"]["regular"]["intervals"] == 92
    assert result["coverage_80"] == 0


def xml(day):
    start = delivery_intervals_utc(day)[0]
    end = delivery_intervals_utc(day)[-1] + timedelta(minutes=15)
    points = "".join(
        f"<Point><position>{i + 1}</position><quantity>100</quantity></Point>"
        for i in range(len(delivery_intervals_utc(day)) // 4)
    )
    return f"""<Publication_MarketDocument xmlns="urn:test"><mRID>fixture</mRID><revisionNumber>1</revisionNumber><type>A65</type><process.processType>A01</process.processType><createdDateTime>2026-10-24T07:00:00Z</createdDateTime><TimeSeries><outBiddingZone_Domain.mRID>10YDK-2--------M</outBiddingZone_Domain.mRID><quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name><curveType>A01</curveType><Period><timeInterval><start>{start.isoformat()}</start><end>{end.isoformat()}</end></timeInterval><resolution>PT60M</resolution>{points}</Period></TimeSeries></Publication_MarketDocument>""".encode()


def test_fundamental_forecast_vintage_validates_timing_resolution_and_domain():
    day = date(2026, 10, 25)
    raw = xml(day)
    result = parse_vintage(raw, "load", day, datetime(2026, 10, 24, 7, 30, tzinfo=UTC))
    assert len(result["rows"]) == 100
    assert result["source_resolutions"] == ["PT60M"]
    assert result["model_initialization_utc"] is None
    with pytest.raises(ValueError, match="cutoff"):
        parse_vintage(raw, "load", day, datetime(2026, 10, 24, 8, tzinfo=UTC))
    with pytest.raises(ValueError, match="never actuals"):
        parse_vintage(
            raw.replace(b"A01</process", b"A16</process"),
            "load",
            day,
            datetime(2026, 10, 24, 7, 30, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="DK2"):
        parse_vintage(
            raw.replace(b"10YDK-2--------M", b"10YDK-1--------W"),
            "load",
            day,
            datetime(2026, 10, 24, 7, 30, tzinfo=UTC),
        )


def test_optional_fundamental_credentials_are_required(tmp_path, monkeypatch):
    monkeypatch.delenv("ENTSOE_API_TOKEN", raising=False)
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(403))
    ) as client:
        with pytest.raises(ValueError, match="ENTSOE_API_TOKEN"):
            capture(tmp_path, "load", date(2099, 1, 1), client)


def test_search_reports_startup_and_adaptive_trials_and_resumes(tmp_path):
    class Data:
        def evaluate(self, family, params, **kwargs):
            result = {
                "mae": params["learning_rate"],
                "parameters": params,
                "folds": [],
                "fit_seconds": 0.01,
            }
            return None, result

    baseline = {"mae": 100.0, "parameters": {}}
    _, first = tune_tree(
        Data(),
        "lightgbm",
        tmp_path,
        "version-one",
        baseline,
        None,
        6,
        60,
        startup_trials=3,
    )
    assert first["random_startup_completed"] == 3
    assert first["adaptive_tpe_completed"] == 3
    _, second = tune_tree(
        Data(),
        "lightgbm",
        tmp_path,
        "version-one",
        baseline,
        None,
        7,
        60,
        startup_trials=3,
    )
    assert second["completed_trials"] == 7
    assert second["adaptive_tpe_completed"] == 4
    assert second["trials"][:6] == first["trials"]
