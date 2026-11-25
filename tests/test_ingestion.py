import json
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from dk2_forecast.ingestion import fetch_dk2_prices, ingest_prices
from dk2_forecast.ingestion import request_prices
from dk2_forecast.storage import connect, digest


def make_records(first_utc, count):
    return [
        {
            "PriceArea": "DK2",
            "TimeUTC": (first_utc + timedelta(minutes=15 * index)).isoformat(),
            "DayAheadPriceEUR": -10.25,
        }
        for index in range(count)
    ]


@pytest.mark.parametrize(
    ("day", "first_utc", "count"),
    [
        (date(2026, 3, 29), datetime(2026, 3, 28, 23, tzinfo=UTC), 92),
        (date(2026, 10, 5), datetime(2026, 10, 4, 22, tzinfo=UTC), 96),
        (date(2026, 10, 25), datetime(2026, 10, 24, 22, tzinfo=UTC), 100),
    ],
)
def test_fetch_complete_delivery_day(day, first_utc, count):
    records = make_records(first_utc.replace(tzinfo=None), count)

    def respond(request):
        assert request.url.path == "/dataset/DayAheadPrices"
        assert request.url.params["start"] == day.isoformat()
        assert request.url.params["end"] == (day + timedelta(days=1)).isoformat()
        assert request.url.params["filter"] == '{"PriceArea":["DK2"]}'
        assert request.url.params["limit"] == "0"
        return httpx.Response(200, json={"records": records[::-1]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        prices = fetch_dk2_prices(client, day)

    assert [price.time_utc for price in prices] == [
        first_utc + timedelta(minutes=15 * index) for index in range(count)
    ]
    assert all(price.price_eur_per_mwh == -10.25 for price in prices)


@pytest.mark.parametrize("problem", ["missing", "conflict", "wrong area", "null price"])
def test_rejects_invalid_delivery_day(tmp_path, problem):
    records = make_records(datetime(2026, 10, 4, 22, tzinfo=UTC), 96)
    if problem == "missing":
        records.pop()
        message = "1 missing"
    elif problem == "conflict":
        records.append({**records[0], "DayAheadPriceEUR": 42.5})
        message = "Conflicting DK2 prices"
    elif problem == "wrong area":
        records[-1]["PriceArea"] = "DK1"
        message = "Expected only DK2"
    else:
        records[-1]["DayAheadPriceEUR"] = None
        message = "Invalid DK2 price"

    day = date(2026, 10, 5)
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"records": records})
    )
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(ValueError, match=message),
    ):
        ingest_prices(tmp_path, day, day + timedelta(days=1), client)

    assert len(list((tmp_path / "raw" / "objects").glob("*.json"))) == 1
    assert not (tmp_path / "prices").exists()


def test_ingest_replays_and_deduplicates_prices(tmp_path):
    day = date(2026, 10, 25)
    first_utc = datetime(2026, 10, 24, 22, tzinfo=UTC)
    records = make_records(first_utc, 100)
    records.append(records[0].copy())
    end = day + timedelta(days=1)
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"records": records})
    )
    with httpx.Client(transport=transport) as client:
        path = ingest_prices(tmp_path, day, end, client)
    original = path.read_bytes()
    path.unlink()
    assert ingest_prices(tmp_path, day, end) == path
    assert path.read_bytes() == original
    raw = (tmp_path / "raw" / f"{day}_{end}.json").read_bytes()
    manifest = path.with_name("manifest.json").read_bytes()
    assert digest(manifest) == path.parent.name
    assert json.loads(manifest)["raw"][0]["sha256"] == digest(raw)
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY delivery_start_utc", [str(path)]
        ).fetchall()

    assert len(rows) == 100
    assert rows[0] == ("DK2", first_utc, -10.25)


def test_source_correction_is_a_new_immutable_version(tmp_path):
    day = date(2026, 10, 5)
    records = make_records(datetime(2026, 10, 4, 22, tzinfo=UTC), 96)
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"records": records})
    )
    with httpx.Client(transport=transport) as client:
        original = ingest_prices(tmp_path, day, day + date.resolution, client)
        original_bytes = original.read_bytes()
        records[0]["DayAheadPriceEUR"] = 7000.0
        corrected = ingest_prices(
            tmp_path, day, day + date.resolution, client, refresh=True
        )
    assert corrected != original
    assert original.read_bytes() == original_bytes
    assert ingest_prices(tmp_path, day, day + date.resolution) == original
    assert list((tmp_path / "raw" / "observations").glob("*.json"))


def test_incomplete_download_recovers_without_poisoning_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("dk2_forecast.ingestion.sleep", lambda _: None)
    day = date(2026, 10, 5)
    rows = make_records(datetime(2026, 10, 4, 22, tzinfo=UTC), 96)
    responses = iter((rows[:-1], rows))
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"records": next(responses)})
        )
    ) as client:
        path = ingest_prices(tmp_path, day, day + date.resolution, client)
    assert path.exists()
    assert len(list((tmp_path / "raw" / "objects").glob("*.json"))) == 2


def test_rate_limit_is_bounded_and_retains_error_response(tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr("dk2_forecast.ingestion.sleep", delays.append)
    responses = iter(
        (
            httpx.Response(
                429, text="Rate limit exceeded", headers={"Retry-After": "2"}
            ),
            httpx.Response(200, json={"records": []}),
        )
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: next(responses))
    ) as client:
        raw = request_prices(
            client, date(2026, 10, 5), date(2026, 10, 6), archive_root=tmp_path
        )
    assert json.loads(raw) == {"records": []}
    assert delays == [2]
    receipts = list((tmp_path / "raw" / "errors").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_bytes())["status_code"] == 429
    assert len(list((tmp_path / "raw" / "errors").glob("*.body"))) == 1
