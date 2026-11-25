import os
import json
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
import duckdb

from dk2_forecast.delivery_day import delivery_intervals_utc
from dk2_forecast.ingestion import PRICE_AREA
from dk2_forecast.storage import connect


def create_app(
    prices: Path | None = None,
    forecasts: Path | None = None,
    experiment: Path | None = None,
) -> FastAPI:
    prices = prices or Path(os.environ["DK2_PRICES"])
    app = FastAPI(title="DK2 day-ahead prices and forecasts")
    experiment = experiment or (
        Path(os.environ["DK2_EXPERIMENT"]) if "DK2_EXPERIMENT" in os.environ else None
    )
    if experiment and experiment.is_file():
        experiment = experiment.parent
    forecasts = forecasts or Path(
        os.environ.get(
            "DK2_FORECASTS",
            str(experiment / "forecasts") if experiment else "data/forecasts",
        )
    )

    @app.get("/health")
    def health():
        return {
            "prices_available": prices.is_file(),
            "units": "EUR/MWh",
            "price_area": PRICE_AREA,
        }

    @app.get("/prices/{delivery_date}")
    def get_prices(delivery_date: date):
        if delivery_date == date.max:
            raise HTTPException(
                status_code=422, detail="Delivery date exceeds supported range"
            )
        if not prices.is_file():
            raise HTTPException(status_code=503, detail="Price storage is unavailable")
        start = delivery_intervals_utc(delivery_date)[0]
        end = delivery_intervals_utc(delivery_date + timedelta(days=1))[0]
        with connect() as connection:
            query = connection.execute(
                """
                SELECT price_area, delivery_start_utc, price_eur_per_mwh
                FROM read_parquet(?)
                WHERE price_area = ? AND delivery_start_utc >= ? AND delivery_start_utc < ?
                ORDER BY delivery_start_utc
                """,
                [str(prices), PRICE_AREA, start, end],
            )
            columns = [column[0] for column in query.description]
            rows = query.fetchall()
        if not rows:
            raise HTTPException(
                status_code=404, detail="No stored prices for this delivery date"
            )
        return {
            "delivery_date": delivery_date,
            "data_version": prices.parent.name,
            "units": "EUR/MWh",
            "intervals": [dict(zip(columns, row, strict=True)) for row in rows],
        }

    @app.get("/forecasts/{delivery_date}")
    def get_forecasts(
        delivery_date: date,
        model_version: str | None = None,
        data_version: str | None = None,
    ):
        if experiment and (experiment / "freeze.json").is_file():
            selected = json.loads((experiment / "freeze.json").read_bytes())
            if selected.get("schema_version") == 2:
                from dk2_forecast.evaluation import validate_publication

                try:
                    validate_publication(
                        experiment / "freeze.json", experiment.parents[2], delivery_date
                    )
                except FileNotFoundError:
                    raise HTTPException(
                        status_code=404,
                        detail="No issued forecast for the selected evaluation",
                    ) from None
                except (ValueError, KeyError, OSError, duckdb.Error) as error:
                    raise HTTPException(
                        status_code=409, detail=f"Invalid publication: {error}"
                    ) from error
        paths = (
            [forecasts]
            if forecasts.is_file()
            else sorted((forecasts / delivery_date.isoformat()).glob("*.parquet"))
        )
        if not paths:
            raise HTTPException(
                status_code=404, detail="No stored forecast for this delivery date"
            )
        with connect() as connection:
            schema = connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [list(map(str, paths))]
            ).fetchall()
            names = {column[0] for column in schema}
            if "model_version" not in names and model_version is not None:
                raise HTTPException(
                    status_code=404,
                    detail="Stored forecast has no matching model version",
                )
            version_filter = (
                "AND (?::VARCHAR IS NULL OR model_version = ?)"
                if "model_version" in names
                else ""
            )
            parameters = [list(map(str, paths)), delivery_date, PRICE_AREA]
            if version_filter:
                parameters.extend((model_version, model_version))
            parameters.extend((data_version, data_version))
            query = connection.execute(
                f"""
                SELECT * FROM read_parquet(?)
                WHERE delivery_date = ? AND price_area = ?
                    {version_filter}
                    AND (?::VARCHAR IS NULL OR data_version = ?)
                ORDER BY delivery_start_utc, data_version
                """,
                parameters,
            )
            columns = [column[0] for column in query.description]
            rows = query.fetchall()
        if not rows:
            raise HTTPException(
                status_code=404, detail="No stored forecast matches the request"
            )
        receipts = [
            json.loads(path.with_suffix(".json").read_bytes())
            for path in paths
            if path.with_suffix(".json").is_file()
        ]
        return {
            "delivery_date": delivery_date,
            "units": "EUR/MWh",
            "metadata": receipts,
            "intervals": [dict(zip(columns, row, strict=True)) for row in rows],
        }

    @app.get("/evaluation")
    def evaluation_status():
        if experiment is None or not (experiment / "freeze.json").is_file():
            raise HTTPException(status_code=404, detail="No evaluation configured")
        plan = json.loads((experiment / "freeze.json").read_bytes())
        operations = []
        for day in sorted((experiment / "operations").glob("*")):
            files = sorted(day.glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
            if files:
                operations.append(json.loads(files[-1].read_bytes()))
        try:
            scores = [
                json.loads(path.read_bytes())
                for path in sorted((experiment / "scores").glob("*.json"))
            ]
        except ValueError as error:
            raise HTTPException(
                status_code=409, detail="Corrupt evaluation score"
            ) from error
        if plan.get("schema_version") == 2:
            from dk2_forecast.evaluation import aggregate, load_evaluation

            try:
                load_evaluation(experiment / "freeze.json")
                from dk2_forecast.storage import digest

                for file in (experiment / "scores").glob("*.json"):
                    stored = json.loads(file.read_bytes())
                    if (
                        digest(file.read_bytes()) != file.stem
                        or stored["evaluation_id"] != plan["evaluation_id"]
                    ):
                        raise ValueError("Aggregate score identity/hash mismatch")
                # Operational inspections may save a coverage-only snapshot. They
                # must not hide an earlier actual-price comparison in the API.
                compared = [
                    score for score in scores if score.get("actuals_data_version")
                ]
                latest = (
                    max(compared or scores, key=lambda score: score["created_at_utc"])
                    if scores
                    else None
                )
                actuals = None
                if latest and latest.get("actuals_data_version"):
                    data_version = latest["actuals_data_version"]
                    if len(data_version) != 64 or any(
                        c not in "0123456789abcdef" for c in data_version
                    ):
                        raise ValueError("Invalid actual-price version")
                    actuals = (
                        experiment.parents[2]
                        / "prices"
                        / data_version
                        / "prices.parquet"
                    )
                    if digest(actuals.read_bytes()) != latest["actuals_sha256"]:
                        raise ValueError("Scored actual-price snapshot hash mismatch")
                state = aggregate(
                    experiment / "freeze.json",
                    experiment.parents[2],
                    actuals,
                    save=False,
                )
            except (ValueError, KeyError, OSError) as error:
                raise HTTPException(
                    status_code=409, detail=f"Invalid evaluation: {error}"
                ) from error
            return {
                "plan": plan,
                "candidate_id": plan["candidate_id"],
                "evaluation_id": plan["evaluation_id"],
                "operations": state["days"],
                "coverage": state["coverage"],
                "latest_aggregate_score": latest,
                "scores": scores,
            }
        return {"plan": plan, "operations": operations, "scores": scores}

    return app
