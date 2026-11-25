import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dk2_forecast.experiment import json_bytes
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.experiment_models import ComparisonTraining
from dk2_forecast.explanatory import align_weather
from dk2_forecast.research import END, START, ResearchData
from dk2_forecast.storage import connect, digest, publish_bytes


def run(report_path, prices, root):
    report = json.loads(report_path.read_bytes())
    if report["data_version"] != prices.parent.name:
        raise ValueError("Ablation requires the same price snapshot as the comparison")
    data = ResearchData(PriceHistory.load(prices), root)
    selected = (data.days >= START) & (data.days < END)
    eligible = np.isfinite(data.weather[selected]).all(axis=1)
    if not eligible.any():
        raise ValueError("No complete weather snapshots overlap development evaluation")
    full_weather = data.weather.copy()
    with connect() as db:
        base = np.asarray(
            db.execute(
                "SELECT lightgbm_tuned FROM read_parquet(?) ORDER BY delivery_start_utc",
                [str(report_path.with_name("predictions.parquet"))],
            ).fetchall()
        )[:, 0]
    results = {
        "price_curves": {
            "mae": float(
                np.abs(data.actual[selected][eligible] - base[eligible]).mean()
            )
        }
    }
    price_matrix = data.matrix
    compact = ComparisonTraining(data.history, START)
    if compact.data.intervals != data.timestamps:
        raise ValueError("Compact and curve feature timestamps differ")
    data.matrix = compact.matrix
    compact_prediction, compact_fit = data.evaluate(
        "lightgbm",
        report["fits"]["lightgbm_tuned"]["parameters"],
        weeks=report["selection"]["training_weeks"],
    )
    results["price_calendar_11"] = compact_fit | {
        "mae": float(
            np.abs(
                data.actual[selected][eligible] - compact_prediction[eligible]
            ).mean()
        )
    }
    data.matrix = price_matrix
    for name, columns in (
        ("plus_temperature", [0]),
        ("plus_temperature_wind", [0, 1]),
        ("plus_temperature_wind_radiation", [0, 1, 2]),
    ):
        data.weather = full_weather[:, columns]
        prediction, fit = data.evaluate(
            "lightgbm",
            report["fits"]["lightgbm_tuned"]["parameters"],
            weeks=report["selection"]["training_weeks"],
            weather=True,
        )
        results[name] = fit | {
            "mae": float(
                np.abs(data.actual[selected][eligible] - prediction[eligible]).mean()
            )
        }
    data.weather, _ = align_weather(
        root, data.timestamps, allow_assumed=True, interpolate=True
    )
    interpolated, fit = data.evaluate(
        "lightgbm",
        report["fits"]["lightgbm_tuned"]["parameters"],
        weeks=report["selection"]["training_weeks"],
        weather=True,
    )
    results["plus_weather_linear_interpolation"] = fit | {
        "mae": float(
            np.abs(data.actual[selected][eligible] - interpolated[eligible]).mean()
        )
    }
    content = json_bytes(
        {
            "status": "exploratory_historical_availability_assumption",
            "created_at_utc": datetime.now(UTC),
            "assumption": "Selected previous-day 18 UTC run, assumed available eight hours after initialization",
            "research_report_sha256": digest(report_path.read_bytes()),
            "data_version": report["data_version"],
            "weather_versions": data.weather_versions,
            "paired_intervals": int(eligible.sum()),
            "expected_intervals": len(eligible),
            "coverage": float(eligible.mean()),
            "results": results,
        }
    )
    path = report_path.parent / "ablations" / f"{digest(content)}.json"
    publish_bytes(path, content)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Compare successive weather additions on identical eligible intervals."
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("prices", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(run(args.report, args.prices, args.root))


if __name__ == "__main__":
    main()
