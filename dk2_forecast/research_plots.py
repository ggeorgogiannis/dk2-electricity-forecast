"""Export publication figures and representative failures from saved predictions."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/plot-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from dk2_forecast.delivery_day import COPENHAGEN
from dk2_forecast.storage import connect


def render(report_path: Path, output: Path):
    report = json.loads(report_path.read_bytes())
    output.mkdir(parents=True, exist_ok=True)
    candidate = report["selection"]["model"]
    if report["selection"]["retraining_days"] == 7:
        candidate = "weekly_refit"
    candidate_label = {
        "catboost": "CatBoost",
        "lightgbm": "LightGBM",
        "ridge": "Ridge",
    }[report["fits"][candidate]["family"]]
    if candidate == "weekly_refit":
        candidate_label += ", weekly refit"
    with connect() as db:
        rows = db.execute(
            f"SELECT delivery_start_utc, actual, previous_day, {candidate}, q10, q90 FROM read_parquet(?) ORDER BY delivery_start_utc",
            [str(report_path.with_name("predictions.parquet"))],
        ).fetchall()
    times = [row[0].astimezone(COPENHAGEN) for row in rows]
    data = np.asarray([row[1:] for row in rows], dtype=float)
    days = sorted({stamp.date() for stamp in times})
    daily = []
    for day in days:
        mask = np.asarray([stamp.date() == day for stamp in times])
        daily.append(np.abs(data[mask, 1:3] - data[mask, :1]).mean(axis=0))
    daily = np.asarray(daily)
    delta = daily[:, 1] - daily[:, 0]
    examples = [
        int(delta.argmin()),
        int(np.argsort(delta)[len(delta) // 2]),
        int(delta.argmax()),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    failures = []
    for ax, index in zip(axes, examples, strict=True):
        day = days[index]
        mask = np.asarray([stamp.date() == day for stamp in times])
        local = [stamp for stamp, keep in zip(times, mask, strict=True) if keep]
        ax.plot(local, data[mask, 0], color="#202936", label="Actual")
        ax.plot(local, data[mask, 1], color="#bb713b", label="Previous Danish day")
        ax.plot(local, data[mask, 2], color="#285f9e", label=candidate_label)
        ax.fill_between(
            local,
            np.minimum(data[mask, 3], data[mask, 4]),
            np.maximum(data[mask, 3], data[mask, 4]),
            color="#285f9e",
            alpha=0.12,
            label="LightGBM 10th–90th quantiles",
        )
        ax.set_title(
            f"{day} · daily MAE {daily[index, 1]:.2f} vs baseline {daily[index, 0]:.2f} EUR/MWh"
        )
        ax.set_ylabel("EUR/MWh")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=COPENHAGEN))
        ax.set_xlabel("Europe/Copenhagen delivery time")
        ax.set_xlim(local[0], local[-1])
        ax.grid(alpha=0.2)
        failures.append(
            {
                "delivery_date": str(day),
                "model_mae": float(daily[index, 1]),
                "baseline_mae": float(daily[index, 0]),
            }
        )
    axes[0].legend(ncol=4, fontsize=8)
    fig.savefig(output / "delivery_examples.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    for index, name, color in (
        (0, "Previous Danish day", "#bb713b"),
        (1, candidate_label, "#285f9e"),
    ):
        rolling = np.convolve(daily[:, index], np.ones(7) / 7, mode="valid")
        ax.plot(days[6:], rolling, label=name, color=color)
    ax.set_ylabel("7-day average daily MAE, EUR/MWh")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output / "rolling_error.png", dpi=160)
    plt.close(fig)
    raw, calibrated = report["quantiles"], report["calibrated_quantiles"]
    months = sorted(raw["month"])
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    locations = np.arange(len(months))
    ax.bar(
        locations - 0.18,
        [raw["month"][month]["coverage_80"] for month in months],
        width=0.36,
        label="Raw",
    )
    ax.bar(
        locations + 0.18,
        [calibrated["month"][month]["coverage_80"] for month in months],
        width=0.36,
        label="Prior-fold calibration",
    )
    ax.axhline(0.8, color="#aa3333", linestyle="--", label="Nominal 80%")
    ax.set_xticks(locations, months)
    ax.set_ylabel("Observed interval coverage")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(output / "interval_calibration.png", dpi=160)
    plt.close(fig)
    (output / "examples.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    return failures


def main():
    parser = argparse.ArgumentParser(description="Render saved research predictions.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    print(json.dumps(render(args.report, args.output), indent=2))


if __name__ == "__main__":
    main()
