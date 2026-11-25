"""Resumable, bounded nested chronological research; never changes a live plan."""

import argparse
from datetime import UTC, date, datetime
from pathlib import Path
import os

from dk2_forecast.delivery_day import COPENHAGEN
from dk2_forecast.experiment import json_bytes
from dk2_forecast.forecast import PriceHistory
from dk2_forecast.provenance import save_code_snapshot
from dk2_forecast.research import ResearchData, tune_tree
from dk2_forecast.storage import digest, publish_bytes

INNER_START = date(2026, 4, 2)
INNER_END = date(2026, 6, 25)
OUTER_END = date(2026, 9, 21)
PROFILES = {
    "smoke": {"trials": 2, "startup": 3, "timeout": 120},
    "research": {"trials": 12, "startup": 3, "timeout": 2700},
}


def live_guard():
    local = datetime.now(UTC).astimezone(COPENHAGEN)
    if 8 <= local.hour < 11:
        raise RuntimeError(
            "Research paused: 08:00â€“11:00 Copenhagen is reserved for live preparation/publication. Resume later."
        )


class InnerData:
    def __init__(self, data):
        self.data = data

    def evaluate(self, family, params, **kwargs):
        return self.data.evaluate(
            family, params, start=INNER_START, end=INNER_END, **kwargs
        )


def run(
    prices,
    root,
    *,
    profile="research",
    trials=None,
    timeout=None,
    families=("lightgbm", "catboost"),
):
    live_guard()
    if os.name == "nt":
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise OSError(
                ctypes.get_last_error(), "Could not lower research process priority"
            )
    settings = PROFILES[profile]
    trials = trials or settings["trials"]
    timeout = timeout or settings["timeout"]
    if not 1 <= trials <= 30 or not 0 < timeout <= 14400:
        raise ValueError(
            "Use 1..30 sequential trials and a timeout up to four hours per tree"
        )
    root.mkdir(parents=True, exist_ok=True)
    history = PriceHistory.load(prices)
    data = ResearchData(history, root.parents[0])
    inner = InnerData(data)
    report = {
        "status": "nested_development_research_not_fresh_evidence",
        "profile": profile,
        "code_snapshot_sha256": save_code_snapshot(root.parent),
        "data_version": history.data_version,
        "inner": {"start": str(INNER_START), "end_exclusive": str(INNER_END)},
        "outer": {"start": str(INNER_END), "end_exclusive": str(OUTER_END)},
        "retraining_days": 28,
        "threads": 1,
        "seed": 42,
        "search_space_version": 2,
        "objective": "interval-weighted inner OOF MAE EUR/MWh",
        "models": {},
    }
    for family in families:
        params = (
            {}
            if family == "lightgbm"
            else {"depth": 4, "learning_rate": 0.07, "l2_leaf_reg": 3}
        )
        _, untuned = inner.evaluate(family, params)
        median = sorted(f["fit_seconds"] for f in untuned["folds"])[
            len(untuned["folds"]) // 2
        ] * len(untuned["folds"])
        identity = digest(
            json_bytes(
                {
                    "contract": {
                        k: v
                        for k, v in report.items()
                        if k not in ("models", "profile")
                    },
                    "family": family,
                    "startup": settings["startup"],
                }
            )
        )
        best, study = tune_tree(
            inner,
            family,
            root,
            identity,
            untuned,
            None,
            trials,
            timeout,
            startup_trials=settings["startup"],
            guard=live_guard,
        )
        live_guard()
        _, outer_default = data.evaluate(family, params, start=INNER_END, end=OUTER_END)
        _, outer_selected = data.evaluate(
            family, best["parameters"], start=INNER_END, end=OUTER_END
        )
        study["estimated_seconds_for_planned_trials"] = median * trials * 1.2
        study["estimated_seconds_to_first_adaptive_trial"] = (
            median * (settings["startup"] + 1) * 1.2
        )
        report["models"][family] = {
            "untuned_inner": untuned,
            "selected_inner": best,
            "untuned_outer": outer_default,
            "selected_outer": outer_selected,
            "search": study,
        }
        # Save progress after each family, preserving any prior report.
        raw = json_bytes(report)
        publish_bytes(root / "reports" / f"{digest(raw)}.json", raw)
    raw = json_bytes(report)
    path = root / "reports" / f"{digest(raw)}.json"
    publish_bytes(path, raw)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prices", type=Path)
    p.add_argument("--root", type=Path, default=Path("data/search-v2"))
    p.add_argument("--profile", choices=PROFILES, default="smoke")
    p.add_argument("--trials", type=int)
    p.add_argument("--timeout", type=float)
    p.add_argument("--family", choices=("lightgbm", "catboost"))
    args = p.parse_args()
    print(
        run(
            args.prices,
            args.root,
            profile=args.profile,
            trials=args.trials,
            timeout=args.timeout,
            families=(args.family,) if args.family else ("lightgbm", "catboost"),
        )
    )


if __name__ == "__main__":
    main()
