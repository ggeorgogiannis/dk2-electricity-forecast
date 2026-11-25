# Implementation and first live-day audit

The checkout was inspected on 18 November 2026. No AGENTS.md was found in
the project or its immediate parent. The existing environment is Conda
Python 3.12.16 at `.venv\python.exe`. Its package check passed.

The starting implementation provided validated Energinet prices, raw response
caching, versioned Parquet, DuckDB queries, DST delivery grids, two baselines,
an expanding model comparison, a frozen prospective comparison, and FastAPI
serving for the original forecast schema. Existing development and diagnostic
prediction artifacts were retained.

The original diagnostic rerun reproduced 5,380 intervals and MAEs of
83.74164681 for the weekly UTC baseline, 64.71372868 for original LightGBM,
and 50.37110414 for the previous Danish day.

## Operational findings

The frozen 18 November and 19 November forecasts are absent. Their observed
coverage is zero, and their operational status is missed. The last recorded
Windows task attempt was at 11:41:27 on 18 November and returned 0x800710E0,
meaning the operator or administrator refused the request. Task history was
disabled, so this last result does not establish the original trigger failure's
cause. Enabling task history was denied by the account's privileges.

A separate runtime check launched the project interpreter through Task
Scheduler and completed with exit code 0. The temporary task was removed
after saving the verification record. The production tasks are enabled
for 19 November at 09:50 and 10:00, use the correct working directory and
interpreter, and now invoke a wrapper that captures output and exit codes.
The separate weather task is enabled for 09:40.

The missing first-block training receipt prevents eligible retrospective
recovery. The next original block starts with delivery 16 December and
preparation on 15 December. The original plan and its implementation hash
continue to validate.

## First-day replay

The actuals snapshot through 18 November contains 39,748 unique quarter-hours.
All five original raw-response hashes and their validated values matched
the Parquet snapshot. A replay of the frozen configurations was created at
12:09:22 UTC on 18 November, after the scheduled 09:00 UTC issue on 17 November.
It is stored under data/replays and excluded from frozen live scoring.

| Method | Intervals | Coverage | MAE EUR/MWh | Within €25 | Within €50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous Danish day | 96 | 100% | 50.97 | 22.9% | 45.8% |
| Weekly UTC | 96 | 100% | 140.78 | 0.0% | 3.1% |
| Elastic Net | 96 | 100% | 51.04 | 12.5% | 41.7% |
| LightGBM | 96 | 100% | 46.62 | 19.8% | 49.0% |
| CatBoost | 96 | 100% | 51.45 | 12.5% | 42.7% |

This is an integration check on one delivery day. It has no statistical
comparison claim. The replay assumes archived prices through the issue-day
delivery were available at the historical cutoff.

## Changes and remaining evidence

The milestone adds correction snapshots and retrieval metadata, archive
verification, operational logging and inspection, individual weather runs,
expanded price-curve research, stored searches, interval evaluation and
figures, explicit replay handling, atomic research publication, and API
support for experiment forecasts and evaluation status.

Historical price revision vintages remain unverifiable. The weather
historical experiment uses a declared availability delay assumption.
Live weather capture has started. ENTSO-E credentials were absent from
the inspected environment, and load, wind and solar ablations await eligible
versioned inputs. Neighbour and interconnector integrations remain subsequent
controlled experiments. Neural models and confirmatory forecast-comparison
tests remain later work.

There is no Git remote in this checkout. GitHub dependency alerts and
secret-scanning settings require repository access. Linux CI is configured
and remains unverified locally. The selected MIT code license is subject
to the user's release review.

The free Energinet Forecasts_Hour alternative was also audited. Its day-ahead
column is generated at 17:50 and published at 18:00 Danish time, so it is
unavailable for this project's 10:00 issue. Other forecast columns require
their own horizon and historical-version checks.

The final checks passed 33 tests in both the project Conda environment and a
clean Python environment. Ruff lint and formatting checks passed. Both test
runs emitted one Starlette/AnyIO deprecation warning. The clean environment
also completed the synthetic DST-spanning forecast demo.

The first and last candidate folds were independently recomputed from the
release code. All 1,056 predictions matched their saved values exactly.
The verification record is retained in reports/verification.json.

The new curve-development period had zero previous-day weekly fallbacks.
Weekday-aware persistence used four UTC-weekly fallbacks for missing local
source slots. The counts are retained in reports/fallbacks.json.

The separate research-candidate replay for 18 November scored 20.76 EUR/MWh
against the previous-day replay's 50.97, over the same 96 intervals. Its
accuracy bands were 66.7% within 25 EUR/MWh and 99.0% within 50 EUR/MWh.
It was created after the issue cutoff and remains excluded from live scores.
The normal FastAPI factory returned HTTP 200 for the stored actuals, replay,
health and evaluation endpoints.

## Operational follow-up verification, 18 November 2026

Verified defects were report-only evaluation identity, fixed scheduler bounds,
missing evaluation-level scoring/API discovery, mutable-checkout execution,
repeated block fits, unclear quantile-origin recovery, and incomplete cache
provenance validation. Fixes are documented in operations.md and the public
operational-followup report. The original eight-module implementation hash
continues to validate as 5b29d634768fbf69466e613aa8258a45a59f342af16de18db187fcc1f1177ec4.
No original freeze or receipt was rewritten.

Final checks passed **57 tests in each of three environments**:

- Project Conda `.venv`: 24.98 seconds.
- Newly installed `data/release-clean-env`: 24.72 seconds.
- Existing `.reproduction-env`: 25.25 seconds.

Each emitted one existing Starlette/AnyIO BlockingPortal deprecation warning.
Ruff lint and formatting passed (44 Python files). All three dependency checks
passed. The clean lock installation and editable install succeeded. Project and
fresh-clean environments ran the offline DST-spanning demo successfully.
The configured Windows/Linux cloud CI was inspected but not executed; no remote
or repository security configuration was created or changed.

Commands actually used included:

```cmd
.venv\python.exe -m pytest -q --tb=short
.venv\python.exe -m ruff check .
.venv\python.exe -m ruff format --check .
.venv\python.exe -m pip check
.venv\python.exe scripts\offline_demo.py
.venv\python.exe -m venv data\release-clean-env
data\release-clean-env\Scripts\python.exe -m pip install -r requirements-lock.txt
data\release-clean-env\Scripts\python.exe -m pip install -e . --no-build-isolation
data\release-clean-env\Scripts\python.exe -m pytest -q --tb=short
.reproduction-env\Scripts\python.exe -m pytest -q --tb=short
```

Regression coverage includes multiple evaluations of a retained candidate,
recovery after checkout edits, both clock changes, complete 100-slot issuance,
missing point/quantile origins, cutoff-completion failure, target/future-label
contamination, late/interrupted/retried publication, cached model origins,
corrupt/replayed/mismatched artifacts, missing actuals and missed forecasts,
prior-only gating, optional fundamental parsing and real saved-schema API scores.
The final source bundle's launcher completed inspection and initial scoring.
All 56 delivery days are not yet due, with zero issued or paired forecasts.
Future execution remains pending.

An initial setup evaluation was superseded before any issue because final
quantile-origin/cache checks changed the implementation version. Its plan and
inspection/score artifacts were retained; its three tasks are disabled. The
final active identity is in configs/active-evaluation.json. No prospective result
was used to select or change the model. The selected model/features/parameters
and point/quantile cadence remain unchanged.

Task registration and CheckOnly passed with the correct interpreter, directory,
account and floating Copenhagen wall-time bounds. Disabling a future task made
CheckOnly return 1; restoring it returned 0. Machine-wide execution policy and
scheduler-history settings were not changed. Original experiment and standalone
weather tasks remain separate. Runtime evidence is retained under data/reviews
and each evaluation's logs directory.

The final normal FastAPI factory smoke check returned 200 for health and saved
18 November actuals (96 intervals), 404 for the not-yet-issued 20 November
forecast, and 200 for /evaluation. The selected evaluation reports 5,376 expected
intervals, zero due or paired intervals, and only not_yet_due day states. Source
and original frozen-plan validation still passed after the final setup.

A final API regression verifies that a later coverage-only inspection does not
hide a comparison with an actual-price snapshot. Original-experiment scheduler
collection dispatch is also covered. The final 57-test suites passed in all three
environments after these compatibility changes. Active research execution keeps
using its retained bundle; later API/CLI changes do not alter its model or score
contract. All review patches were checked against the current working files.
