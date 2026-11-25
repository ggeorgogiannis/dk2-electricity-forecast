# Operational fixes and controlled research follow-up

This milestone preserves the original 18 November 2026–12 January 2027
comparison and all receipts. Its missed first block remains missed. The new
research evaluation has its own identity and does not rewrite that evidence.

## Active prospective evaluation

- Evaluation: f23d494fd3e14712135466641812f37c92fbbf58e0fa319bae24391336b7f95c.
- Candidate: 326385f66d1c7f75ecbdb67cd8dce96d7656033a366c1f1225120de18f6d5e62.
- Delivery dates: 20 November 2026–14 January 2027 inclusive (56 days, 5,376 slots).
- Expanding-history CatBoost, previous-day/calendar curves, seven-day point refit.
- Quantiles retain their 28-day refit and fixed last-development calibration margin.
- Primary benchmark: previous Danish day, earliest-UTC repeated slot, weekly-UTC gap fallback.
- Snapshot cutoff 09:50; scheduled preparation 09:40; issue minute 10:00–10:01 Copenhagen.
- First scheduled actions: 19 November at 09:40, 10:00 and 14:30 for collection/scoring.

These dates were calculated from the actual freeze time. All three tasks were
registered, enabled, and passed CheckOnly, with the project interpreter and
working directory. Triggers use floating local wall time; Windows has compatible
Copenhagen clock rules. A deliberate disabled-task test returned exit 1, followed
by restoration and a successful check. These are configuration checks: all
production executions remain pending, with no prospective accuracy claim.
The retained launcher ran inspection and initial aggregate scoring successfully.
No publication was manufactured. Scores currently contain not-yet-due days and
no paired forecasts. A copy of the exact plan is in configs/evaluations.

## Confirmed issues and fixes

Candidate and evaluation identities are now separate. Multiple future periods
can reference one candidate; restart reuses its retained implementation even
after checkout edits. Every operation selects freeze.json explicitly. Legacy
report-hash plans remain in place and are read through a documented compatibility
path. The original validation contract is unchanged.

Scheduling derives issue-date bounds from the selected plan, uses distinct task
names, supports actuals/scoring and separate optional weather capture, and detects
material mismatches. A verified source/dependency bundle protects live execution
from research edits. Actions retain stdout, stderr, exit status, timestamps,
failure reason and exclusive locks. Models are cached by frozen origin and
configuration. Complete preparation/publication directories commit atomically.

Aggregate scoring includes every scheduled delivery day and UTC interval. It
separates future, missed, invalid/late, eligible-awaiting-actuals and scored states.
Forecast coverage, actual availability, common paired denominators, operational
errors, model-only errors and fallback subsets are explicit. Replays are rejected.
The API discovers forecasts and aggregate scores from the selected evaluation,
validates provenance, and reuses the latest score's actual-price snapshot for
current state. Tests serve real saved research-schema Parquet, receipts and scores.

## Development failure comparison

All methods cover the same 16,512 intervals, 2 April–20 September 2026, with
seven-day point refitting. These are selection-affected development results.

| Method | Overall MAE | Negative MAE (350) | ≥€200 MAE (852) | Extreme MAE (166) |
| --- | ---: | ---: | ---: | ---: |
| Previous Danish day | 32.52 | 7.97 | 79.54 | 200.19 |
| Frozen CatBoost candidate | 28.11 | 37.54 | 86.14 | 228.18 |
| Fixed 50/50 candidate/baseline | 28.35 | 21.99 | 75.91 | 203.01 |
| Weekly residual correction | 28.62 | 24.84 | 86.40 | 218.42 |
| Prior-only gate | 27.89 | 23.50 | 77.90 | 204.66 |

MAEs are EUR/MWh. The frozen development extreme threshold is |price| ≥276.019989.
The gate uses only prior validation losses and available previous-day price
categories. Neither gate nor blend replaces the active candidate. Tail tradeoffs
remain material: persistence still leads negative and extreme slices. Candidate
bias is −2.06 EUR/MWh (prediction minus actual); the fixed blend's is −0.65.
The JSON includes month/hour/weekend/horizon/available-price groups and large
signed-error examples. Target regimes appear only in evaluation slices.

## Search accounting and chronological assessment

Original searches completed one CatBoost and four LightGBM trials while five
startup trials were configured: they provided no adaptive TPE evidence.
The separate bounded reference run completed six trials per family: three random
startup plus three adaptive TPE, zero failed or pruned. The ceiling was six trials
and 900 seconds per model; actual search time was 99.2 seconds LightGBM and
318.8 seconds CatBoost, excluding untuned fits and later assessment.

| Model | Inner untuned | Inner selected | Later untuned | Later selected |
| --- | ---: | ---: | ---: | ---: |
| LightGBM | 30.52 | 30.47 | 28.42 | 29.27 |
| CatBoost | 29.67 | 29.07 | 28.70 | 28.91 |
| Previous Danish day | 31.06 | same | 33.91 | same |
| Weekly UTC | 39.79 | same | 42.05 | same |

Inner contains 8,064 intervals; later contains 8,448.
Baseline references are calculated independently from the saved development
predictions on these exact dates. Inner data end before 25 June; the later block covers 25 June–20 September.
Both use expanding history and 28-day refitting. This differs from the active
candidate's seven-day schedule and is not a fresh test. Parameters are chosen
only on inner MAE. Both selected models worsened later MAE; no live configuration
was changed. Median trial times were 14.2 seconds LightGBM and 58.6 seconds
CatBoost. Twelve trials at observed maximum time plus 25% suggest roughly
367/1,117 seconds; the configured 2,700-second research budget leaves headroom.
Larger tree configurations or machines may require more. The smoke profile is
two startup-only trials; substantive defaults plan twelve with three startup.
Studies, sampling phases, parameters, seeds, folds, runtimes and sampler state
are retained. Six samples are limited evidence, not thorough optimization.

## Probabilistic follow-up

All intervals use the same 16,512 targets. Lower interval score is better.

| 80% interval method | Coverage | Mean width | Interval score |
| --- | ---: | ---: | ---: |
| Raw sorted model quantiles | 60.8% | 69.65 | 158.58 |
| Existing prior-fold calibration | 79.5% | 97.02 | 154.59 |
| Bounded 28-day rolling calibration | 76.5% | 92.25 | 150.23 |
| Prior 56-day persistence residuals | 79.2% | 109.11 | 189.48 |

Raw model quantiles crossed on 693 intervals; raw values, pinball loss and sorted
operational values remain distinct. Rolling calibration updates weekly using
only prior OOF residual scores, with a 14-day minimum and raw fallback. Its lower
score trades against undercoverage. None has a formal drift-robust coverage
guarantee. Monthly/hour/regime counts and widths appear in the JSON. The active
plan retains the existing policy, not the new rolling alternative.

## Verification and remaining evidence

The final pytest totals and commands are recorded in docs/audit.md. Ruff lint,
format checks, dependency checks, clean pinned installation and offline demos
were exercised on Windows. The configured Linux/GitHub workflow was not executed;
there is no remote. No commit, push, repository or release was created.

An optional ENTSO-E collector validates live DK2 load/wind/solar XML vintages,
units, domain, resolution, revision and cutoff, retaining raw evidence. Credentials
were absent, so no live capture or feature improvement is claimed. Variable-block
curves require explicit later support. Neighbour/interconnector integrations and
seasonal fresh results remain pending. Historical price revisions and the earlier
weather availability assumption remain limitations. The Copenhagen proxy ablation
does not establish that weather or renewable forecasts cannot help.

See [operational commands](../docs/operations.md), [research commands](../docs/research.md)
and [machine-readable follow-up](operational-followup.json). Detailed generated
predictions and source bundles remain under ignored data with hashes recorded
in the JSON; existing benchmark reports are preserved.

The first setup plan was retained and superseded before any scheduled issue.
Final quantile-origin recovery and cache-identity checks changed only the
implementation version, so a final freeze was created and its tasks registered;
the preliminary plan's tasks were disabled. Neither plan has issued forecasts.
configs/active-evaluation.json identifies the final selected plan explicitly.
When a quantile origin is missed, a seven-day point origin alone is insufficient:
inspection reports the next shared point/quantile origin, or a new future
evaluation can be created with restart. Existing plans are never edited.


## Calibration and error slices

All counts below refer to the same development intervals; coverage percentages
are observed 80% interval coverage, not guarantees.

| Month | Intervals | Raw | Existing calibration | Rolling 28-day | Historical residuals |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-04 | 2784 | 53.9% | 55.1% | 64.7% | 79.0% |
| 2026-05 | 2976 | 64.1% | 95.4% | 89.8% | 91.3% |
| 2026-06 | 2880 | 61.6% | 84.4% | 79.0% | 72.7% |
| 2026-07 | 2976 | 68.8% | 89.5% | 81.4% | 81.7% |
| 2026-08 | 2976 | 63.7% | 82.5% | 74.9% | 80.6% |
| 2026-09 | 1920 | 47.3% | 63.0% | 64.4% | 64.6% |

| Regime | Intervals | Raw coverage | Existing | Rolling | Residual benchmark |
| --- | ---: | ---: | ---: | ---: | ---: |
| high | 852 | 36.2% | 45.3% | 43.8% | 55.6% |
| negative | 350 | 22.3% | 64.3% | 55.7% | 98.9% |
| regular | 15310 | 63.0% | 81.8% | 78.8% | 80.1% |

| Local hour | Intervals | Existing coverage | Rolling coverage |
| --- | ---: | ---: | ---: |
| 0:00 | 688 | 88.2% | 84.7% |
| 1:00 | 688 | 88.4% | 83.7% |
| 2:00 | 688 | 87.4% | 85.6% |
| 3:00 | 688 | 87.9% | 84.4% |
| 4:00 | 688 | 85.3% | 82.4% |
| 5:00 | 688 | 84.7% | 81.4% |
| 6:00 | 688 | 79.5% | 76.6% |
| 7:00 | 688 | 79.5% | 77.5% |
| 8:00 | 688 | 78.5% | 75.6% |
| 9:00 | 688 | 78.9% | 72.1% |
| 10:00 | 688 | 76.3% | 71.4% |
| 11:00 | 688 | 76.7% | 72.4% |
| 12:00 | 688 | 78.6% | 77.0% |
| 13:00 | 688 | 79.9% | 81.2% |
| 14:00 | 688 | 79.5% | 79.7% |
| 15:00 | 688 | 76.6% | 74.4% |
| 16:00 | 688 | 73.8% | 68.6% |
| 17:00 | 688 | 73.1% | 69.2% |
| 18:00 | 688 | 79.1% | 75.3% |
| 19:00 | 688 | 75.1% | 72.1% |
| 20:00 | 688 | 70.6% | 69.3% |
| 21:00 | 688 | 74.4% | 70.9% |
| 22:00 | 688 | 77.6% | 75.4% |
| 23:00 | 688 | 78.9% | 75.3% |

| Month | Intervals | Previous-day MAE | Candidate MAE | Candidate signed error |
| --- | ---: | ---: | ---: | ---: |
| 2026-04 | 2784 | 35.03 | 32.73 | 7.40 |
| 2026-05 | 2976 | 22.58 | 21.31 | -5.65 |
| 2026-06 | 2880 | 38.75 | 30.61 | -6.48 |
| 2026-07 | 2976 | 28.83 | 23.17 | 2.26 |
| 2026-08 | 2976 | 29.48 | 23.93 | -5.10 |
| 2026-09 | 1920 | 45.36 | 42.31 | -5.60 |

| Lead-time bin in hours | Intervals | Previous-day MAE | Candidate MAE | Fixed blend MAE |
| --- | ---: | ---: | ---: | ---: |
| 12–<18 | 2752 | 21.66 | 14.38 | 16.86 |
| 18–<24 | 4128 | 30.63 | 23.62 | 25.30 |
| 24–<30 | 4128 | 31.62 | 30.36 | 29.01 |
| 30–<36 | 4128 | 44.02 | 39.71 | 39.05 |
| 36–<42 | 1376 | 28.09 | 27.45 | 26.45 |

Lead time is delivery UTC start minus the scheduled issue UTC timestamp.
The first bin begins at the actual 14-hour horizon. The JSON retains signed
errors, sample counts and available/predicted price bins, plus weekday/weekend
and local-hour point metrics. Price-bin codes are 0: below zero, 1: [0,100),
2: [100,200), 3: at least 200 EUR/MWh.

