# Research design

The new price-curve experiment uses 2 April through 20 September 2026 as
development. That start provides 26 weeks of prices for every window candidate.
It compares expanding history with 8, 12 and 26 weeks at identical outer
origins. The primary window comparison uses untuned LightGBM and 28-day
blocks, then applies its chosen history window to every family. Weekly refitting
is subsequently compared for the selected candidate. The model and schedule
selection use development scores, which are optimistic estimates after search.
The previously inspected September–November period remains diagnostic.

All point methods use the same available price and calendar information.
The expanded matrix includes local quarter-hour cyclic terms, weekday
indicators, annual cyclic terms, UTC offset, previous-day and UTC weekly
point lags, both full local price curves, and the preceding day's mean,
standard deviation, minimum and maximum. An autumn repeated slot uses its
earlier occurrence. Spring missing curve values are passed as missing.
The regularized model fits median imputation and scaling within each fold.

The expanded autoregressive benchmark is a pooled quarter-hour Ridge model with
daily curve regressors. The reference LEAR implementation is an hourly,
multi-output LASSO ARX model with its own scaling and feature construction.
The implementation here borrows the full-curve design and regularization
principle. It does not reproduce LEAR's hourly model or claim its published
benchmark performance. The earlier 11-input Elastic Net experiment is retained.
Ridge was chosen for the correlated full-curve matrix after an Elastic Net
reference run took over eight CPU minutes before producing its first study.
[LEAR implementation](https://github.com/jeslago/epftoolbox/blob/master/epftoolbox/models/_lear.py)

The correction model trains on actual price minus the previous-day baseline
and adds the predicted correction back at inference. The fixed ensemble
averages Ridge and CatBoost. Adaptive weights are selected from
0, .25, .5, .75 and 1 using only earlier out-of-fold predictions. Its first
block uses equal weights. No current block's labels set its own weight.

Tree searches use sequential seeded TPE and bounded threads. A local SQLite
study stores completed and failed trials, parameters, fold results and runtime.
A local sampler checkpoint preserves TPE RNG state across sequential resumes.
Changing dataset, features or window creates a distinct study identity.
Change the study contract version when modifying the estimator or search
space. Each report separately records the code hash used for that run.
Only load local generated checkpoint files. The timeout is checked between
trials, and a final trial may finish beyond the budget.

The point report includes interval-weighted MAE, RMSE, absolute-error
percentiles, signed error, skill against the previous-day baseline, accuracy
bands, slice counts, and local calendar groups. Signed error is prediction
minus actual. Daily summaries give each Danish day equal weight.

Paired uncertainty uses 1,000 circular moving-block resamples of daily loss
differences, block length seven days and seed 42. The intervals are exploratory
and unadjusted for model selection or multiple comparisons. A model search on
the same development days invalidates a confirmatory significance claim.
Serial dependence and changing seasonal error distributions also limit the
bootstrap. Diebold–Mariano and Giacomini–White tests are deferred until enough
untouched daily losses support their stationarity and estimation assumptions.
These tests would use daily losses, including 92- and 100-interval days, and
require a declared multiplicity policy.

The Giacomini–White paper focuses on limited-memory estimation procedures.
Applying its test unchanged to the selected expanding-history procedure
would require further justification.
[Giacomini and White, 2006](https://economia.uc3m.es/jgonzalo/teaching/PhdTimeSeries/GiacominiWhite.pdf)
and [Diebold and Mariano, 1995](https://users.ssc.wisc.edu/~behansen/718/DieboldMariano1995.pdf)
provide the reference testing frameworks.

Quantile LightGBM models estimate .1, .5 and .9 with the chosen development
tree parameters and the common 28-day schedule. The research live plan
retains that schedule for all three quantile models, including when the
point-model ablation selects weekly refitting.
The research live interval uses the last prior-development calibration
margin as a fixed expansion. Raw bounds, the expansion and its source fold
are retained with each forecast. Its performance under future drift is
unverified.
The report retains raw crossing counts and sorts quantiles
for the operational range. Calibration widens prior-fold intervals using
the 80th percentile of earlier interval nonconformity scores, starting after
one complete block. The chronological implementation excludes current and
future labels. Time-series dependence means a finite-sample exchangeability
guarantee does not apply. Calibration is evaluated by month, local hour and
price regime.

The existing frozen CatBoost comparison remains the operational candidate.
A research winner becomes a proposal for another prospective evaluation.
Neural models remain a later experiment because the present sample and
unverified external-input vintages favour a smaller auditable comparison.

## Operational and research follow-up

The April–September curve experiment remains development evidence affected by
window/model/refit selection. The September–November diagnostic was already
observed. The 18 November replays remain historical integration checks. None
becomes fresh evidence through a new report or nested split.

The new controlled development analysis compares the fixed weekly CatBoost
candidate, previous-day benchmark, a 50/50 blend, weekly residual correction,
and a prior-only gate on identical intervals. Gate inputs use available
previous-day price categories; weights minimize MAE over the prior 56 delivery
days' validation predictions, update weekly, require 192 prior category slots,
and otherwise default to weight 0.5. Realized target regimes are used only for
reporting, never routing. All preprocessing remains train-only. Candidate
hyperparameters were previously selected on this development period, so the
new gate's OOF losses still have selection bias. It does not replace the live
candidate. Detailed groups include month, local hour, weekday/weekend, horizon,
predicted price and available previous-day price, signed errors and worst cases.

The interval follow-up adds the 80% interval score (width plus ten times each
outside miss), raw/operational pinball loss and sample-counted calibration
slices. The historical-residual benchmark uses preceding 56-day previous-day
errors, updates weekly and adds their 10/50/90% quantiles to persistence.
A separate bounded calibration uses preceding 28 days of OOF residual scores,
updates weekly, requires 14 prior days, and otherwise uses raw sorted quantiles.
Both policies are research-only. Their aggregate coverage is not a guarantee
under serial dependence or drift. The live candidate retains its fixed last
development margin and 28-day quantile model cadence.

```cmd
.venv\python.exe -m dk2_forecast.failure_analysis "%DK2_REPORT%" "%DK2_PRICES%"
.venv\python.exe -m dk2_forecast.search "%DK2_PRICES%" --profile smoke
.venv\python.exe -m dk2_forecast.search "%DK2_PRICES%" --profile research
```

The smoke profile caps at two trials/120 seconds per model and is explicitly
startup-only. The research profile plans twelve trials, three random startup
samples and up to nine adaptive TPE trials, with 2,700 seconds per model. Use
`--trials 6 --timeout 900` for the bounded reference run recorded in this
milestone, or `--trials 30 --timeout 7200` for a longer approved local run.
The actual report separates startup, adaptive, failed and pruned trials, planned
and completed budgets, selected parameters, elapsed time and runtime-based
estimates. A trial may finish beyond the per-family budget. No small search is
described as thorough optimization. Original studies completed one CatBoost and
four LightGBM samples against five startup trials: none was adaptive TPE.

New tuning uses inner origins from 2 April through 24 June and later outer
assessment from 25 June through 20 September, with the same expanding-history
28-day schedule and CPU thread limit for both families. Outer scores are not
used to choose inner parameters. This outer period remains previously observed
development data, not an untouched test. Study identity includes dataset,
fold contract, feature/source bundle, objective/search-space version and seed.
Changes cannot silently reuse incompatible studies. SQLite and a local sampler
checkpoint allow sequential resumes; retain both. Never load foreign pickle
state. Research runs below normal Windows priority, uses one model CPU thread,
and refuses to start trials between 08:00 and 11:00 Copenhagen. Schedule long
runs outside the live window; the next trial is checked, but an already-running
trial can finish later. Normal live tasks run from an isolated retained source
bundle, with their own locks and origin model caches.


The bounded reference run completed three startup and three adaptive trials
per model (six total, zero failed/pruned). Measured median trial times were
14.2 seconds LightGBM and 58.6 seconds CatBoost; observed maximums were 24.5 and
74.5 seconds. Twelve trials at those maximums plus 25% imply about 367 and
1,117 seconds. The 2,700-second substantive default provides headroom, but remains
a soft budget between trials and may be insufficient on slower hardware or a
changed search space. Later assessment worsened for both tuned models; see the
[follow-up](../reports/operational-followup.md). Exact selected parameters and
folds are retained in its JSON. The active prospective candidate stays unchanged.

## Two-month price-only challenger comparison

The [26 September–25 November comparison](../reports/two-month-2026-11-25.md)
contains 61 whole Danish days and 5,860 paired intervals. Original CatBoost and
both baselines are verified against the existing annual replay and price archive.
The new driver is `scripts/compare_two_months.py`; Windows CMD commands are in
the README. Its policy and source hashes are saved before fitting and scoring.

The simple challengers are a fixed half blend, residual Ridge with the common
210 price/calendar inputs, and a prior-only gate. The three Ridge alpha values
are selected on weekly chronological folds from 31 July through 24 September,
before the first 25 September training origin used in the outer period. Every
pipeline fits imputation/scaling on its own earlier training rows. Expanding
fits and all gate/calibration updates exclude origin-day labels. Existing
candidate parameters remain selection-affected, even though the new residual
hyperparameter selection is chronological.

Weekly gates use previous-day price regimes, never actual target regimes, and
the preceding 56 days of out-of-fold errors. During the outer period, earlier
outer outcomes enter subsequent weekly updates according to the saved policy.
This is a sequential policy evaluation, not a gate fitted once on the full test.
LightGBM raw quantiles use fixed parameters and a 28-day refit cycle; rolling
28-day calibration and historical 56-day residual intervals use only prior errors.
Raw quantiles, sorted operational bounds, pinball loss, crossings, interval score,
coverage and width are retained with monthly/hour/regime breakdowns.

Residual Ridge lowers overall MAE to EUR 37.29/MWh from CatBoost 41.71 and the
previous-day benchmark 47.01. It remains biased downward and loses to persistence
on extreme prices. The half blend has lower high-price errors. Zero negative
actual-price intervals occur, so this window cannot establish negative-price
improvement. These findings do not change the frozen live candidate or establish
untouched test skill, on-time reliability or profitability.

Verified historical fundamental forecast vintages and a specified executable
trading decision/cost model are still required for those extensions. The replay
does not substitute realized target-day load, generation or weather. All runs
remain manual; no scheduled tasks are enabled by either command.
