# Model card

## Intended use

Research and analyst review of DK2 day-ahead price curves in EUR/MWh. Forecast
issue is 10:00 Europe/Copenhagen on D−1, covering the full next Danish delivery
day. The research workflow declares a 09:50 input cutoff. UTC uniquely identifies
each 15-minute interval, including 92/100-slot clock-change days.

No profitability, autonomous trading suitability or prospective accuracy is
established. No order-execution integration is implemented.

## Frozen research candidate

Expanding-history CatBoost, 160 iterations, MAE objective, seed 42, CPU with one
training thread. Point models refit every seven delivery days. Inputs are 210
price-history/calendar features, including previous-day and weekly curves.
Exact parameters, dependencies and retained source identity are in
[candidate configuration](../configs/research-candidate.json) and the
[evaluation plan](../configs/evaluations/f23d494fd3e14712135466641812f37c92fbbf58e0fa319bae24391336b7f95c/freeze.json).

The reference benchmark matches the previous Danish date and local clock slot,
chooses the earlier UTC occurrence on ambiguous days and falls back to the
168-hour UTC lag for missing slots. Fallbacks are counted.

## Experimental challenger

Residual Ridge predicts a correction to the previous-day baseline from the
same 210 features. Imputation and scaling fit inside each expanding training
fold. Alpha 1000 was selected from [100, 1000, 10000] on chronological weekly
validation from 31 July through 24 September 2026. Seven-day outer refits are
anchored to 20 November. This challenger has not replaced the frozen candidate.

The [two-month report](../reports/two-month-2026-11-25.md) describes fixed blends,
prior-only gates and interval comparisons. Gating uses available previous-day
prices, never realized target regimes. Earlier outer outcomes can enter later
weekly updates under the predeclared sequential policy.

## Evidence and limitations

- Selected development result: CatBoost MAE approximately €28.11 versus €32.52
  for persistence on 2 April–20 September 2026. Model selection affects this result.
- Annual replay: 1 January–25 November 2026, 31,584 intervals; CatBoost €30.00
  versus persistence €35.80. The Friday-anchored refit schedule differs from
  the original development folds, and the dates were already examined.
- Two-month comparison: 26 September–25 November, 5,860 intervals; Residual Ridge
  €37.29, CatBoost €41.71, persistence €47.01. It remains a previously observed
  retrospective comparison, not an independent final test.
- In that window, extreme-price MAE is €77.92 for Residual Ridge versus €63.28
  for persistence. Residual Ridge's average signed error is −€15.73/MWh.
  There are no negative-price targets in the window. The annual CatBoost
  negative-price weakness remains material and unresolved.
- Historical price publication/revision vintages are unknown. Archived issue-day
  delivery prices are assumed available before cutoff. Real live use requires
  saved pre-cutoff input observations and on-time publication receipts.
- No eligible prospective performance is presented. Missed runs and replays are
  not converted into prospective evidence. Operation is manual by default.

## Uncertainty and optional inputs

Raw LightGBM quantiles, sorted operational intervals, historical-residual
benchmarks and prior-only calibration are evaluated separately from point models.
The two-month rolling calibration gives 78.1% coverage for nominal 80% intervals,
with wider bounds. Drift and temporal dependence preclude a formal coverage claim.

Load/wind/solar inputs are not part of these price-only results. Optional
collectors require eligible forecast vintages and sometimes credentials. The
limited Copenhagen weather-proxy ablation cannot establish that broader weather
or renewable forecasts are unhelpful. See [availability](availability.md).
