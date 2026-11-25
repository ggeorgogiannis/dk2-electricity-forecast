# Experiment index

All errors are EUR/MWh. The primary benchmark is previous Danish delivery date
at the same local clock time, with the documented DST fallback. Dates, feature
sets, refit schedules and evidence categories must stay attached to each result.

| Report | Evaluated dates | Features and schedule | Evidence |
|---|---|---|---|
| [Original research](research.md) | 2 Apr–20 Sep 2026 | Price/calendar; selected CatBoost weekly refits starting on development fold origins | Development used for selection; €28.11 versus €32.52 benchmark MAE |
| [Operational follow-up](operational-followup.md) | Dates specified per analysis | Tail, interval and nested-search comparisons; operational validation | Development/diagnostic findings and offline integration evidence |
| [Annual replay](year-to-date-2026-11-25.md) | 1 Jan–25 Nov 2026 | Price/calendar; retained CatBoost, Friday-anchored seven-day refits | 329 days, 31,584 intervals; €30.00 versus €35.80; already observed replay |
| [Two-month challengers](two-month-2026-11-25.md) | 26 Sep–25 Nov 2026 | Same saved CatBoost; residual Ridge, fixed blend, weekly gate/calibration | 61 days, 5,860 intervals; Ridge €37.29 versus persistence €47.01; already observed replay |

The original 21 September–15 November diagnostic period was already inspected.
Its reported 5,380 paired intervals and earlier model scores are reproduction
checks, not a new final test. The [reproduction reference](../docs/reproduction.md)
retains those details. Methods and date ranges differ across reports, so changes
between their MAEs must not be presented as improvements on a common test set.

The public research summaries and figures are dated artifacts. JSON companions
provide machine-readable metrics and provenance references. Bulk raw prices,
model caches, replay receipts and interval exports remain under ignored `data`;
their absence from Git is intentional. Retained configuration/source copies in
`configs` preserve frozen implementation identities.

No eligible prospective scores are included. Missed forecasts remain missed,
and historical replays retain their actual creation times and ineligibility.
The latest challenger has not replaced the frozen candidate. Full-year averages
do not resolve the documented negative-price, extreme-price and November failures.
