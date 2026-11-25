# Source coverage and information boundary

Research freezes the input snapshot by 09:50 Europe/Copenhagen and publishes
at 10:00. Run preparation at 09:40 to allow retrieval and validation. Both
timestamps are saved. Training labels must describe delivery days strictly
before the training origin, and every feature must satisfy its own input
availability condition. UTC identifies intervals throughout storage and scoring.

The original frozen comparison starts preparation at 09:50. Its input receipt
records actual verification time, which its existing rule permits through
10:00. It consequently uses a snapshot completed during preparation. Its
09:50 schedule alone provides no evidence of a 09:50 data cutoff.

| Input | Coverage and source resolution | Timing and revisions | Experiment eligibility |
| --- | --- | --- | --- |
| Energinet DayAheadPrices DK2 | From 1 October 2025, PT15M, EUR/MWh | Historical responses expose delivery time, with no first-publication timestamp or revision vintage. Corrections are captured with `--refresh` into immutable response objects and new manifests. | Historical prices for the issue-day delivery are assumed known by the cutoff. Live verification receipts prove the stored snapshot was observed before the cutoff. |
| Price curves and derived statistics | Same archive, local-clock curves with 96 feature positions | Repeated local slots use the first UTC occurrence. Spring gaps remain missing for preprocessing. The baseline uses weekly fallback. | Reconstructed exclusively from preceding delivery dates. |
| Open-Meteo ECMWF IFS 0.25° | Individual runs generally archived from 2 April 2026, hourly at a Copenhagen grid-point proxy | Initialization, availability and retrieval have separate meanings. Live snapshots preserve metadata and use a 10-minute availability buffer. A historical run can be absent. | Strict mode accepts captured live availability only. The exploratory historical ablation assumes an eight-hour delay for the preceding 18 UTC run. |
| DK2 load forecasts | ENTSO-E A65/A01, MW, dataset-dependent PT15M/PT60M | Publication deadlines and later revisions do not identify the version actually visible at 09:50. A free API token is required. | Requires token access and pre-cutoff version capture before a load ablation. |
| DK2 wind and solar forecasts | ENTSO-E A69, MW | The day-ahead submission deadline can follow the price auction. A day-ahead label alone does not establish eligibility. | Requires a demonstrably earlier version and live collection. |
| Energinet Forecasts_Hour wind and solar | Hourly, MWh per hour, updated every five minutes | Dataset metadata state that ForecastDayAhead is generated at 17:50 and published at 18:00 Danish time. Historical rows also contain intraday and short-horizon forecasts with separate timing. | The documented day-ahead series fails the 10:00 cutoff. Other columns require a target-horizon and version audit. |
| Neighbouring zone prices | Energinet/ENTSO-E, source-dependent resolution | Same-delivery-day auction prices arrive after our cutoff. Historical prices have the same vintage limitation as DK2. | Prior delivery-day curves are eligible under the stated price assumption. Integration is pending. |
| Interconnector capacity and outages | ENTSO-E document series, MW | Versioned documents can be revised. Document creation and distribution must both precede cutoff. | Integration requires token access and retained versions. |

The weather proxy represents one Copenhagen grid cell, not aggregate DK2
renewable production. Hourly instantaneous fields are held constant across
the four quarter-hours starting at that hour. The collector records source
valid time and source resolution. Radiation uses the instantaneous field,
avoiding a backward-hour average with a different alignment meaning.

The historical weather study is conditional on its declared timing assumption.
It cannot establish a verified live improvement. The collector records missing
runs as failures and evaluates weather additions on their common covered
intervals. Load and renewable ablations have no valid scores until eligible
versioned inputs are available.

Source descriptions and access terms are documented by
[Energinet](https://energidataservice.dk/guides/api-guides),
[Open-Meteo individual runs](https://open-meteo.com/en/docs/single-runs-api),
[Open-Meteo availability metadata](https://open-meteo.com/en/docs/model-updates),
[Open-Meteo terms](https://open-meteo.com/en/terms),
and the [ENTSO-E transparency guidance](https://transparencyplatform.zendesk.com/hc/en-us/articles/17260622859412-Transparency-Platform-Help-page).

The free Energinet alternative was checked directly through its
[Forecasts_Hour metadata](https://api.energidataservice.dk/meta/dataset/Forecasts_Hour).
Its generation time and publication time are distinct, and both occur after
this project's issue cutoff for the day-ahead column.

Open-Meteo's free API permits non-commercial use, and its data require
CC BY 4.0 attribution. The code's MIT license covers project code. Source data
retain their respective terms. Keep bulk raw responses out of Git.

## Optional fundamentals collection

`dk2_forecast.fundamentals` implements a strict optional live collector for
ENTSO-E A65/A01 DK2 total-load forecasts and A69/A01 DK2 wind/solar forecasts.
It requires ENTSOE_API_TOKEN with approved API access. No token was present in
this checkout, so no real fundamental capture or ablation is claimed.

The parser verifies DK2 domain 10YDK-2--------M, MW (MAW), forecast process A01,
production type, document creation/revision, UTC coverage and fixed source
resolution. PT15M/PT30M/PT60M values are held over their source period, preserving
source resolution. Variable-block curves and ambiguous/overlapping series fail
explicitly. Raw XML and retrieval observations are immutable; each revision
becomes another captured vintage. Model initialization and publication times
remain null when not supplied. Document creation is not treated as publication.
Successful retrieval before 09:50 is the eligibility evidence. There is no
historical-vintage guarantee and no realized-input substitution.

ENTSO-E's [load guidance](https://transparencyplatform.zendesk.com/hc/en-us/articles/16647979768084-Actual-Total-Load-Day-ahead-Per-Bidding-Zone-6-1-A-6-1-B)
sets publication relative to gate closure and requires significant updates.
That deadline does not establish visibility before our earlier snapshot cutoff.
The [wind/solar guidance](https://transparencyplatform.zendesk.com/hc/en-us/articles/16648445340180-Generation-Forecasts-for-Wind-and-Solar-14-1-D)
distinguishes current, 18:00 day-ahead and 08:00 intraday versions. Only a version
actually captured before our cutoff may enter a future experiment. An absent
next-day curve is a missing optional input, not permission to use actuals.

For an explicitly chosen future delivery date, before its cutoff:

```cmd
for /f "delims=" %I in ('.venv\python.exe -c "from dk2_forecast.evaluation import earliest_origin; print(earliest_origin())"') do @set "DK2_DELIVERY=%I"
.venv\python.exe -m dk2_forecast.fundamentals load %DK2_DELIVERY%
.venv\python.exe -m dk2_forecast.fundamentals renewables %DK2_DELIVERY%
```

The example computes a feasible future delivery date. Supply ENTSOE_API_TOKEN via
your local environment, without putting it in a committed file or command log.
Request URLs and token values are omitted from collector errors and receipts.
The collector is optional and is not wired into the price-only live candidate.
Before adding it, collect sufficient pre-cutoff vintages and freeze a separate
feature experiment. Neighbour-zone forecasts and capacity/outage documents need
the same geographic, horizon, revision and availability audit; no integration
or scores for those sources are claimed here.

Review [ENTSO-E's API guide](https://transparencyplatform.zendesk.com/hc/en-us/articles/15692855254548-Sitemap-for-Restful-API-Integration)
and [current reuse terms and eligible-data list](https://transparencyplatform.zendesk.com/hc/en-us/articles/40921911218961-Legal-Terms-and-Conditions)
before redistribution. Attribute the Transparency Platform and respect primary
owners' rights. Bulk captured data remain ignored. The Copenhagen weather-proxy
ablation is one limited experiment, not a conclusion about the usefulness of
weather or DK2 renewable forecasts generally.
