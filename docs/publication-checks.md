# Local publication verification — 25 November 2026

The project is prepared for review as a forecasting research repository. No
commit, remote creation, push or GitHub release was performed during preparation.
No Windows tasks were registered or enabled. Original freezes, receipts, model
artifacts and dated research results were preserved.

| Check | Actual local result |
|---|---|
| Existing project environment | Windows, Python 3.12.16; 75 pytest tests passed in 23.16 seconds |
| Clean public-only copy | New isolated environment; pinned dependencies installed, wheel built and installed successfully |
| Installed-package isolation | An isolated Python import resolved to the new environment's site-packages |
| Clean-copy test suite | 75 tests passed in 23.14 seconds, including publication-audit regressions |
| Ruff lint and formatting | Passed in both copies; 52 Python files formatted |
| Dependency consistency | pip check passed in both environments |
| Offline demo | Passed from the public-only copy without private captured data or API credentials |
| Candidate-file audit | Passed for local links, configured sensitive-file patterns, private runtime exclusions and retained source provenance |
| Retained implementation bundle | Archive/report/content identities verified; no frozen source bytes changed |

Both test runs emitted the same existing Starlette/AnyIO deprecation warning.
It does not fail the tests. The wheel was installed without editing the current
project's Conda environment; the new environment lives under ignored `data`.

The clean copy contains audited public files and an export manifest, not the
private data archive or trained model caches. Tests and the demo ran from that
copy. The code tests establish reproducibility of the offline pipeline; synthetic
demo outputs are not evidence of market accuracy.

## Initial GitHub publication

After the preparation checks, the owner authorized publication. The public
[repository](https://github.com/ggeorgogiannis/dk2-electricity-forecast) was
created and initial commit
[`9a79bf2`](https://github.com/ggeorgogiannis/dk2-electricity-forecast/commit/9a79bf255238cfbfd3b501084c47a4f33a230916)
was pushed on 25 November 2026. The
[first hosted workflow](https://github.com/ggeorgogiannis/dk2-electricity-forecast/actions/runs/45602150215)
completed successfully on both Windows and Linux. Each job includes dependency
installation/consistency, the publication audit, Ruff, pytest and the offline demo.
An explicit LF line-ending policy preserves hashed provenance across checkouts;
the exact staged public files and retained hashes were checked before committing.
No forecast tasks were enabled and no GitHub release was created.

The earlier no-commit/no-remote statements describe the preparation phase before
that authorization. No prospective accuracy evidence was created by publication.

## Checks that remain pending

- Repository-specific dependency alerts, secret scanning and security settings.
  A passing CI run does not establish these settings.
- A genuinely fresh prospective accuracy period and verified on-time live runs.
  Existing missed days and historical replays remain in their original categories.
- Reproduction of the exact market archive from a fresh public clone without
  the retained raw responses. Newly downloaded prices can differ after revisions.
- Verified historical fundamental forecast vintages and any trading profitability
  evaluation. Neither is established by the price-only replay.

See the [publication procedure](release.md) for copyable Windows CMD commands.
The ignored `data/publication-audit.json` contains the current public-file manifest
and hashes. The supplied publication review diff records this preparation step;
ordinary git diff omits untracked contents before the first commit.
