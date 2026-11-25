# Publication preparation

The project is prepared as a reproducible forecasting research repository.
Historical accuracy, operational checks and prospective evidence must remain
distinct. The README and model card disclose selection effects, missing historical
vintages and stress-regime weaknesses. The MIT license applies to code; source
data retain their provider terms. See [availability](availability.md).

## Review the candidate files

From regular Windows CMD in the project folder, using the local Conda interpreter:

```cmd
.venv\python.exe scripts\check_publication.py
git status --short --untracked-files=all
.venv\python.exe -m pytest -q
.venv\python.exe -m ruff check .
.venv\python.exe -m ruff format --check .
.venv\python.exe -m pip check
.venv\python.exe scripts\offline_demo.py
```

The audit reads tracked and nonignored untracked paths without staging or
committing. It checks local Markdown file links, common credential/private-key
patterns, personal Windows paths, oversized files and runtime artifacts. Findings
contain locations and categories, never matched credentials. It also inspects
the retained source ZIP members and verifies report/archive/content provenance
hashes. A pattern scan cannot guarantee that no secret exists. External links,
GitHub settings and private local ignored files are outside its scope.

`data/publication-audit.json` records the exact candidate file hashes and findings.
Raw prices, model caches, local environments, receipts, logs and search databases
are excluded. The one intentional public ZIP contains frozen implementation code
and dependency/configuration files, not market captures or trained models. The
matching exact research report and candidate/plan copies live under `configs`.
Never edit these immutable provenance copies to remove an inconvenient result.

On an unborn branch, ordinary `git diff` omits untracked file contents. Review
the supplied milestone patch and public manifest as well as `git status`.
Review publication-specific changes before the first commit. No commit, push,
remote repository or GitHub release is created by these commands.

## Verify a clean public-only copy

```cmd
.venv\python.exe scripts\check_publication.py --export
```

This copies only audited files into the printed directory
`data/publication/<manifest-sha256>/source`. It adds an export-only hidden manifest
so the copy can validate its own files without a Git database. It does not copy
ignored local data, create a remote or publish anything. Its files can be checked
independently; the manifest is not a substitute for reviewing file contents.

For an isolated install, the current checkout's `.venv` remains untouched:

```cmd
.venv\python.exe -m venv data\publication-check-env
data\publication-check-env\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Set the directory printed by `--export`, then install from that copy:

```cmd
set "DK2_PUBLIC=PATH_PRINTED_BY_EXPORT"
data\publication-check-env\Scripts\python.exe -m pip install "%DK2_PUBLIC%" --no-deps --no-build-isolation
data\publication-check-env\Scripts\python.exe -m pip check
data\publication-check-env\Scripts\python.exe "%DK2_PUBLIC%\scripts\offline_demo.py" --root data\publication-clean-demo
data\publication-check-env\Scripts\python.exe -m pytest "%DK2_PUBLIC%\tests" -q
data\publication-check-env\Scripts\python.exe -m ruff check "%DK2_PUBLIC%"
data\publication-check-env\Scripts\python.exe -m ruff format --check "%DK2_PUBLIC%"
data\publication-check-env\Scripts\python.exe "%DK2_PUBLIC%\scripts\check_publication.py"
```

The demo generates synthetic data, not a market-accuracy result. Its first run
must work without private captures or a frozen runtime directory. Exact market
results require the original captured price snapshot; new downloads may differ.
The [reproduction reference](reproduction.md) explains the retained-source annual
replay and two-month prerequisite. A public copy alone cannot recreate an on-time
receipt, a missed forecast or unknown historical source vintages.

## CI and remaining publication steps

The [workflow](../.github/workflows/ci.yml) configures Python 3.12 on Windows and
Linux, dependency checks, publication audit, Ruff, pytest and the offline demo.
Actions are pinned to verified upstream release commits:
[checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1) and
[setup-python v7.1.0](https://github.com/actions/setup-python/releases/tag/v7.1.0).
Jobs have read-only repository permissions and a bounded timeout. They run on
push/pull request; there is no daily workflow or deployment job. Installing or
serving this project does not register Windows tasks. Scheduling remains an
explicit optional operation; existing tasks were disabled at the owner's request.

Hosted CI, repository dependency alerts and GitHub-specific security settings
cannot be verified before a remote exists and the workflow runs. After approval
to publish, review the actual initial commit file list, configure repository
security settings and require successful hosted checks before describing either
platform as verified. No badges claim checks that have not run.

Suggested repository description:

> Reproducible DK2 electricity-price forecasting with chronological backtests,
> persistence benchmarks and uncertainty evaluation.

Suggested topics: `electricity-price-forecasting`, `energy`, `time-series`,
`catboost`, `duckdb`, `fastapi`, `reproducible-research`.

The original frozen experiment, missed days, research evaluation plans and replay
creation timestamps remain unchanged. The latest challenger is experimental and
has not replaced the frozen candidate. Back up private `data` separately before
moving machines; a Git repository intentionally omits those runtime artifacts.
