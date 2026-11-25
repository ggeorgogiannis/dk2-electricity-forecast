# Contributing

Use Python 3.12 and the pinned dependencies. See the README for Windows CMD
setup and the offline demo. No API token or external data is needed for tests.

Before proposing a change, run from the project folder:

```cmd
.venv\python.exe -m pytest -q
.venv\python.exe -m ruff check .
.venv\python.exe -m ruff format --check .
.venv\python.exe -m pip check
.venv\python.exe scripts\offline_demo.py
.venv\python.exe scripts\check_publication.py
```

The commands use the project-local Conda interpreter. Standard venv and Linux
paths are described in the README. CI runs the offline checks on Windows and
Linux when a push or pull request occurs; it does not run daily forecasts.

Keep changes focused and add regression tests for changed behavior. Explain
the problem, resulting behavior, validation and limitations. Do not add raw
market data, trained model pickles, tokens, local environments or runtime logs.

Research changes must declare dates, data versions, feature sets, refit policy,
selection procedure and denominators. Compare methods on identical UTC delivery
intervals and report coverage separately. Preserve negative/extreme slices and
losing periods. Never fit preprocessing or calibration on target/future outcomes.
Do not label previously inspected dates as an untouched test.

Preserve immutable freezes, source bundles and receipt identities. A new model
or policy requires a separate candidate/evaluation. Historical replays cannot
repair missed prospective publications. Scheduling is an explicit optional
operation and must not be introduced into installation, tests or API startup.
