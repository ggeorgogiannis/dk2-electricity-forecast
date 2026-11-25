# Prospective operation and recovery

Operation is manual by default. Installing the package, running the offline demo,
or serving the API does not register Windows tasks. The existing local tasks
were disabled at the owner's request. Scheduling commands later in this guide
are optional and must be invoked explicitly.

The original frozen comparison and research evaluations have separate identities
and directories. Never edit an existing freeze, origin receipt, issued forecast,
or replay. A missing first origin cannot be recovered retrospectively.

## Select or create a research evaluation

Use regular Windows CMD from the project folder. All dates below are computed
from the actual clock; no example date is reused as a future origin.

```cmd
rem Run from the cloned project folder.
set "DK2_REPORT=data\research\803ba646661d6a4f4691f305e8c00dac5384e7022751f4c5a0dbfb0f065ef02a\report.json"
for /f "delims=" %I in ('.venv\python.exe -m dk2_forecast.research_ops freeze "%DK2_REPORT%"') do @set "DK2_PLAN=%I"
type "%DK2_PLAN%"
```

The command creates a candidate under data/research_live/candidates and an
explicit evaluation under data/research_live/evaluations. Candidate identity
covers model, features, parameters, training policy, quantiles, execution source
and dependencies. Evaluation identity covers candidate, delivery dates, issue
policy, scoring and evaluation version. Repeating the same request is idempotent.
`--start YYYY-MM-DD --days 56` explicitly chooses another period. Different dates
produce another evaluation of the same candidate, preserving all missed days.
Allow at least ten minutes before the first 09:40 preparation for registration.
The automatic choice reserves another ten minutes.
After a missed origin, reuse the retained candidate even if the checkout has
changed:

```cmd
for /f "delims=" %I in ('.venv\python.exe -m dk2_forecast.research_ops restart "%DK2_PLAN%"') do @set "DK2_NEW_PLAN=%I"
```

The old DK2_PLAN is preserved; review DK2_NEW_PLAN and register its tasks
explicitly. Restart copies the old candidate and execution bundle reference,
not the latest checkout implementation.

Once selected, save the exact plan path in your CMD session. Do not use a report
hash as the evaluation selector. `DK2_PLAN` may also select an older research
freeze for inspection. Legacy report-hash directories and receipts stay in place;
use their matching retained source for execution. They are not migrated or
silently given schema-v2 eligibility. The original comparison continues through
its existing operations/prospective commands and validation contract.

## Register and verify tasks

```cmd
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\schedule.ps1 -Plan "%DK2_PLAN%"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\schedule.ps1 -Plan "%DK2_PLAN%" -CheckOnly
```

The override permits the reviewed script for that process only. It changes no
machine execution policy. Research tasks prepare at 09:40, verify inputs by
09:50, publish in the 10:00–10:01 Copenhagen minute, and collect actuals and score
at 14:30 on each issue day. Energinet publication can be delayed; incomplete
collection fails with retained logs and may be retried later. Forecast coverage
remains visible through inspection while actuals are pending. The final 14:30
collection is on the final issue date, when next-day actual prices are normally
published. If delayed, run collect-score manually once available.

Bounds derive from the selected plan, from start minus one day through
end-exclusive minus two days. Tasks have evaluation-specific names, an explicit
working directory and `.venv\python.exe`. Re-registering updates only those
names. Another evaluation's tasks are untouched. `-Weather` optionally adds an
identified 09:30 collection task. The existing standalone weather task may stay
in place; omit `-Weather` to avoid collecting an additional vintage.

Windows must use a timezone with Copenhagen's EU clock rules (this machine uses
Romance Standard Time). Local daily triggers then follow both clock changes;
the Python contract uses Europe/Copenhagen and UTC delivery keys. The script
rejects incompatible Windows zones without changing them. CheckOnly validates
actions, paths, dates, recurrence, enabled state, account, wake/catch-up/overlap
settings and next run time, and exits nonzero for mismatches or missing tasks.
It does not treat a previous failed result as a configuration success for that
past run: inspect logs separately.

Tasks use interactive logon: keep this account signed in. WakeToRun is enabled,
but firmware/power policy must permit wake; a powered-off PC cannot run tasks.
Delayed catch-up is disabled. Task execution is bounded to 19 minutes. Sign-out,
sleep that cannot wake, or a stalled preparation can still cause a missed run.
Future scheduled runs are pending until execution and receipt validation succeed.

Enable scheduler history from an administrator CMD if it is disabled:

```cmd
wevtutil sl Microsoft-Windows-TaskScheduler/Operational /e:true
```

This account previously lacked permission. Result 0x800710E0 means the operator
or administrator refused the request; it does not identify the underlying cause.

## Inspect, serve, collect and score

Set the evaluation directory from its selected plan filename:

```cmd
for %I in ("%DK2_PLAN%") do @set "DK2_EVAL=%~dpI"
.venv\python.exe -I "%DK2_EVAL%launch.py" inspect "%DK2_PLAN%" --root "%CD%\data"
.venv\python.exe -I "%DK2_EVAL%launch.py" collect-score "%DK2_PLAN%" --root "%CD%\data"
```

For existing actuals, preserve the exact snapshot identity:

```cmd
set "DK2_ACTUALS=data\prices\feb736cdf1bde74776333206f2416661e8f1e74c9eb5718eb2cc2c894d50bc3b\prices.parquet"
.venv\python.exe -I "%DK2_EVAL%launch.py" score "%DK2_PLAN%" --root "%CD%\data" --actuals "%DK2_ACTUALS%"
set "DK2_PRICES=%DK2_ACTUALS%"
set "DK2_EXPERIMENT=%DK2_EVAL%"
set "DK2_FORECASTS="
.venv\python.exe -m uvicorn dk2_forecast.api:create_app --factory
```

The API discovers issued forecasts from the selected evaluation and exposes
/evaluation with plan/candidate identities, current operational state, coverage
and latest aggregate score with an actual-price snapshot. Coverage-only inspections
do not replace an existing actual-price comparison. /forecasts/YYYY-MM-DD exposes provenance, fallback
metadata and raw/operational quantiles. An explicit DK2_FORECASTS override is
retained for compatibility. Invalid publications return 409; absent forecasts
return 404. Inspection and aggregate scoring validate all scheduled days,
including 92/96/100-slot delivery days. Scores distinguish not due, missed,
invalid/late, eligible awaiting actuals and scored. Accuracy uses identical
paired intervals with complete actual delivery days. Full-window and due-only
forecast coverage are separate from actual-price availability. A completed
evaluation means the issue window and actuals are complete; missed forecasts
still reduce coverage and do not become successful publications. Missing errors
are never replaced by zero. Operational, model-only and fallback subsets remain
separate. Replays are ineligible.

## Frozen execution, locks and recovery

Each schema-v2 evaluation retains launch.py, a source ZIP and exact dependency
versions. The launcher runs Python with -I, verifies itself, archive hashes and
packages, extracts only validated paths, then imports the retained code ahead of
the checkout. Later research edits cannot silently change execution. Restore
pinned dependencies if the environment changes. Retain data/code and the plan's
report.json for audit/recovery. Source archives are local generated artifacts.

Actions retain separate stdout/stderr logs, exit status, start/finish times and
failure reason beneath the selected evaluation's logs directory. A shared
execution.lock prevents overlap among preparation, publication and scoring.
Point and quantile models are cached by origin, source snapshot and configuration
with checked hashes. Publication and preparation commit complete directories
atomically. Retrying an existing valid publication returns it unchanged.

After termination, inspect the lock's PID, start time and running processes.
Only after confirming that process is no longer running, remove that exact lock:

```cmd
type "%DK2_EVAL%locks\execution.lock"
tasklist /FI "IMAGENAME eq python.exe"
del "%DK2_EVAL%locks\execution.lock"
```

Interrupted hidden .preparation-/.publication-/.model- staging folders are not
issued artifacts. Inspect and quarantine only abandoned staging folders; never
delete valid prepared/issued directories. Normal exceptions clean staging.
A missing block-origin receipt reports its next origin and recovery instruction.
Create a new future evaluation using the same report, or wait for the next
origin. Never backdate training or turn a replay into prospective evidence.

For the original frozen comparison, retain its existing plan and commands:

```cmd
set "DK2_ORIGINAL=data\experiments\c8081103f9516cd0b82d4c5e714f1cc406065d7e877c877434d636e540a7fd86\freeze.json"
.venv\python.exe -m dk2_forecast.operations inspect "%DK2_ORIGINAL%" --day 2026-11-18
.venv\python.exe -m dk2_forecast.operations score "%DK2_ORIGINAL%" --actuals "%DK2_ACTUALS%"
```

Its missed first training block remains missed. The next original block begins
with delivery 16 December, prepared 15 December. The original 18 November–12
January evaluation is not restarted or relabelled.

Point and quantile training receipts are independent obligations. If only a
seven-day point origin is available but the 28-day quantile origin was missed,
waiting one week cannot recover the interval model. Inspection reports the
missing quantile origin and the next shared origin (28 days for this candidate).
Restarting a future evaluation of the retained candidate is the earlier recovery
option. Cache receipts verify their exact origin/configuration key before reuse.
