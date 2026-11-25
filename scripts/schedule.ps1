param(
    [Parameter(Mandatory=$true)][string]$Plan,
    [switch]$CheckOnly,
    [switch]$Weather
)
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
$python = Join-Path $project '.venv\python.exe'
$Plan = (Resolve-Path -LiteralPath $Plan).Path
$allowedZones = @('Romance Standard Time', 'W. Europe Standard Time', 'Central Europe Standard Time', 'Central European Standard Time')
$zone = Get-TimeZone
if ($zone.Id -notin $allowedZones) {
    throw "Windows timezone '$($zone.Id)' does not match Copenhagen scheduling. Select a Copenhagen-compatible timezone before registration; no machine settings were changed."
}
$specArgs = @('-m','dk2_forecast.research_ops','schedule-spec',$Plan)
if ($Weather) { $specArgs += '--weather' }
Push-Location $project
try {
    $raw = & $python @specArgs
    if ($LASTEXITCODE -ne 0) { throw 'Failed to validate plan and derive scheduler bounds' }
    $spec = $raw | ConvertFrom-Json
} finally { Pop-Location }
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 19)
$failures = @()
foreach ($definition in $spec.tasks) {
    if (-not $CheckOnly) {
        $trigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::ParseExact($definition.start_boundary, 'yyyy-MM-ddTHH:mm:ss', [cultureinfo]::InvariantCulture))
        # Floating local wall time follows Copenhagen DST; do not retain New-ScheduledTaskTrigger's numeric offset.
        $trigger.StartBoundary = $definition.start_boundary
        $trigger.EndBoundary = $definition.end_boundary
        $action = New-ScheduledTaskAction -Execute $definition.execute -Argument $definition.arguments -WorkingDirectory $definition.directory
        Register-ScheduledTask -TaskName $definition.name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    }
    $task = Get-ScheduledTask -TaskName $definition.name -ErrorAction SilentlyContinue
    if (-not $task) { $failures += "Missing task: $($definition.name)"; continue }
    $info = Get-ScheduledTaskInfo -InputObject $task
    $mismatch = @()
    if (-not $task.Settings.Enabled -or -not $task.Triggers[0].Enabled) { $mismatch += 'disabled' }
    if ($task.Actions.Count -ne 1 -or $task.Actions[0].Execute -ne $definition.execute -or $task.Actions[0].Arguments -ne $definition.arguments -or $task.Actions[0].WorkingDirectory -ne $definition.directory) { $mismatch += 'action/interpreter/directory' }
    if ($task.Triggers.Count -ne 1 -or $task.Triggers[0].StartBoundary -ne $definition.start_boundary -or $task.Triggers[0].EndBoundary -ne $definition.end_boundary -or $task.Triggers[0].DaysInterval -ne 1) { $mismatch += 'trigger dates or recurrence' }
    $registeredSid = ([Security.Principal.NTAccount]::new($task.Principal.UserId)).Translate([Security.Principal.SecurityIdentifier]).Value
    if ([string]$task.Principal.LogonType -ne 'Interactive' -or $registeredSid -ne [Security.Principal.WindowsIdentity]::GetCurrent().User.Value) { $mismatch += 'logon account/type' }
    if (-not $task.Settings.WakeToRun -or $task.Settings.StartWhenAvailable -or [string]$task.Settings.MultipleInstances -ne 'IgnoreNew' -or $task.Settings.ExecutionTimeLimit -ne 'PT19M') { $mismatch += 'wake/catch-up/overlap/time limit' }
    $now = Get-Date
    $first = [datetime]::ParseExact($definition.start_boundary, 'yyyy-MM-ddTHH:mm:ss', [cultureinfo]::InvariantCulture)
    $end = [datetime]::ParseExact($definition.end_boundary, 'yyyy-MM-ddTHH:mm:ss', [cultureinfo]::InvariantCulture)
    $next = $now.Date.Add($first.TimeOfDay)
    if ($next -le $now) { $next = $next.AddDays(1) }
    if ($next -lt $first) { $next = $first }
    if ($next -lt $end -and $info.NextRunTime -ne $next) { $mismatch += 'next run differs from plan' }
    if ($mismatch.Count) { $failures += "$($definition.name): $($mismatch -join ', ')" }
    [pscustomobject]@{
        Name=$task.TaskName; State=[string]$task.State; Enabled=$task.Settings.Enabled
        NextRun=$info.NextRunTime; LastRun=$info.LastRunTime; LastResult=('0x{0:X8}' -f $info.LastTaskResult)
        FirstIssue=$spec.first_issue_date; LastIssue=$spec.last_issue_date; Timezone=$zone.Id
        Interpreter=$task.Actions[0].Execute; Directory=$task.Actions[0].WorkingDirectory
        Arguments=$task.Actions[0].Arguments; Validation= $(if ($mismatch.Count) { $mismatch -join ', ' } else { 'OK' })
    } | Format-List
}
if ($failures.Count) {
    $failures | ForEach-Object { Write-Error $_ -ErrorAction Continue }
    exit 1
}
