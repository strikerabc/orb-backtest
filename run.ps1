<#
run.ps1 -- Windows wrapper for run.py.

Two differences from the launcher it replaces:

1. **No hardcoded path.** The old start_gpu_cpu_sweep.ps1 embedded
   C:\Users\strik\Downloads\orb-backtest, so it broke for any other checkout or
   clone. This derives the repo root from its own location.

2. **Foreground by default.** The old script always used Start-Process with a
   hidden window, so it could only ever be a fire-and-forget background job -- the
   interactive prompts would have been invisible and the run would hang waiting on
   input nobody could see. Foreground here, with -Background as an opt-in that also
   forces non-interactive flags.

Usage:
    .\run.ps1                        # interactive prompts
    .\run.ps1 -Accelerated           # skip prompts, maximum safe capacity
    .\run.ps1 -SingleCore            # skip prompts, one core
    .\run.ps1 -Accelerated -Fresh    # discard checkpoints
    .\run.ps1 -SelfTest              # CPU/CUDA parity only
    .\run.ps1 -Accelerated -Background   # detached, logs to runs/
#>
param(
    [switch]$Accelerated,
    [switch]$SingleCore,
    [switch]$Fresh,
    [switch]$SelfTest,
    [switch]$DryRun,
    [switch]$Background,
    [int]$Workers = 0
)

$ErrorActionPreference = "Stop"

$repo = $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"   # fall back to PATH if there is no local venv
}

$runner = Join-Path $repo "run.py"
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Cannot find run.py next to this script (looked in $repo)"
}

$arguments = @("-u", $runner)
if ($Accelerated) { $arguments += "--accelerated" }
if ($SingleCore)  { $arguments += "--single-core" }
if ($Fresh)       { $arguments += "--fresh" }
if ($SelfTest)    { $arguments += "--self-test" }
if ($DryRun)      { $arguments += "--dry-run" }
if ($Workers -gt 0) { $arguments += @("--workers", "$Workers") }

if ($Background) {
    # A detached run cannot answer prompts, so a mode must be explicit. Defaulting
    # silently to accelerated here would hand the whole machine to a background job
    # the user did not ask to be greedy.
    if (-not ($Accelerated -or $SingleCore -or $SelfTest)) {
        throw "-Background requires -Accelerated, -SingleCore or -SelfTest (a detached run cannot answer prompts)"
    }
    $runs = Join-Path $repo "runs"
    New-Item -ItemType Directory -Force -Path $runs | Out-Null
    $stdout = Join-Path $runs "run.stdout.log"
    $stderr = Join-Path $runs "run.stderr.log"
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $repo -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath (Join-Path $runs "run.pid") -Value $process.Id
    Write-Output "Started PID $($process.Id)"
    Write-Output "  stdout : $stdout"
    Write-Output "  stderr : $stderr"
    Write-Output "  status : $(Join-Path $runs 'gpu_cpu_sweep.status.json')"
    Write-Output ""
    Write-Output "Follow with:  Get-Content '$stdout' -Wait"
} else {
    & $python @arguments
    exit $LASTEXITCODE
}
