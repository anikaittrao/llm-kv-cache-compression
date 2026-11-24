# Audit wave 4 (2026-11-24): C23, C20, C19, C21, C22, cheapest first, one job at a time. Run from anywhere in Windows PowerShell 5.1:
#   powershell -ExecutionPolicy Bypass -File "<project>\experiments\rerun_2026-11-22\run_wave4.ps1"
# GPT-2 jobs use 8 threads, SmolLM2 jobs 4 (as wave 3). Outputs: experiments\rerun_2026-11-22\c19..c23 JSONs, \fig_wave4, \logs\c<N>.log,
# logs\wave4_progress.log, logs\WAVE4_DONE.txt. No reported JSON or figure is touched. A failed step is logged and the next one still runs.
$ErrorActionPreference = 'Continue'
$env:CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$py = '.\.venv\Scripts\python.exe'
$rr = 'experiments\rerun_2026-11-22'
$fig = "$rr\fig_wave4"
New-Item -ItemType Directory -Force "$rr\logs", $fig | Out-Null

function Step([string]$name, [string[]]$pyargs) {
    $log = "$rr\logs\$name.log"
    $t0 = Get-Date
    "[$t0] START $name : $($pyargs -join ' ')" | Tee-Object -FilePath "$rr\logs\wave4_progress.log" -Append
    & $py -u @pyargs 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Encoding utf8
    $code = $LASTEXITCODE
    $t1 = Get-Date
    "[$t1] END   $name exit=$code minutes=$([math]::Round(($t1 - $t0).TotalMinutes, 1))" | Tee-Object -FilePath "$rr\logs\wave4_progress.log" -Append
}

Step 'c23' @('experiments\kv_additivity.py', '--threads', '8', '--out', "$rr\c23_kv_additivity.json")
Step 'c20' @('experiments\kv_keep0_diag.py', '--threads', '8', '--out', "$rr\c20_kv_keep0_diag.json")
Step 'c19' @('experiments\kv_rope_sink.py', '--threads', '4', '--out', "$rr\c19_rope_sink.json")
# C21: per-KV-head basis, equal stored numbers (K21/V43 per head = 63/129 per layer); layer-scope cells come from c11 (windows 0-7)
Step 'c21_head'  @('experiments\kv_rope_port.py', '--scope', 'head', '--RK', '63', '--RV', '129', '--windows', '8', '--only', 'static', 'update / buffer',
                   '--threads', '4', '--out', "$rr\c21_rope_scope_head.json", '--figdir', "$fig\c21")
Step 'c21_analyze' @('experiments\c21_analyze.py')
Step 'c22' @('experiments\kv_domainshift.py', '--threads', '8', '--out', "$rr\c22_kv_domainshift.json")

"wave 4 finished" | Tee-Object -FilePath "$rr\logs\wave4_progress.log" -Append
'WAVE4 DONE' | Out-File "$rr\logs\WAVE4_DONE.txt"
