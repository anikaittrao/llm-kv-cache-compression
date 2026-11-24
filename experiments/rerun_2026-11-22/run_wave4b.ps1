# Wave 4b (2026-11-24): restart of the unfinished Wave 4 items (C23 is done and NOT rerun): C20, C19, C21, C22, strictly sequential.
# Restartable: a step is skipped when its output JSON already exists. Run:
#   powershell -ExecutionPolicy Bypass -File "<project>\experiments\rerun_2026-11-22\run_wave4b.ps1"
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
$prog = "$rr\logs\wave4b_progress.log"

function Step([string]$name, [string]$outjson, [string[]]$pyargs) {
    if (Test-Path $outjson) { "[$(Get-Date)] SKIP  $name (exists: $outjson)" | Tee-Object -FilePath $prog -Append; return }
    $log = "$rr\logs\${name}b.log"
    $t0 = Get-Date
    "[$t0] START $name : $($pyargs -join ' ')" | Tee-Object -FilePath $prog -Append
    & $py -u @pyargs 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Encoding utf8
    $code = $LASTEXITCODE
    $t1 = Get-Date
    "[$t1] END   $name exit=$code minutes=$([math]::Round(($t1 - $t0).TotalMinutes, 1))" | Tee-Object -FilePath $prog -Append
}

Step 'c20' "$rr\c20_kv_keep0_diag.json" @('experiments\kv_keep0_diag.py', '--threads', '8', '--out', "$rr\c20_kv_keep0_diag.json")
Step 'c19' "$rr\c19_rope_sink.json" @('experiments\kv_rope_sink.py', '--threads', '4', '--out', "$rr\c19_rope_sink.json")
Step 'c21_head' "$rr\c21_rope_scope_head.json" @('experiments\kv_rope_port.py', '--scope', 'head', '--RK', '63', '--RV', '129', '--windows', '8', '--only', 'static', 'update / buffer',
    '--threads', '4', '--out', "$rr\c21_rope_scope_head.json", '--figdir', "$fig\c21")
Step 'c21_analyze' "$rr\c21_summary.json" @('experiments\c21_analyze.py')
Step 'c22' "$rr\c22_kv_domainshift.json" @('experiments\kv_domainshift.py', '--threads', '8', '--out', "$rr\c22_kv_domainshift.json")

"wave 4b finished" | Tee-Object -FilePath $prog -Append
'WAVE4B DONE' | Out-File "$rr\logs\WAVE4B_DONE.txt"
