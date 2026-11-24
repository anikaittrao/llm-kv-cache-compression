# Audit wave 2 (2026-11-22): C5, C7, C8, C6, C9, C10. One job at a time; about 4.3 h on an idle CPU (measured smoke rates, 2026-11-22).
# Run from anywhere in Windows PowerShell 5.1:
#   powershell -ExecutionPolicy Bypass -File "<project>\experiments\rerun_2026-11-22\run_wave2.ps1"
# Every output goes to experiments\rerun_2026-11-22\ (JSON), \fig (figures), \logs (logs). No reported JSON or figure
# is touched. A failed step is logged and the next step still runs.
$ErrorActionPreference = 'Continue'
$env:CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP = '1'   # harmless outside Claude Code
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$py = '.\.venv\Scripts\python.exe'
$rr = 'experiments\rerun_2026-11-22'
$fig = "$rr\fig"
New-Item -ItemType Directory -Force "$rr\logs", $fig | Out-Null

function Step([string]$name, [string[]]$pyargs) {
    $log = "$rr\logs\$name.log"
    $t0 = Get-Date
    "[$t0] START $name : $($pyargs -join ' ')" | Tee-Object -FilePath "$rr\logs\wave2_progress.log" -Append
    & $py -u @pyargs 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Encoding utf8
    $code = $LASTEXITCODE
    $t1 = Get-Date
    "[$t1] END   $name exit=$code minutes=$([math]::Round(($t1 - $t0).TotalMinutes, 1))" | Tee-Object -FilePath "$rr\logs\wave2_progress.log" -Append
}

# C5: equal-memory baselines (int8/int4/int2, sink+window, H2O-style, low-rank + int8)
Step 'c5_kv_baselines' @('experiments\kv_baselines.py', '--out', "$rr\c5_kv_baselines.json", '--figdir', $fig)        # ~40 min
# C7: sink mechanism on 16 windows + centered basis with mu_hat appended (+ the patched mechanism script on 16 windows)
Step 'c7_kv_sink2'     @('experiments\kv_sink2.py', '--out', "$rr\c7_kv_sink2.json", '--figdir', $fig)                # ~12-15 min
Step 'c7_kv_sink_check16' @('experiments\kv_sink_check.py', '--windows', '16', '--out', "$rr\c7_kv_sink_mech_16win.json")  # ~15 min (about 50 s per window)
# C8: block-diagonal value weight
Step 'c8_kv_gvblock'   @('experiments\kv_gvblock.py', '--out', "$rr\c8_kv_gvblock.json")                               # ~25 min
# C6: copy probe and greedy drift
Step 'c6_kv_probe'     @('experiments\kv_probe.py', '--out', "$rr\c6_kv_probe.json", '--figdir', $fig)                # ~75 min
# C9: trivial error proxies vs the bounds' ranking
Step 'c9_kv_proxy'     @('experiments\kv_proxy.py', '--out', "$rr\c9_kv_proxy.json")                                  # ~20 min
# C10: Frequent Directions + Prop 8.1 check (kv_cheapupdate.py --fd), main and P=384 settings
Step 'c10_kv_cheapupdate_fd' @('experiments\kv_cheapupdate.py', '--fd', '--only', 'svd every 32', 'brand every 32', 'fd every 32',
                               '--out', "$rr\c10_kv_cheapupdate_fd.json", '--figdir', $fig)                          # ~75 min
"wave 2 finished" | Tee-Object -FilePath "$rr\logs\wave2_progress.log" -Append
