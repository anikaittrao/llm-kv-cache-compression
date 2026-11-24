# Audit wave 1 (2026-11-22): C1 + C1b, C3, C2, C4, then the C1 analysis. One job at a time; about 5 h on an idle CPU (measured smoke rates, 2026-11-22).
# Run from anywhere in Windows PowerShell 5.1:
#   powershell -ExecutionPolicy Bypass -File "<project>\experiments\rerun_2026-11-22\run_wave1.ps1"
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
    "[$t0] START $name : $($pyargs -join ' ')" | Tee-Object -FilePath "$rr\logs\wave1_progress.log" -Append
    & $py -u @pyargs 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Encoding utf8
    $code = $LASTEXITCODE
    $t1 = Get-Date
    "[$t1] END   $name exit=$code minutes=$([math]::Round(($t1 - $t0).TotalMinutes, 1))" | Tee-Object -FilePath "$rr\logs\wave1_progress.log" -Append
}

# C1: fresh windows 16-31 (paired per-window NLL is stored by the patched scripts)
Step 'c1_kv_buffer'    @('experiments\kv_buffer.py', '--window_offset', '16', '--settings', 'KV 256/512 P=768',
                         '--out', "$rr\c1_kv_buffer.json", '--figdir', $fig)                                     # ~55 min (12 methods, ~200 s/window)
Step 'c1_kv_update'    @('experiments\kv_update.py', '--window_offset', '16', '--settings', 'KV 256/512 P=768 B=32',
                         '--only', 'update', 'oracle', '--out', "$rr\c1_kv_update.json", '--figdir', $fig)         # ~55 min (also runs 'rsvd q=1 + update': --only 'update' is a substring match)
Step 'c1_kv_attnbasis' @('experiments\kv_attnbasis.py', '--window_offset', '16', '--settings', 'KV 256/512 P=768',
                         '--only', 'online /', 'attn-late /', 'offline / no buffer',
                         '--out', "$rr\c1_kv_attnbasis.json", '--figdir', $fig)                                   # ~25-35 min
# C1b: offline statistics without position 0, on the reported windows 0-15
Step 'c1b_kv_buffer_skip0' @('experiments\kv_buffer.py', '--calib_skip0', '--settings', 'KV 256/512 P=768', '--only', 'offline',
                             '--out', "$rr\c1b_kv_buffer_skip0.json", '--figdir', $fig)                           # ~10-15 min
# C3: out-of-sample (novelty) test + offline + update
Step 'c3_kv_novelty'   @('experiments\kv_novelty.py', '--out', "$rr\c3_kv_novelty.json", '--figdir', $fig)        # ~35 min
# C2: which exact tokens matter (B=1, sliding, random, top-attended; age split)
Step 'c2_kv_synergy'   @('experiments\kv_synergy.py', '--out', "$rr\c2_kv_synergy.json", '--figdir', $fig)       # ~65 min
# C4: prompt of 256 tokens, 768 new tokens
Step 'c4_kv_p256'      @('experiments\kv_p256.py', '--out', "$rr\c4_kv_p256.json", '--figdir', $fig)             # ~50 min
# C1 analysis (seconds)
Step 'c1_analyze'      @('experiments\c1_analyze.py')
"wave 1 finished" | Tee-Object -FilePath "$rr\logs\wave1_progress.log" -Append
