# Audit wave 3 (2026-11-24): C17, C18, C16, C14, C15. One job at a time. GPT-2 jobs use 8 threads; the SmolLM2 jobs (kv_rope_port.py,
# kv_rope_baselines.py) use 4 threads because the 1-window smoke tests were ~20% SLOWER with 8 (results identical). Estimated total ~6.5 h on an idle CPU
# Run from anywhere in Windows PowerShell 5.1:
#   powershell -ExecutionPolicy Bypass -File "<project>\experiments\rerun_2026-11-22\run_wave3.ps1"
# Every output goes to experiments\rerun_2026-11-22\ (JSON, c<N>_summary.json), \fig_wave3 (figures of kv_rope_port.py; kept apart
# from fig\ so that the C11 figure is not overwritten) and \logs (logs c<N>*.log, wave3_progress.log, WAVE3_DONE.txt at the end).
# No reported JSON or figure is touched. A failed step is logged and the next step still runs.
$ErrorActionPreference = 'Continue'
$env:CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP = '1'   # harmless outside Claude Code
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$py = '.\.venv\Scripts\python.exe'
$rr = 'experiments\rerun_2026-11-22'
$fig = "$rr\fig_wave3"
New-Item -ItemType Directory -Force "$rr\logs", $fig | Out-Null

function Step([string]$name, [string[]]$pyargs) {
    $log = "$rr\logs\$name.log"
    $t0 = Get-Date
    "[$t0] START $name : $($pyargs -join ' ')" | Tee-Object -FilePath "$rr\logs\wave3_progress.log" -Append
    & $py -u @pyargs 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Encoding utf8
    $code = $LASTEXITCODE
    $t1 = Get-Date
    "[$t1] END   $name exit=$code minutes=$([math]::Round(($t1 - $t0).TotalMinutes, 1))" | Tee-Object -FilePath "$rr\logs\wave3_progress.log" -Append
}

# ---- C17: AttSVD accumulating mode (continuation never compressed), GPT-2, main setting + P=384 keys only (~15 min)
Step 'c17'          @('experiments\kv_accum.py', '--threads', '8', '--out', "$rr\c17_kv_accum.json")
Step 'c17_analyze'  @('experiments\c17_analyze.py')

# ---- C18: (a) equal average exact count, per-step NLL (~26 min) + C12 controls with per-step NLL (~28 min); (b) interleaved novelty (~5 min)
Step 'c18a'         @('experiments\kv_equalcount.py', '--threads', '8', '--out', "$rr\c18a_kv_equalcount.json")
Step 'c18a_control' @('experiments\kv_control.py', '--per_step', '--threads', '8', '--out', "$rr\c18a_kv_control_perstep.json")
Step 'c18b'         @('experiments\kv_novelty_interleaved.py', '--threads', '8', '--out', "$rr\c18b_kv_novelty_interleaved.json")
Step 'c18_analyze'  @('experiments\c18_analyze.py')

# ---- C16: low rank + quantization on GPT-2, 14 methods, 16 windows (calibration ~1.5 min + ~7 min per window: see smoke estimate)
Step 'c16'          @('experiments\kv_quantlr.py', '--threads', '8', '--out', "$rr\c16_kv_quantlr.json")
Step 'c16_analyze'  @('experiments\c16_analyze.py')

# ---- C14: why SmolLM2 differs from GPT-2
# (a) SmolLM2 at GPT-2 geometry: P 768, seq 1024, K64/V128, 16 windows, 9 methods
Step 'c14a'         @('experiments\kv_rope_port.py', '--P', '768', '--seq', '1024', '--RK', '64', '--RV', '128', '--windows', '16',
                      '--threads', '4', '--out', "$rr\c14a_smollm2_gpt2geom.json", '--figdir', "$fig\c14a")
# (b) SmolLM2 main geometry with plain G_K (kweight plain), 8 windows (offset 0), 9 methods
Step 'c14b'         @('experiments\kv_rope_port.py', '--kweight', 'plain', '--windows', '8', '--threads', '4',
                      '--out', "$rr\c14b_smollm2_plainK.json", '--figdir', "$fig\c14b")
# (c) GPT-2 at low R/n: K64/V128, P 768, 16 windows
Step 'c14c'         @('experiments\kv_lowrn.py', '--threads', '8', '--out', "$rr\c14c_kv_lowrn.json")
# optional intermediate rank (K128/V256, R/n 0.17/0.33): remove the leading # to run (~40 min)
Step 'c14c_mid'   @('experiments\kv_lowrn.py', '--threads', '8', '--RK', '128', '--RV', '256', '--out', "$rr\c14c_kv_lowrn_K128V256.json")
Step 'c14_analyze'  @('experiments\c14_analyze.py')

# ---- C15: SmolLM2, fresh windows 16-31: the 9 low-rank cells, then the baselines + offline basis
Step 'c15a'         @('experiments\kv_rope_port.py', '--window_offset', '16', '--windows', '16', '--threads', '4',
                      '--out', "$rr\c15a_rope_w16-31.json", '--figdir', "$fig\c15a")
Step 'c15b'         @('experiments\kv_rope_baselines.py', '--threads', '4', '--out', "$rr\c15b_rope_baselines_w16-31.json")
Step 'c15_analyze'  @('experiments\c15_analyze.py')

"wave 3 finished" | Tee-Object -FilePath "$rr\logs\wave3_progress.log" -Append
'WAVE3 DONE' | Out-File "$rr\logs\WAVE3_DONE.txt"

