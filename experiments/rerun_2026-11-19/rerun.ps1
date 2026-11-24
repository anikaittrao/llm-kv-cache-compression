# Rerun of the timing-producing scripts on an idle CPU (2026-11-19). Outputs stay in this folder;
# the original JSONs/figures are untouched until the runs are compared.
$ErrorActionPreference = "Continue"
$env:CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP = "1"
$root = "<project>"
$d = "$root\experiments\rerun_2026-11-19"
Set-Location $root
$py = ".venv\Scripts\python.exe"
$jobs = @(
  @("model_quality",   "experiments\model_quality.py",   "results_quality.json"),
  @("kv_online",       "experiments\kv_online.py",       "results_kv_online.json"),
  @("llm_compression", "experiments\llm_compression.py", "results.json")
)
foreach ($j in $jobs) {
  $name = $j[0]
  $load = (Get-CimInstance Win32_Processor).LoadPercentage
  "$(Get-Date -Format s) START $name (cpu load $load%)" | Out-File -Append -Encoding utf8 "$d\chain.log"
  $extra = @()
  if ($name -eq "model_quality") { $extra = @("--fresh") }
  & $py -u $j[1] --out "$d\$($j[2])" --figdir "$d\fig" @extra *> "$d\$name.log"
  "$(Get-Date -Format s) END $name exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 "$d\chain.log"
}
"ALL DONE" | Out-File -Append -Encoding utf8 "$d\chain.log"
