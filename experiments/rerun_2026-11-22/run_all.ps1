$env:CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP='1'
Set-Location '<project>'
& '<project>\experiments\rerun_2026-11-22\run_wave1.ps1'
& '<project>\experiments\rerun_2026-11-22\run_wave2.ps1'
.\.venv\Scripts\python.exe -u experiments\kv_rope_port.py --windows 8 --out experiments\rerun_2026-11-22\c11_rope_smollm2.json --figdir experiments\rerun_2026-11-22\fig *> experiments\rerun_2026-11-22\logs\c11_rope.log
'ALL DONE' | Out-File experiments\rerun_2026-11-22\logs\ALL_DONE.txt
