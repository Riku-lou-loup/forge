$ErrorActionPreference = "Stop"
$forgeRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $forgeRoot
$forgePython = Join-Path $forgeRoot '.venv\Scripts\python.exe'
& $forgePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
& $forgePython -m forge doctor
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
& $forgePython -m ruff check .
if ($LASTEXITCODE -ne 0) { throw 'Lint failed.' }
& $forgePython -m ruff format --check .
if ($LASTEXITCODE -ne 0) { throw 'Formatting check failed.' }
& $forgePython scripts/smoke.py
if ($LASTEXITCODE -ne 0) { throw 'Foundation smoke check failed.' }
& $forgePython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Behavioral tests failed.' }
