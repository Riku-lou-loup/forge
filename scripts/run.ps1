param([int]$Port = 8511)
$ErrorActionPreference = "Stop"
$forgeRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $forgeRoot
& '.\.venv\Scripts\python.exe' -m forge app --port $Port
exit $LASTEXITCODE
