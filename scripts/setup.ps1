param([string]$PythonCommand = "python")
$ErrorActionPreference = "Stop"
$forgeRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $forgeRoot
$forgePython = Join-Path $forgeRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $forgePython)) {
    & $PythonCommand -c "import sys; assert sys.version_info[:2] == (3, 13), 'FORGE requires Python 3.13'"
    if ($LASTEXITCODE -ne 0) { throw 'Select a Python 3.13 interpreter with -PythonCommand.' }
    & $PythonCommand -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$env:PIP_CACHE_DIR = Join-Path $forgeRoot '.cache\pip'
if (Test-Path -LiteralPath 'requirements-lock.txt') {
    & $forgePython -m pip install -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed.' }
    & $forgePython -m pip install --no-deps --no-build-isolation -e .
} else {
    & $forgePython -m pip install -e '.[dev]'
}
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
& $forgePython -m forge doctor
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
