$ErrorActionPreference = "Stop"
$forgeRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $forgeRoot
$forgePython = Join-Path $forgeRoot '.venv\Scripts\python.exe'
& $forgePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Resolve dependency conflicts before recording versions.' }
# Keep the optional CPU training stack in requirements-torch.txt.
$forgeVersions = @(& $forgePython -m pip freeze --all --exclude forge-maintenance `
    --exclude torch --exclude filelock --exclude fsspec --exclude mpmath `
    --exclude networkx --exclude sympy)
if ($LASTEXITCODE -ne 0) { throw 'Could not read installed dependency versions.' }
$forgeHeader = @(
    '# FORGE dependency snapshot: Windows x86-64, Python 3.13.',
    '# Regenerate with scripts/lock.ps1 after an intentional dependency update.',
    '# Install the local project separately with --no-deps --no-build-isolation -e .'
)
($forgeHeader + $forgeVersions) | Set-Content -LiteralPath 'requirements-lock.txt' -Encoding utf8
Write-Output 'Saved requirements-lock.txt'
