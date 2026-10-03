param([int]$Port = 11434)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot '.cache\ollama'
$portable = Join-Path $runtimeRoot 'runtime\ollama.exe'
if (Test-Path -LiteralPath $portable) {
    $executable = $portable
} else {
    $executable = (Get-Command ollama -ErrorAction Stop).Source
}
try {
    $existing = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/version" -TimeoutSec 2
    Write-Output "Ollama $($existing.version) is already listening on port $Port. No server settings changed."
    exit 0
} catch {}
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
$env:OLLAMA_HOST = "127.0.0.1:$Port"
$env:OLLAMA_MODELS = Join-Path $runtimeRoot 'models'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_CONTEXT_LENGTH = '4096'
$process = Start-Process -FilePath $executable -ArgumentList 'serve' -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runtimeRoot 'server.log') `
    -RedirectStandardError (Join-Path $runtimeRoot 'server-error.log')
$process.Id | Set-Content -LiteralPath (Join-Path $runtimeRoot 'server.pid')
Write-Output "Started local Ollama on port $Port (PID $($process.Id)). Logs: $runtimeRoot"
