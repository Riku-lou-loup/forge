@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run scripts\setup.ps1 first. See README.md.
  pause
  exit /b 1
)
echo FORGE will be available at http://127.0.0.1:8511
".venv\Scripts\python.exe" -m forge app
if errorlevel 1 pause
