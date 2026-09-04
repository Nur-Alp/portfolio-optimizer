@echo off
REM Double-click this file to stop the portfolio optimizer.
REM Thin wrapper around stop-optimizer.ps1 - see that file for the actual logic.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-optimizer.ps1"
timeout /t 2 >nul
