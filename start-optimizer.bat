@echo off
REM Double-click this file to open the portfolio optimizer.
REM Thin wrapper around start-optimizer.ps1 - see that file for the actual logic.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-optimizer.ps1"
if errorlevel 1 pause
