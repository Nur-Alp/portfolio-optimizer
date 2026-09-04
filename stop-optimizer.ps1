# Double-click stop-optimizer.bat to stop the portfolio optimizer.

Set-Location -Path $PSScriptRoot
$StateDir = ".data\optimizer"

function Stop-One($label, $pidFile) {
    if (-not (Test-Path $pidFile)) {
        Write-Host "$label was not running."
        return
    }
    $procId = Get-Content $pidFile -ErrorAction SilentlyContinue
    $proc = if ($procId) { Get-Process -Id $procId -ErrorAction SilentlyContinue } else { $null }
    if ($proc) {
        Stop-Process -Id $procId -ErrorAction SilentlyContinue
        for ($i = 0; $i -lt 20; $i++) {
            if (-not (Get-Process -Id $procId -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 500
        }
        if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            Write-Host "$label force-stopped."
        } else {
            Write-Host "$label stopped."
        }
    } else {
        Write-Host "$label was not running (stale PID file removed)."
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

Stop-One "Frontend" "$StateDir\frontend.pid"
Stop-One "Backend" "$StateDir\backend.pid"
