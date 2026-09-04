# Double-click start-optimizer.bat to open the portfolio optimizer.
# First run installs backend + frontend dependencies (needs internet access,
# takes a minute); later runs are fast.
# This is the Windows equivalent of start-optimizer.command (macOS) - same
# state dir, same ports, same fast-path-if-already-running behavior.

Set-Location -Path $PSScriptRoot

$StateDir = ".data\optimizer"
$BackendPidFile = "$StateDir\backend.pid"
$FrontendPidFile = "$StateDir\frontend.pid"
$BackendLog = "$StateDir\backend.log"
$FrontendLog = "$StateDir\frontend.log"
$BackendInstallMarker = "backend\.venv\.optimizer-install-fingerprint"
$BackendPort = 8511
$FrontendPort = 5173

function Fail($msg) {
    Write-Host ""
    Write-Host $msg
    Read-Host "Press Enter to close this window"
    exit 1
}

function Find-Python {
    foreach ($c in @("python3.13", "python3.12", "python3.11", "python3", "python")) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        & $c -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { return $c }
    }
    return $null
}

function Wait-For($url, $proc) {
    for ($i = 0; $i -lt 120; $i++) {
        try {
            Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null
            return $true
        } catch {}
        if ($proc.HasExited) { return $false }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

function Test-Alive($pidFile) {
    if (-not (Test-Path $pidFile)) { return $false }
    $procId = Get-Content $pidFile -ErrorAction SilentlyContinue
    if (-not $procId) { return $false }
    return [bool](Get-Process -Id $procId -ErrorAction SilentlyContinue)
}

if ((Test-Alive $BackendPidFile) -and (Test-Alive $FrontendPidFile)) {
    Write-Host "Optimizer is already running - opening it in your browser..."
    Start-Process "http://localhost:$FrontendPort"
    exit 0
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Fail "Node.js was not found. Install it from https://nodejs.org and double-click this file again."
}
$BasePython = Find-Python
if (-not $BasePython) {
    Fail "Python 3.11+ was not found. Install it from https://www.python.org/downloads and double-click this file again."
}

# --- Backend ---
if (-not (Test-Path "backend\.venv\Scripts\python.exe")) {
    Write-Host "Creating backend virtual environment (one-time)..."
    & $BasePython -m venv backend\.venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the backend virtual environment." }
}
$BackendPython = "backend\.venv\Scripts\python.exe"
$BackendReqHash = (Get-FileHash -Algorithm SHA256 "backend\requirements.txt").Hash

$needsInstall = $true
if (Test-Path $BackendInstallMarker) {
    $existing = Get-Content $BackendInstallMarker -ErrorAction SilentlyContinue
    if ($existing -eq $BackendReqHash) { $needsInstall = $false }
}
if ($needsInstall) {
    Write-Host "Installing backend dependencies (this can take a few minutes on first run)..."
    & $BackendPython -m pip install --quiet --upgrade pip
    if ($LASTEXITCODE -ne 0) { Fail "Could not update pip." }
    & $BackendPython -m pip install --quiet -r backend\requirements.txt
    if ($LASTEXITCODE -ne 0) { Fail "Could not install backend dependencies." }
    Set-Content -Path $BackendInstallMarker -Value $BackendReqHash
} else {
    Write-Host "Backend dependencies already up to date."
}

# --- Frontend ---
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing frontend dependencies (this can take a few minutes on first run)..."
    Push-Location frontend
    npm install --silent
    $installCode = $LASTEXITCODE
    Pop-Location
    if ($installCode -ne 0) { Fail "Could not install frontend dependencies." }
} else {
    Write-Host "Frontend dependencies already installed."
}

Write-Host "Starting the backend..."
$backendProc = Start-Process -FilePath $BackendPython `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--port", "$BackendPort" `
    -RedirectStandardOutput $BackendLog -RedirectStandardError "$BackendLog.err" `
    -WindowStyle Hidden -PassThru
Set-Content -Path $BackendPidFile -Value $backendProc.Id

if (-not (Wait-For "http://localhost:$BackendPort/api/health" $backendProc)) {
    Remove-Item $BackendPidFile -ErrorAction SilentlyContinue
    Fail "The backend did not start. Check $BackendLog for details."
}

Write-Host "Starting the frontend..."
$viteBin = Resolve-Path "frontend\node_modules\.bin\vite.cmd"
$frontendProc = Start-Process -FilePath $viteBin `
    -ArgumentList "frontend", "--port", "$FrontendPort", "--strictPort" `
    -RedirectStandardOutput $FrontendLog -RedirectStandardError "$FrontendLog.err" `
    -WindowStyle Hidden -PassThru
Set-Content -Path $FrontendPidFile -Value $frontendProc.Id

if (Wait-For "http://localhost:$FrontendPort" $frontendProc) {
    Start-Process "http://localhost:$FrontendPort"
    exit 0
}

Stop-Process -Id $backendProc.Id -ErrorAction SilentlyContinue
Remove-Item $BackendPidFile, $FrontendPidFile -ErrorAction SilentlyContinue
Fail "The frontend did not start. Check $FrontendLog for details."
