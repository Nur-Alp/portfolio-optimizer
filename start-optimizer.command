#!/usr/bin/env bash
# Double-click this file in Finder to open the portfolio optimizer.
# First run installs backend + frontend dependencies (needs internet access,
# takes a minute); later runs are fast.
set -euo pipefail
cd "$(dirname "$0")"

STATE_DIR=".data/optimizer"
BACKEND_PID_FILE="$STATE_DIR/backend.pid"
FRONTEND_PID_FILE="$STATE_DIR/frontend.pid"
BACKEND_LOG="$STATE_DIR/backend.log"
FRONTEND_LOG="$STATE_DIR/frontend.log"
BACKEND_INSTALL_MARKER="backend/.venv/.optimizer-install-fingerprint"
BACKEND_PORT=8511
FRONTEND_PORT=5173

close_launcher_window() {
  if command -v osascript >/dev/null 2>&1; then
    (
      sleep 0.2
      osascript <<'APPLESCRIPT' >/dev/null 2>&1
tell application "Terminal"
  repeat with terminalWindow in windows
    if (name of terminalWindow contains "start-optimizer.command") then
      close terminalWindow
      exit repeat
    end if
  end repeat
end tell
APPLESCRIPT
    ) &
  fi
}

fail() {
  echo
  echo "$1"
  read -n 1 -s -r -p "Press any key to close this window..."
  exit 1
}

find_python() {
  for candidate in python3.13 python3.12 python3.11 python3 python \
    /opt/anaconda3/bin/python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    if command -v "$candidate" >/dev/null 2>&1 && \
      "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

wait_for() {
  # wait_for <url> <pid> - poll a URL until it answers or the process dies.
  local url="$1" pid="$2"
  for _ in $(seq 1 120); do
    curl --silent --fail "$url" >/dev/null 2>&1 && return 0
    kill -0 "$pid" >/dev/null 2>&1 || return 1
    sleep 0.5
  done
  return 1
}

mkdir -p "$STATE_DIR"

already_running=true
[ -f "$BACKEND_PID_FILE" ] && kill -0 "$(cat "$BACKEND_PID_FILE")" >/dev/null 2>&1 || already_running=false
[ -f "$FRONTEND_PID_FILE" ] && kill -0 "$(cat "$FRONTEND_PID_FILE")" >/dev/null 2>&1 || already_running=false

if [ "$already_running" = true ]; then
  echo "Optimizer is already running - opening it in your browser..."
  open "http://localhost:$FRONTEND_PORT"
  close_launcher_window
  exit 0
fi

command -v node >/dev/null 2>&1 || fail "Node.js was not found. Install it from https://nodejs.org and double-click this file again."
BASE_PYTHON="$(find_python)" || fail "Python 3.11+ was not found. Install it from https://www.python.org/downloads/ and double-click this file again."

# --- Backend ---
if [ ! -x "backend/.venv/bin/python" ]; then
  echo "Creating backend virtual environment (one-time)..."
  "$BASE_PYTHON" -m venv backend/.venv || fail "Could not create the backend virtual environment."
fi
BACKEND_PYTHON="backend/.venv/bin/python"
BACKEND_REQ_HASH="$(shasum -a 256 backend/requirements.txt | awk '{print $1}')"

if [ ! -f "$BACKEND_INSTALL_MARKER" ] || [ "$(cat "$BACKEND_INSTALL_MARKER" 2>/dev/null)" != "$BACKEND_REQ_HASH" ]; then
  echo "Installing backend dependencies (this can take a few minutes on first run)..."
  "$BACKEND_PYTHON" -m pip install --quiet --upgrade pip || fail "Could not update pip."
  "$BACKEND_PYTHON" -m pip install --quiet -r backend/requirements.txt || fail "Could not install backend dependencies."
  echo "$BACKEND_REQ_HASH" > "$BACKEND_INSTALL_MARKER"
else
  echo "Backend dependencies already up to date."
fi

# --- Frontend ---
if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies (this can take a few minutes on first run)..."
  (cd frontend && npm install --silent) || fail "Could not install frontend dependencies."
else
  echo "Frontend dependencies already installed."
fi

echo "Starting the backend..."
"$BACKEND_PYTHON" -m uvicorn app.main:app --app-dir backend --port "$BACKEND_PORT" \
  > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"

wait_for "http://localhost:$BACKEND_PORT/api/health" "$BACKEND_PID" \
  || { rm -f "$BACKEND_PID_FILE"; fail "The backend did not start. Check $BACKEND_LOG for details."; }

echo "Starting the frontend..."
frontend/node_modules/.bin/vite frontend --port "$FRONTEND_PORT" --strictPort \
  > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"

if wait_for "http://localhost:$FRONTEND_PORT" "$FRONTEND_PID"; then
  open "http://localhost:$FRONTEND_PORT"
  close_launcher_window
  exit 0
fi

kill "$BACKEND_PID" 2>/dev/null || true
rm -f "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"
fail "The frontend did not start. Check $FRONTEND_LOG for details."
