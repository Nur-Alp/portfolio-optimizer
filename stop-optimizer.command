#!/usr/bin/env bash
# Double-click this file in Finder to stop the portfolio optimizer.
set -euo pipefail
cd "$(dirname "$0")"

STATE_DIR=".data/optimizer"

close_launcher_window() {
  if command -v osascript >/dev/null 2>&1; then
    (
      sleep 0.2
      osascript <<'APPLESCRIPT' >/dev/null 2>&1
tell application "Terminal"
  repeat with terminalWindow in windows
    if (name of terminalWindow contains "stop-optimizer.command") then
      close terminalWindow
      exit repeat
    end if
  end repeat
end tell
APPLESCRIPT
    ) &
  fi
}

stop_one() {
  # stop_one <label> <pid-file>
  local label="$1" pid_file="$2"
  if [ ! -f "$pid_file" ]; then
    echo "$label was not running."
    return
  fi
  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" >/dev/null 2>&1 || break
      sleep 0.5
    done
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" 2>/dev/null || true
      echo "$label force-stopped."
    else
      echo "$label stopped."
    fi
  else
    echo "$label was not running (stale PID file removed)."
  fi
  rm -f "$pid_file"
}

stop_one "Frontend" "$STATE_DIR/frontend.pid"
stop_one "Backend" "$STATE_DIR/backend.pid"

close_launcher_window
