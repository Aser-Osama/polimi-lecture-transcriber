#!/usr/bin/env bash
# Normal startup: build if needed, start FastAPI on 127.0.0.1, open browser.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${PT_HOST:-127.0.0.1}"
PORT="${PT_PORT:-8765}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31merror:\033[0m %s\n' "$*"; exit 1; }

if [[ "${1:-}" == "--test" ]]; then
  say "Running backend tests"
  .venv/bin/python -m pytest backend/tests -q
  shift
fi

[[ -x .venv/bin/python ]] || fail "Python environment missing. Run ./setup.sh first."

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  say "FFmpeg is missing; the app will start but transcription requires it."
  say "Install with: brew install ffmpeg"
fi

if [[ ! -f frontend/dist/index.html ]]; then
  say "Frontend build missing; building now"
  (cd frontend && npm install --no-fund --no-audit && npm run build)
fi

say "Starting Polimi Lecture Transcriber at http://$HOST:$PORT"
export PT_HOST="$HOST" PT_PORT="$PORT"

.venv/bin/python -m app.main &
SERVER_PID=$!

cleanup() {
  say "Shutting down (pid $SERVER_PID)"
  kill -INT "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 50); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 0.1
  done
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

say "Waiting for the server to become ready"
READY=0
for _ in $(seq 1 100); do
  if curl -sf "http://$HOST:$PORT/api/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 0.2
done

if [[ "$READY" == "1" ]]; then
  say "Opening browser"
  open "http://$HOST:$PORT"
else
  say "Server did not report healthy; it may still be starting. Open http://$HOST:$PORT manually."
fi

wait "$SERVER_PID"
