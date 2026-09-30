#!/usr/bin/env bash
# Development mode: backend with reload on 8765, Vite dev server on 5173.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${PT_HOST:-127.0.0.1}"
PORT="${PT_PORT:-8765}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31merror:\033[0m %s\n' "$*"; exit 1; }

[[ -x .venv/bin/python ]] || fail "Python environment missing. Run ./setup.sh first."
[[ -d frontend/node_modules ]] || (cd frontend && npm install --no-fund --no-audit)

export PT_HOST="$HOST" PT_PORT="$PORT"

say "Backend: http://$HOST:$PORT (reload) - API docs at /api/docs"
say "Frontend dev server: http://127.0.0.1:5173"
say "Press Ctrl+C to stop both."

(cd backend && ../.venv/bin/python -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload) &
BACKEND_PID=$!

(cd frontend && npm run dev) &
FRONTEND_PID=$!

cleanup() {
  say "Stopping dev servers"
  kill -INT "$BACKEND_PID" 2>/dev/null || true
  kill -INT "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
  wait "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

wait
