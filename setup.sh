#!/usr/bin/env bash
# First-time setup for Polimi Lecture Transcriber.
# Safe to run multiple times.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

INSTALL_FFMPEG=0
for arg in "$@"; do
  case "$arg" in
    --install-ffmpeg) INSTALL_FFMPEG=1 ;;
    -h|--help)
      echo "Usage: ./setup.sh [--install-ffmpeg]"
      echo "  --install-ffmpeg  run 'brew install ffmpeg' automatically (asks nothing)"
      exit 0
      ;;
  esac
done

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31merror:\033[0m %s\n' "$*"; exit 1; }

# ---------------------------------------------------------------- platform
say "Checking platform"
[[ "$(uname -s)" == "Darwin" ]] || fail "This app targets macOS (Apple Silicon). Detected: $(uname -s)."
if [[ "$(uname -m)" != "arm64" ]]; then
  warn "Detected architecture $(uname -m); MLX requires Apple Silicon and will not work here."
fi
say "macOS $(sw_vers -productVersion) on $(uname -m)"

# ------------------------------------------------------------------- tools
say "Checking Homebrew"
if command -v brew >/dev/null 2>&1; then
  say "Homebrew found: $(brew --version | head -1)"
  HAVE_BREW=1
else
  HAVE_BREW=0
  warn "Homebrew is not installed. Install it from https://brew.sh and re-run ./setup.sh"
  warn "Continuing; other checks may fail without it."
fi

say "Checking FFmpeg / ffprobe"
MISSING_FFMPEG=0
command -v ffmpeg >/dev/null 2>&1 || MISSING_FFMPEG=1
command -v ffprobe >/dev/null 2>&1 || MISSING_FFMPEG=1
if [[ "$MISSING_FFMPEG" == "1" ]]; then
  if [[ "$INSTALL_FFMPEG" == "1" && "$HAVE_BREW" == "1" ]]; then
    say "Installing FFmpeg with Homebrew (requested with --install-ffmpeg)"
    brew install ffmpeg
  elif [[ "$HAVE_BREW" == "1" ]]; then
    fail "FFmpeg is missing. Install it with:

    brew install ffmpeg

  or re-run: ./setup.sh --install-ffmpeg"
  else
    fail "FFmpeg is missing and Homebrew is not available. Install FFmpeg first."
  fi
fi
say "FFmpeg: $(ffmpeg -version 2>/dev/null | head -1)"

say "Checking Node.js / npm"
command -v node >/dev/null 2>&1 || fail "Node.js is required to build the interface. Install it from https://nodejs.org (or 'brew install node')."
command -v npm >/dev/null 2>&1 || fail "npm is required to build the interface."
say "Node $(node --version), npm $(npm --version)"

# ------------------------------------------------------------------ python
say "Locating Python 3.11/3.12"
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    case "$version" in
      3.11|3.12) PYTHON_BIN="$candidate"; break ;;
    esac
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  # uv-installed pythons are common on developer Macs
  if [[ -x "$HOME/.local/bin/python3.12" ]]; then
    PYTHON_BIN="$HOME/.local/bin/python3.12"
  fi
fi
[[ -n "$PYTHON_BIN" ]] || fail "Python 3.11 or 3.12 is required. Install e.g. 'brew install python@3.12' and re-run."
say "Using $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

# -------------------------------------------------------------------- venv
if [[ ! -x .venv/bin/python ]]; then
  say "Creating Python virtual environment in .venv"
  "$PYTHON_BIN" -m venv .venv
else
  say "Virtual environment already exists"
fi

say "Installing backend dependencies"
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -e "backend[dev]"

# ---------------------------------------------------------------- frontend
say "Installing frontend dependencies"
(cd frontend && npm install --no-fund --no-audit)

say "Building frontend"
(cd frontend && npm run build)

# ------------------------------------------------------------- app folders
say "Initializing application directories and database"
.venv/bin/python - <<'PY'
from app.config import AppPaths
from app.db.database import Database

paths = AppPaths.from_env()
paths.ensure_directories()
Database(paths.db_path)
print(f"  data:    {paths.data_dir}")
print(f"  logs:    {paths.logs_dir}")
print(f"  outputs: {paths.default_output_dir}")
print(f"  temp:    {paths.temp_dir}")
PY

say "Setup complete."
echo
echo "Next steps:"
echo "  ./run.sh              start the app (opens http://127.0.0.1:8765)"
echo "  ./dev.sh              development mode (Vite hot reload)"
echo "  ./run.sh --test       run the backend test suite first"
echo
echo "Models download on demand on first transcription (Quality model ~3 GB)."
