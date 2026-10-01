#!/usr/bin/env python3
"""Cross-platform launcher for Polimi Lecture Transcriber.

Works on macOS, Windows and Linux:

    python run.py            set up (first run), build the UI, start the app
    python run.py --check    verify prerequisites and print a summary, then exit
    python run.py --no-browser
    python run.py --skip-build

On macOS the local MLX backend and WhisperX alignment are available. On
Windows and Linux the app runs the OpenRouter cloud path only (the UI hides
the local options automatically).

Environment: PT_HOST, PT_PORT (defaults 127.0.0.1:8765) are honored, like the
macOS ./run.sh script.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
FRONTEND = ROOT / "frontend"
BACKEND = ROOT / "backend"

MIN_PYTHON = (3, 11)


def say(message: str) -> None:
    print(f"==> {message}", flush=True)


def warn(message: str) -> None:
    print(f"warning: {message}", flush=True)


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def venv_python(venv: Path = VENV) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def npm_command() -> str | None:
    return shutil.which("npm.cmd") if os.name == "nt" else shutil.which("npm")


def ffmpeg_hint() -> str:
    if sys.platform == "darwin":
        return "brew install ffmpeg"
    if sys.platform == "win32":
        return "winget install Gyan.FFmpeg"
    return "sudo apt install ffmpeg (or your distribution's package manager)"


def platform_capabilities() -> dict:
    sys.path.insert(0, str(BACKEND))
    try:
        from app.capabilities import capabilities

        return capabilities()
    finally:
        sys.path.pop(0)


def ensure_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        fail(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; "
            f"this is {sys.version.split()[0]} ({sys.executable})."
        )
    say(f"Using Python {sys.version.split()[0]} ({sys.executable})")


def ensure_venv() -> Path:
    python = venv_python()
    if python.exists():
        say(f"Virtual environment found at {VENV}")
        return python
    say(f"Creating virtual environment in {VENV}")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(
            f"Could not create the virtual environment ({exc}). "
            "On Debian/Ubuntu install python3-venv first."
        )
    return venv_python()


def dependencies_ok(python: Path) -> bool:
    probe = subprocess.run(
        [str(python), "-c", "import fastapi, uvicorn, app"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


def ensure_dependencies(python: Path) -> None:
    if dependencies_ok(python):
        say("Backend dependencies already installed")
        return
    say("Installing backend dependencies (first run only)")
    try:
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
            check=True,
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", "-e", str(BACKEND)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"Dependency installation failed ({exc}). Check your internet connection.")
    if not dependencies_ok(python):
        fail("Backend dependencies are still not importable after installation.")


def ensure_frontend() -> None:
    dist = FRONTEND / "dist" / "index.html"
    if dist.is_file():
        say("Frontend build found")
        return
    npm = npm_command()
    if npm is None:
        fail(
            "Node.js/npm is required to build the interface. Install it from "
            "https://nodejs.org and re-run."
        )
    if not (FRONTEND / "node_modules").is_dir():
        say("Installing frontend dependencies")
        try:
            subprocess.run([npm, "install", "--no-fund", "--no-audit"], cwd=str(FRONTEND), check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            fail(f"npm install failed ({exc}).")
    say("Building frontend")
    try:
        subprocess.run([npm, "run", "build"], cwd=str(FRONTEND), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"Frontend build failed ({exc}).")


def check_ffmpeg() -> bool:
    ok = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
    if not ok:
        warn(
            "FFmpeg/ffprobe not found. Transcription needs them. Install with: "
            f"{ffmpeg_hint()}"
        )
    return ok


def print_capabilities() -> None:
    caps = platform_capabilities()
    if caps["local_transcription"]:
        mode = "local MLX Whisper + OpenRouter"
    else:
        mode = "OpenRouter cloud only (local MLX is macOS-only)"
    say(f"Platform: {caps['platform_label']} - {mode}")
    key_storage = "macOS Keychain" if caps["key_storage"] == "keychain" else "app data folder"
    say(f"OpenRouter key storage: {key_storage}")


def run_server(python: Path, open_browser: bool) -> int:
    host = os.environ.get("PT_HOST", "127.0.0.1")
    port = os.environ.get("PT_PORT", "8765")
    url = f"http://{host}:{port}"
    say(f"Starting Polimi Lecture Transcriber at {url}")
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        return subprocess.call([str(python), "-m", "app.main"], cwd=str(ROOT))
    except KeyboardInterrupt:
        return 0


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    open_browser = "--no-browser" not in argv
    skip_build = "--skip-build" in argv

    ensure_python_version()
    print_capabilities()
    python = ensure_venv()
    ensure_dependencies(python)
    if not skip_build:
        ensure_frontend()
    ffmpeg_ok = check_ffmpeg()
    if check_only:
        say("Check complete." if ffmpeg_ok else "Check complete (with warnings).")
        return 0
    return run_server(python, open_browser)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
