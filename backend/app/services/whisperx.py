"""WhisperX forced alignment in an isolated virtual environment.

WhisperX pulls torch + transformers (~1.3 GB), so it lives in its own venv
(`.venv-whisperx`) and never touches the app's dependencies. The user can
install it from Settings (or with ./setup.sh --with-whisperx). Alignment runs
as a subprocess with a small JSON protocol (see scripts/whisperx_align.py).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from app.core.errors import AppError
from app.providers.base import CancellationToken

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WHISPERX_VENV = Path(os.environ.get("PT_WHISPERX_VENV", PROJECT_ROOT / ".venv-whisperx"))
ALIGN_SCRIPT = PROJECT_ROOT / "scripts" / "whisperx_align.py"
INSTALL_TIMEOUT_SECONDS = 3600

_status_cache: tuple[float, dict] | None = None
_STATUS_CACHE_SECONDS = 30.0


class WhisperXInstallError(AppError):
    code = "whisperx_install_error"
    user_message = "The WhisperX alignment environment could not be installed. See the logs."


def venv_python(venv: Path | None = None) -> Path:
    return (venv or WHISPERX_VENV) / "bin" / "python"


def whisperx_status(force: bool = False, venv: Path | None = None) -> dict:
    """Inspect the alignment environment (cached briefly)."""
    global _status_cache
    now = time.monotonic()
    if not force and _status_cache is not None and now - _status_cache[0] < _STATUS_CACHE_SECONDS:
        return _status_cache[1]

    python = venv_python(venv)
    status: dict = {
        "installed": False,
        "venv_path": str(venv or WHISPERX_VENV),
        "venv_exists": python.exists(),
        "script_exists": ALIGN_SCRIPT.is_file(),
        "device": None,
        "version": None,
        "message": "Not installed",
    }
    if python.exists():
        probe = (
            "import json, whisperx, torch;"
            "print(json.dumps({"
            "'version': getattr(whisperx, '__version__', 'unknown'),"
            "'device': 'mps' if torch.backends.mps.is_available() else 'cpu'"
            "}))"
        )
        try:
            result = subprocess.run(
                [str(python), "-c", probe], capture_output=True, text=True, timeout=120, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            status["message"] = f"WhisperX environment is not usable: {exc}"
        else:
            if result.returncode == 0:
                try:
                    info = json.loads(result.stdout.strip().splitlines()[-1])
                    status.update(
                        installed=True,
                        device=info.get("device"),
                        version=info.get("version"),
                        message=f"Installed (device: {str(info.get('device', 'cpu')).upper()})",
                    )
                except (ValueError, IndexError):
                    status["message"] = "WhisperX environment is broken; reinstall it."
            else:
                status["message"] = "WhisperX environment is incomplete; reinstall it."
    _status_cache = (time.monotonic(), status)
    return status


def invalidate_status_cache() -> None:
    global _status_cache
    _status_cache = None


ProgressReporter = Callable[[str, float | None], None]


def install_whisperx(
    progress: ProgressReporter,
    cancel: CancellationToken | None = None,
    venv: Path | None = None,
) -> None:
    """Create the isolated venv and install whisperx into it (idempotent)."""
    target = venv or WHISPERX_VENV
    python = venv_python(target)

    if not python.exists():
        progress("Creating alignment environment", None)
        _run_checked([sys.executable, "-m", "venv", str(target)])

    if cancel is not None:
        cancel.raise_if_cancelled()

    progress("Installing WhisperX and torch (this can take a few minutes)", None)
    _stream_pip(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--quiet",
            "pip",
        ],
        cancel,
    )
    _stream_pip(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-input",
            "whisperx",
        ],
        cancel,
    )

    progress("Verifying installation", None)
    result = subprocess.run(
        [str(python), "-c", "import whisperx, torch; print('ok')"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        log.error("WhisperX verification failed: %s", (result.stderr or "")[-800:])
        raise WhisperXInstallError(f"whisperx import failed: {result.stderr[-400:]}")
    invalidate_status_cache()
    progress("WhisperX alignment is ready", 1.0)


def _run_checked(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True, timeout=600, check=False)
    if result.returncode != 0:
        raise WhisperXInstallError(
            f"command failed ({result.returncode}): {' '.join(args[:3])}... {result.stderr[-300:]}"
        )


def _stream_pip(args: list[str], cancel: CancellationToken | None) -> None:
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    tail: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if cancel is not None and cancel.cancelled:
                process.terminate()
                process.wait(timeout=10)
                raise AppError("WhisperX installation cancelled", user_message="Installation cancelled.")
            line = line.rstrip()
            if line:
                tail.append(line)
                del tail[:-40]
        returncode = process.wait(timeout=INSTALL_TIMEOUT_SECONDS)
    except BaseException:
        process.kill()
        raise
    if returncode != 0:
        log.error("pip install failed: %s", "\n".join(tail[-20:]))
        raise WhisperXInstallError("pip install failed: " + "\n".join(tail[-5:]))
