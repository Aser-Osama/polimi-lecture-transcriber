"""Subprocess and desktop integration helpers.

All external commands are executed with argument arrays; no shell strings are
ever assembled from user input. Desktop actions (reveal in file manager, native
file picker) are implemented per platform; features that only exist on macOS
report their capability via :mod:`app.capabilities` and are hidden by the UI.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger(__name__)

_PICKER_SCRIPT = """
on run argv
    set promptText to item 1 of argv
    try
        set theFiles to choose file with prompt promptText of type {"public.movie", "public.audio"} with multiple selections allowed
    on error number -128
        return ""
    end try
    set out to ""
    repeat with f in theFiles
        set out to out & (POSIX path of f) & linefeed
    end repeat
    return out
end run
"""


def run_command(
    args: Sequence[str],
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(args))
    return subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def open_in_finder(path: Path) -> bool:
    """Reveal a file in the file manager (or open a directory)."""
    try:
        if sys.platform == "darwin":
            args = ["open", "-R", str(path)] if path.is_file() else ["open", str(path)]
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "win32":
            if path.is_file():
                subprocess.Popen(
                    ["explorer", f"/select,{path}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            target = path if path.is_dir() else path.parent
            subprocess.Popen(
                ["xdg-open", str(target)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True
    except OSError:
        log.exception("Failed to reveal %s", path)
        return False


def open_external(path: Path) -> bool:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(
                ["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        elif sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(
                ["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        return True
    except OSError:
        log.exception("Failed to open %s", path)
        return False


def pick_files_native(prompt: str) -> list[Path]:
    """Open the macOS native file picker and return chosen paths (no copying).

    Returns an empty list on other platforms; callers expose a capability flag
    so the UI hides this option there.
    """
    if sys.platform != "darwin":
        log.info("Native file picker requested on %s; not supported", sys.platform)
        return []
    try:
        result = subprocess.run(
            ["osascript", "-e", _PICKER_SCRIPT, prompt],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        log.exception("Native file picker failed")
        return []
    if result.returncode != 0:
        log.warning("File picker cancelled or failed: %s", result.stderr.strip())
        return []
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def kill_process_group(pid: int, sig: int = signal.SIGTERM) -> None:
    """Terminate a child process (and, on POSIX, its whole process group).

    Workers call ``os.setsid()`` on POSIX so this reaches FFmpeg grandchildren.
    Windows has no process groups; ``os.kill`` maps to TerminateProcess.
    """
    if os.name == "nt":
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            log.exception("Failed to signal process %s", pid)
        return
    try:
        os.killpg(os.getpgid(pid), sig)
    except ProcessLookupError:
        pass
    except OSError:
        log.exception("Failed to signal process group %s", pid)
