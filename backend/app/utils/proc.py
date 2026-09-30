"""Subprocess and macOS integration helpers.

All external commands are executed with argument arrays; no shell strings are
ever assembled from user input.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
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
    """Reveal a file in Finder (or open a directory)."""
    target = str(path)
    args = ["open", "-R", target] if path.is_file() else ["open", target]
    try:
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError:
        log.exception("Failed to reveal %s", path)
        return False


def open_external(path: Path) -> bool:
    try:
        subprocess.Popen(
            ["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except OSError:
        log.exception("Failed to open %s", path)
        return False


def pick_files_native(prompt: str) -> list[Path]:
    """Open the macOS native file picker and return chosen paths (no copying)."""
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
    """Signal the whole process group of a child started with start_new_session."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except ProcessLookupError:
        pass
    except OSError:
        log.exception("Failed to signal process group %s", pid)
