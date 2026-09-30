"""Rotating file logging plus console output."""

from __future__ import annotations

import logging
import platform
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.version import APP_VERSION

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_configured = False


def configure_logging(logs_dir: Path, debug: bool = False) -> Path:
    """Configure root logging once; returns the log file path."""
    global _configured
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "app.log"
    if _configured:
        return log_file

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    _configured = True
    return log_file


def log_environment(app_version: str = APP_VERSION) -> None:
    log = logging.getLogger("app.startup")
    log.info(
        "Starting %s v%s on %s %s (%s), Python %s",
        "Polimi Lecture Transcriber",
        app_version,
        platform.system(),
        platform.release(),
        platform.machine(),
        sys.version.split()[0],
    )
