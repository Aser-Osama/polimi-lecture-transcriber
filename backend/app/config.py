"""Application paths and defaults.

Follows normal per-platform conventions: Application Support on macOS, APPDATA
on Windows, XDG directories on Linux. Every path can be overridden with
environment variables so tests never touch the real user directories.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.capabilities import sys_platform

APP_SLUG = "Polimi Lecture Transcriber"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def _default_data_dir() -> Path:
    platform = sys_platform()
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_SLUG
    if platform == "win32":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_SLUG
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_SLUG


def _default_logs_dir() -> Path:
    platform = sys_platform()
    if platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_SLUG
    if platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / APP_SLUG / "Logs"
    base = os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")
    return Path(base) / APP_SLUG / "logs"


@dataclass(frozen=True)
class AppPaths:
    data_dir: Path
    logs_dir: Path
    temp_dir: Path
    default_output_dir: Path
    db_path: Path

    @classmethod
    def from_env(cls) -> AppPaths:
        data_dir = _env_path("PT_DATA_DIR", _default_data_dir())
        logs_dir = _env_path("PT_LOGS_DIR", _default_logs_dir())
        temp_dir = _env_path("PT_TEMP_DIR", data_dir / "tmp")
        default_output_dir = _env_path(
            "PT_OUTPUT_DIR", Path.home() / "Documents" / "Polimi Transcripts"
        )
        db_path = _env_path("PT_DB_PATH", data_dir / "transcriber.db")
        return cls(
            data_dir=data_dir,
            logs_dir=logs_dir,
            temp_dir=temp_dir,
            default_output_dir=default_output_dir,
            db_path=db_path,
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.logs_dir,
            self.temp_dir,
            self.default_output_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, str]:
        return {
            "data_dir": str(self.data_dir),
            "logs_dir": str(self.logs_dir),
            "temp_dir": str(self.temp_dir),
            "default_output_dir": str(self.default_output_dir),
            "db_path": str(self.db_path),
        }


def default_host() -> str:
    return os.environ.get("PT_HOST", "127.0.0.1")


def default_port() -> int:
    return int(os.environ.get("PT_PORT", "8765"))


def debug_mode() -> bool:
    return os.environ.get("PT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
