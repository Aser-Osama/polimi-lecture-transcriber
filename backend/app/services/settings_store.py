"""Persistent application settings (stored in SQLite as one JSON row)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from app.config import AppPaths
from app.models.domain import AppSettings

if TYPE_CHECKING:
    from app.db.database import Database

log = logging.getLogger(__name__)

_SETTINGS_KEY = "app_settings"


def resolve_output_dir(settings: AppSettings, paths: AppPaths) -> Path:
    if settings.output_dir:
        return Path(settings.output_dir).expanduser()
    return paths.default_output_dir


class SettingsStore:
    def __init__(self, database: Database):
        self._db = database
        self._lock = threading.RLock()
        self._cached: AppSettings | None = None

    def get(self) -> AppSettings:
        with self._lock:
            if self._cached is None:
                self._cached = self._load()
            return self._cached.model_copy(deep=True)

    def _load(self) -> AppSettings:
        raw = self._db.get_setting(_SETTINGS_KEY)
        if raw is None:
            return AppSettings()
        try:
            return AppSettings.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            log.warning("Malformed stored settings, falling back to defaults: %s", exc)
            return AppSettings()

    def update(self, patch: dict) -> AppSettings:
        """Merge a partial update; raises pydantic ValidationError on bad values."""
        with self._lock:
            current = self.get().model_dump(mode="json")
            merged = {**current, **patch}
            updated = AppSettings.model_validate(merged)
            self._db.set_setting(_SETTINGS_KEY, updated.model_dump_json())
            self._cached = updated
            return updated.model_copy(deep=True)

    def resolved_output_dir(self, paths: AppPaths) -> Path:
        return resolve_output_dir(self.get(), paths)
