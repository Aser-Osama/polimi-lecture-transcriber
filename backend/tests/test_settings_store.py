from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.domain import AppSettings, LanguageChoice
from app.services.settings_store import SettingsStore


def test_defaults_when_empty(database):
    store = SettingsStore(database)
    settings = store.get()
    assert settings.default_model_key == "quality"
    assert settings.default_language == LanguageChoice.ENGLISH
    assert settings.output_dir is None
    assert settings.subtitles.max_line_chars == 42


def test_update_persists_across_instances(database):
    store = SettingsStore(database)
    store.update({"default_model_key": "fast", "glossary": "NUMA\nTLB"})
    fresh = SettingsStore(database)
    settings = fresh.get()
    assert settings.default_model_key == "fast"
    assert settings.glossary == "NUMA\nTLB"


def test_partial_update_keeps_other_fields(database):
    store = SettingsStore(database)
    store.update({"default_language": "it"})
    store.update({"glossary": "MESI"})
    settings = store.get()
    assert settings.default_language == LanguageChoice.ITALIAN
    assert settings.glossary == "MESI"


def test_nested_subtitle_preferences(database):
    store = SettingsStore(database)
    settings = store.update(
        {"subtitles": {"max_line_chars": 38, "max_lines": 2, "max_cue_duration": 6.0}}
    )
    assert settings.subtitles.max_line_chars == 38
    fresh = SettingsStore(database).get()
    assert fresh.subtitles.max_cue_duration == 6.0


def test_invalid_update_rejected(database):
    store = SettingsStore(database)
    with pytest.raises(ValidationError):
        store.update({"default_model_key": "gigantic"})
    with pytest.raises(ValidationError):
        store.update({"subtitles": {"max_line_chars": 500}})
    # stored value unchanged
    assert store.get().default_model_key == "quality"


def test_malformed_stored_settings_recover(database):
    database.set_setting("app_settings", "{not json")
    store = SettingsStore(database)
    assert store.get().default_model_key == "quality"


def test_unknown_stored_keys_recover(database):
    database.set_setting("app_settings", '{"totally_unknown": 1, "default_model_key": "fast"}')
    store = SettingsStore(database)
    assert store.get().default_model_key == "fast"


def test_resolved_output_dir(database, paths):
    store = SettingsStore(database)
    assert store.resolved_output_dir(paths) == paths.default_output_dir
    custom = paths.data_dir / "custom-out"
    store.update({"output_dir": str(custom)})
    assert store.resolved_output_dir(paths) == custom


def test_output_dir_tilde_expansion(database, paths):
    store = SettingsStore(database)
    store.update({"output_dir": "~/Somewhere"})
    assert store.resolved_output_dir(paths) == Path.home() / "Somewhere"
