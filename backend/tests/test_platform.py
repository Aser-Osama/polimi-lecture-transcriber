"""Platform capability, path, provider-gating and key-storage tests.

The suite runs on macOS, so non-macOS behavior is exercised through the
``PT_SIMULATE_PLATFORM`` override that :mod:`app.capabilities` honors.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest

from app import capabilities
from app.config import AppPaths
from app.core.errors import SecretsError
from app.providers import create_provider, provider_names
from app.services import media as media_service
from app.services import secrets
from app.services import whisperx as whisperx_service

ROOT = Path(__file__).resolve().parents[2]

_PT_PATH_VARS = (
    "PT_DATA_DIR",
    "PT_LOGS_DIR",
    "PT_TEMP_DIR",
    "PT_OUTPUT_DIR",
    "PT_DB_PATH",
    "PT_KEY_FILE",
)


@pytest.fixture
def clean_path_env(monkeypatch):
    for name in _PT_PATH_VARS:
        monkeypatch.delenv(name, raising=False)


def test_simulated_platform_override(monkeypatch):
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "win32")
    assert capabilities.sys_platform() == "win32"
    assert not capabilities.is_macos()
    assert capabilities.platform_label() == "Windows"
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "linux")
    assert capabilities.platform_label() == "Linux"
    monkeypatch.delenv("PT_SIMULATE_PLATFORM")
    assert capabilities.sys_platform() == sys.platform


def test_capabilities_matrix(monkeypatch):
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "linux")
    caps = capabilities.capabilities()
    assert caps["local_transcription"] is False
    assert caps["local_alignment"] is False
    assert caps["native_file_picker"] is False
    assert caps["key_storage"] == "file"
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "darwin")
    caps = capabilities.capabilities()
    assert caps["local_transcription"] is True
    assert caps["local_alignment"] is True
    assert caps["native_file_picker"] is True
    assert caps["key_storage"] == "keychain"


def test_config_default_paths_windows(monkeypatch, tmp_path, clean_path_env):
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    paths = AppPaths.from_env()
    assert paths.data_dir == tmp_path / "roaming" / "Polimi Lecture Transcriber"
    assert paths.logs_dir == tmp_path / "local" / "Polimi Lecture Transcriber" / "Logs"
    assert paths.db_path == paths.data_dir / "transcriber.db"


def test_config_default_paths_linux(monkeypatch, tmp_path, clean_path_env):
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    paths = AppPaths.from_env()
    assert paths.data_dir == tmp_path / "share" / "Polimi Lecture Transcriber"
    assert paths.logs_dir == tmp_path / "state" / "Polimi Lecture Transcriber" / "logs"


def test_config_env_overrides_win(monkeypatch, tmp_path):
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "win32")
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "custom"))
    monkeypatch.setenv("PT_DB_PATH", str(tmp_path / "custom" / "db.sqlite"))
    paths = AppPaths.from_env()
    assert paths.data_dir == tmp_path / "custom"
    assert paths.db_path == tmp_path / "custom" / "db.sqlite"


def test_provider_gating(monkeypatch):
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "win32")
    assert "local_mlx" not in provider_names()
    assert "openrouter" in provider_names()
    with pytest.raises(ValueError, match="only available on macOS"):
        create_provider("local_mlx")
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "darwin")
    assert "local_mlx" in provider_names()
    provider = create_provider("local_mlx")
    assert provider.name == "local_mlx"


def test_key_storage_file_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "linux")
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("PT_KEY_FILE", raising=False)
    assert secrets.load_api_key() is None
    assert secrets.has_api_key() is False

    secrets.store_api_key("sk-or-test-123")
    assert secrets.load_api_key() == "sk-or-test-123"
    assert secrets.has_api_key() is True

    key_file = tmp_path / "data" / "openrouter.key"
    assert key_file.is_file()
    if os.name == "posix":
        assert stat.S_IMODE(key_file.stat().st_mode) == 0o600

    secrets.store_api_key("sk-or-replaced")
    assert secrets.load_api_key() == "sk-or-replaced"

    assert secrets.delete_api_key() is True
    assert secrets.load_api_key() is None
    assert secrets.delete_api_key() is False


def test_key_storage_file_rejects_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "win32")
    monkeypatch.setenv("PT_KEY_FILE", str(tmp_path / "key"))
    with pytest.raises(SecretsError):
        secrets.store_api_key("   ")


def test_whisperx_status_unsupported(monkeypatch):
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "win32")
    status = whisperx_service.whisperx_status(force=True)
    assert status["supported"] is False
    assert status["installed"] is False
    assert "macOS" in status["message"]


def test_ffmpeg_hint_per_platform(monkeypatch):
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "darwin")
    assert "brew" in media_service.ffmpeg_install_hint()
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "win32")
    assert "winget" in media_service.ffmpeg_install_hint()
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "linux")
    assert "apt" in media_service.ffmpeg_install_hint()


def _load_run_module():
    spec = importlib.util.spec_from_file_location("run_launcher", ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_py_helpers(monkeypatch):
    run = _load_run_module()
    python_path = run.venv_python()
    if os.name == "nt":
        assert python_path.name == "python.exe"
    else:
        assert python_path.name == "python"
    assert python_path.parent.name in {"bin", "Scripts"}
    monkeypatch.setattr("sys.platform", "win32")
    assert "winget" in run.ffmpeg_hint()
    monkeypatch.setattr("sys.platform", "linux")
    assert "apt" in run.ffmpeg_hint()
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "linux")
    caps = run.platform_capabilities()
    assert caps["local_transcription"] is False
