"""Secure storage for the OpenRouter API key.

macOS: the key is stored as a generic password in the user's login Keychain
via the ``security`` CLI. ``PT_KEYCHAIN_SERVICE`` exists so tests can use an
isolated keychain entry.

Windows/Linux: there is no Keychain, so the key is kept in a single file
(``openrouter.key``) inside the app data directory with user-only permissions
(0600; best effort on Windows). It is never written to the app database, never
logged, never returned to the browser and never committed. ``PT_KEY_FILE``
overrides the location (used by tests).
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from pathlib import Path

from app.capabilities import key_storage_kind
from app.config import AppPaths
from app.core.errors import SecretsError

log = logging.getLogger(__name__)

SERVICE_DEFAULT = "PolimiLectureTranscriber"
ACCOUNT = "openrouter"

_NOT_FOUND_RETURNCODE = 44
_KEY_FILENAME = "openrouter.key"


def _service() -> str:
    return os.environ.get("PT_KEYCHAIN_SERVICE", SERVICE_DEFAULT)


def _key_file() -> Path:
    override = os.environ.get("PT_KEY_FILE")
    if override:
        return Path(override).expanduser()
    return AppPaths.from_env().data_dir / _KEY_FILENAME


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False, timeout=30)


# --------------------------------------------------------------- keychain


def _store_keychain(key: str, service: str | None, account: str) -> None:
    result = _run(
        [
            "security",
            "add-generic-password",
            "-U",
            "-a",
            account,
            "-s",
            service or _service(),
            "-w",
            key,
        ]
    )
    if result.returncode != 0:
        # stderr may echo the command; do not log it to avoid leaking the key.
        log.error("Keychain write failed (returncode %s)", result.returncode)
        raise SecretsError(f"security add-generic-password failed: {result.returncode}")
    log.info("OpenRouter API key stored in macOS Keychain")


def _load_keychain(service: str | None, account: str) -> str | None:
    result = _run(
        [
            "security",
            "find-generic-password",
            "-a",
            account,
            "-s",
            service or _service(),
            "-w",
        ]
    )
    if result.returncode == _NOT_FOUND_RETURNCODE:
        return None
    if result.returncode != 0:
        log.error("Keychain read failed (returncode %s)", result.returncode)
        raise SecretsError(f"security find-generic-password failed: {result.returncode}")
    key = result.stdout.strip()
    return key or None


def _delete_keychain(service: str | None, account: str) -> bool:
    result = _run(
        [
            "security",
            "delete-generic-password",
            "-a",
            account,
            "-s",
            service or _service(),
        ]
    )
    return result.returncode == 0


# --------------------------------------------------------------- key file


def _store_file(key: str) -> None:
    path = _key_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(key, encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(temp, 0o600)
        os.replace(temp, path)
    except OSError as exc:
        log.error("API key file write failed: %s", exc)
        raise SecretsError("The API key could not be saved to the app data folder.") from exc
    log.info("OpenRouter API key stored in %s", path)


def _load_file() -> str | None:
    path = _key_file()
    try:
        if not path.is_file():
            return None
        key = path.read_text(encoding="utf-8").strip()
        return key or None
    except OSError as exc:
        log.error("API key file read failed: %s", exc)
        raise SecretsError("The stored API key could not be read.") from exc


def _delete_file() -> bool:
    path = _key_file()
    try:
        if not path.is_file():
            return False
        path.unlink()
        return True
    except OSError as exc:
        log.error("API key file delete failed: %s", exc)
        raise SecretsError("The stored API key could not be removed.") from exc


# ------------------------------------------------------------- public API


def store_api_key(key: str, service: str | None = None, account: str = ACCOUNT) -> None:
    if not key.strip():
        raise SecretsError("Empty API key", user_message="The API key is empty.")
    if key_storage_kind() == "keychain":
        _store_keychain(key.strip(), service, account)
    else:
        _store_file(key.strip())


def load_api_key(service: str | None = None, account: str = ACCOUNT) -> str | None:
    if key_storage_kind() == "keychain":
        return _load_keychain(service, account)
    return _load_file()


def delete_api_key(service: str | None = None, account: str = ACCOUNT) -> bool:
    if key_storage_kind() == "keychain":
        return _delete_keychain(service, account)
    return _delete_file()


def has_api_key(service: str | None = None, account: str = ACCOUNT) -> bool:
    return load_api_key(service=service, account=account) is not None
