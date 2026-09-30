"""Secure storage for the OpenRouter API key (macOS Keychain).

The key is stored as a generic password in the user's login Keychain via the
`security` CLI. It is never written to the app database, never logged, never
returned to the browser and never committed. `PT_KEYCHAIN_SERVICE` exists only
so tests can use an isolated keychain entry.
"""

from __future__ import annotations

import logging
import os
import subprocess

from app.core.errors import SecretsError

log = logging.getLogger(__name__)

SERVICE_DEFAULT = "PolimiLectureTranscriber"
ACCOUNT = "openrouter"

_NOT_FOUND_RETURNCODE = 44


def _service() -> str:
    return os.environ.get("PT_KEYCHAIN_SERVICE", SERVICE_DEFAULT)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False, timeout=30)


def store_api_key(key: str, service: str | None = None, account: str = ACCOUNT) -> None:
    if not key.strip():
        raise SecretsError("Empty API key", user_message="The API key is empty.")
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
            key.strip(),
        ]
    )
    if result.returncode != 0:
        # stderr may echo the command; do not log it to avoid leaking the key.
        log.error("Keychain write failed (returncode %s)", result.returncode)
        raise SecretsError(f"security add-generic-password failed: {result.returncode}")
    log.info("OpenRouter API key stored in macOS Keychain")


def load_api_key(service: str | None = None, account: str = ACCOUNT) -> str | None:
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


def delete_api_key(service: str | None = None, account: str = ACCOUNT) -> bool:
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


def has_api_key(service: str | None = None, account: str = ACCOUNT) -> bool:
    return load_api_key(service=service, account=account) is not None
