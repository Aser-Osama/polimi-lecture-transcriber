from __future__ import annotations

import shutil
import subprocess
import uuid

import pytest

from app.services import secrets as secrets_service

SECURITY_AVAILABLE = shutil.which("security") is not None
requires_keychain = pytest.mark.skipif(
    not SECURITY_AVAILABLE, reason="macOS security CLI not available"
)


def test_store_api_key_rejects_empty():
    from app.core.errors import SecretsError

    with pytest.raises(SecretsError):
        secrets_service.store_api_key("   ")


def test_mocked_roundtrip(monkeypatch):
    store: dict[tuple[str, str], str] = {}

    class Result:
        def __init__(self, returncode=0, stdout=""):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(args: list[str]):
        if args[1] == "add-generic-password":
            store[(args[args.index("-s") + 1], args[args.index("-a") + 1])] = args[
                args.index("-w") + 1
            ]
            return Result()
        if args[1] == "find-generic-password":
            key = (args[args.index("-s") + 1], args[args.index("-a") + 1])
            if key in store:
                return Result(stdout=store[key] + "\n")
            return Result(returncode=44)
        if args[1] == "delete-generic-password":
            key = (args[args.index("-s") + 1], args[args.index("-a") + 1])
            existed = key in store
            store.pop(key, None)
            return Result(returncode=0 if existed else 44)
        return Result(returncode=1)

    monkeypatch.setattr(secrets_service, "_run", fake_run)
    service = "test-service"
    assert secrets_service.load_api_key(service=service) is None
    secrets_service.store_api_key("sk-or-test", service=service)
    assert secrets_service.load_api_key(service=service) == "sk-or-test"
    assert secrets_service.has_api_key(service=service) is True
    assert secrets_service.delete_api_key(service=service) is True
    assert secrets_service.load_api_key(service=service) is None


@requires_keychain
def test_real_keychain_roundtrip():
    service = f"PolimiLectureTranscriber-test-{uuid.uuid4().hex[:8]}"
    try:
        secrets_service.store_api_key("sk-or-integration-test", service=service)
        assert secrets_service.load_api_key(service=service) == "sk-or-integration-test"
        # overwrite works
        secrets_service.store_api_key("sk-or-second", service=service)
        assert secrets_service.load_api_key(service=service) == "sk-or-second"
    finally:
        secrets_service.delete_api_key(service=service)
    assert secrets_service.load_api_key(service=service) is None


def test_security_cli_never_used_with_shell(monkeypatch):
    calls: list[list[str]] = []

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: calls.append(args) or subprocess.CompletedProcess(args, 0, "", ""),
    )
    secrets_service.store_api_key("sk-or-x", service="svc")
    assert calls and isinstance(calls[0], list)
    assert calls[0][0] == "security"
