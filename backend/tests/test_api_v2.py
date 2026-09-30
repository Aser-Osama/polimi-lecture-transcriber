from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from app.services import secrets as secrets_service
from tests.conftest import requires_ffmpeg


class MockSTTHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.requests.append(  # type: ignore[attr-defined]
            {
                "model": body.get("model"),
                "authorization": self.headers.get("Authorization", ""),
                "verbose": body.get("response_format") == "verbose_json",
                "format": (body.get("input_audio") or {}).get("format"),
                "language": body.get("language"),
            }
        )
        payload = {
            "text": "Welcome to the computer architecture lecture.",
            "language": "en",
            "duration": 6.0,
            "segments": [
                {"start": 0.0, "end": 3.0, "text": "Welcome to the computer"},
                {"start": 3.0, "end": 6.0, "text": "architecture lecture."},
            ],
            "words": [
                {"word": "Welcome", "start": 0.1, "end": 0.6},
                {"word": "computer", "start": 1.2, "end": 1.9},
                {"word": "architecture", "start": 3.1, "end": 3.9},
                {"word": "lecture.", "start": 4.0, "end": 4.6},
            ],
            "usage": {"cost": 0.0012, "seconds": 6.0},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # silence test output
        pass


@pytest.fixture
def mock_stt_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockSTTHandler)
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def keychain_service(monkeypatch):
    service = f"PolimiLectureTranscriber-test-{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("PT_KEYCHAIN_SERVICE", service)
    yield service
    secrets_service.delete_api_key(service=service)


@pytest.fixture
async def client(tmp_path: Path, monkeypatch, keychain_service, mock_stt_server):
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PT_TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("PT_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("PT_DB_PATH", str(tmp_path / "data" / "db.db"))
    monkeypatch.setenv("PT_FAKE_DELAY", "0")
    host, port = mock_stt_server.server_address
    monkeypatch.setenv("PT_OPENROUTER_BASE_URL", f"http://{host}:{port}")
    from app.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=30
        ) as client:
            yield client


async def poll_job(client: httpx.AsyncClient, job_id: str, statuses: set[str], timeout: float = 120):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = await client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        last = response.json()
        if last["status"] in statuses:
            return last
        await asyncio.sleep(0.1)
    raise AssertionError(f"job {job_id} never reached {statuses}; last={last}")


async def test_openrouter_models_catalog(client):
    response = await client.get("/api/models/openrouter")
    assert response.status_code == 200
    models = response.json()["models"]
    assert len(models) == 5
    assert models[0]["default"] is True
    assert models[0]["id"] == "microsoft/mai-transcribe-2"


async def test_settings_report_no_key_initially(client):
    response = await client.get("/api/settings")
    payload = response.json()
    assert payload["openrouter_key_present"] is False
    assert payload["settings"]["default_provider"] == "local_mlx"


async def test_key_store_and_delete_roundtrip(client, keychain_service):
    response = await client.put("/api/settings/openrouter_key", json={"key": "sk-or-unit-test"})
    assert response.status_code == 200
    assert (await client.get("/api/settings")).json()["openrouter_key_present"] is True
    # the key itself is never returned
    assert "sk-or-unit-test" not in (await client.get("/api/settings")).text

    response = await client.delete("/api/settings/openrouter_key")
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert (await client.get("/api/settings")).json()["openrouter_key_present"] is False


async def test_openrouter_job_without_key_is_rejected(client, sine_wav):
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "provider": "openrouter",
            "openrouter_model": "openai/whisper-large-v3",
            "language": "en",
        },
    )
    assert response.status_code == 422
    assert "OpenRouter API key" in response.json()["detail"]


async def test_invalid_openrouter_model_is_rejected(client):
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": "/tmp/whatever.mp4"}],
            "provider": "openrouter",
            "openrouter_model": "not/a-model",
            "language": "en",
        },
    )
    assert response.status_code == 422


async def test_alignment_status_endpoint(client):
    response = await client.get("/api/alignment/status")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["installed"], bool)
    assert payload["install"]["state"] == "idle"


async def test_alignment_install_endpoint_with_stubbed_installer(client, monkeypatch):
    events: list[dict] = []
    state = {"run": False}

    def fake_install(progress, cancel=None, venv=None):
        state["run"] = True
        progress("Installing WhisperX and torch", None)
        progress("WhisperX alignment is ready", 1.0)

    monkeypatch.setattr("app.services.whisperx.install_whisperx", fake_install)
    response = await client.post("/api/alignment/install")
    assert response.status_code == 200
    deadline = time.monotonic() + 10
    final = None
    while time.monotonic() < deadline:
        final = (await client.get("/api/alignment/status")).json()["install"]
        if final["state"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.05)
    assert state["run"] is True
    assert final is not None and final["state"] == "completed"
    assert events == []  # no SSE subscriber in this test


@requires_ffmpeg
async def test_openrouter_job_end_to_end_with_mock_server(client, keychain_service, mock_stt_server, sine_wav):
    await client.put("/api/settings/openrouter_key", json={"key": "sk-or-e2e"})

    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "provider": "openrouter",
            "openrouter_model": "openai/whisper-large-v3",
            "language": "en",
        },
    )
    assert response.status_code == 201
    job_id = response.json()["jobs"][0]["id"]
    job = await poll_job(client, job_id, {"completed", "failed"}, timeout=180)
    assert job["status"] == "completed", job.get("error")

    assert job["provider_meta"]["openrouter_model"] == "openai/whisper-large-v3"
    assert job["provider_meta"]["cost_usd"] == pytest.approx(0.0012, abs=1e-9)
    assert job["provider_meta"]["alignment_provider"] == "native_word_timestamps"
    assert job["outputs"]["srt"]
    assert Path(job["outputs"]["srt"]).is_file()
    assert Path(job["outputs"]["vtt"]).is_file()

    requests = mock_stt_server.requests
    assert requests, "mock server received no requests"
    assert all(request["verbose"] for request in requests)
    assert all(request["format"] == "mp3" for request in requests)
    assert all(request["authorization"].startswith("Bearer sk-or-e2e") for request in requests)
    assert all(request["model"] == "openai/whisper-large-v3" for request in requests)
    assert all(request["language"] == "en" for request in requests)

    preview = await client.get(f"/api/jobs/{job_id}/preview")
    assert preview.status_code == 200
    assert preview.json()["cues"]
