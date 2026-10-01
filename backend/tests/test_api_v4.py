"""API-level V4: alignment mode validation and parallel cloud jobs (E2E)."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import httpx
import pytest

from tests.conftest import requires_ffmpeg
from tests.test_api_v2 import MockSTTHandler  # noqa: F401  (ensures module import parity)


@pytest.fixture
async def client(tmp_path: Path, monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer

    from app.main import create_app

    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PT_TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("PT_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("PT_DB_PATH", str(tmp_path / "data" / "db.db"))
    monkeypatch.setenv("PT_FAKE_DELAY", "0")
    monkeypatch.setenv("PT_KEYCHAIN_SERVICE", f"pt-test-{uuid.uuid4().hex[:8]}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), MockSTTHandler)
    server.requests = []  # type: ignore[attr-defined]
    server.lock = threading.Lock()  # type: ignore[attr-defined]
    server.active = 0  # type: ignore[attr-defined]
    server.peak = 0  # type: ignore[attr-defined]
    server.delay = 0.0  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.setenv("PT_OPENROUTER_BASE_URL", f"http://{host}:{port}")

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=30
        ) as client:
            yield client, server
    server.shutdown()
    thread.join(timeout=5)


async def poll_job(client, job_id: str, statuses: set[str], timeout: float = 180):
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


async def test_cloud_alignment_requires_openrouter_backend(client, sine_wav):
    client, _server = client
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "provider": "local_mlx",
            "alignment_mode": "cloud",
            "language": "en",
        },
    )
    assert response.status_code == 422
    assert "OpenRouter" in response.json()["detail"]


async def test_cloud_alignment_accepted_with_openrouter(client, sine_wav):
    client, _server = client
    await client.put("/api/settings/openrouter_key", json={"key": "sk-test"})
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "provider": "openrouter",
            "openrouter_model": "openai/whisper-large-v3",
            "alignment_mode": "cloud",
            "language": "en",
        },
    )
    assert response.status_code == 201
    config = response.json()["jobs"][0]["config"]
    assert config["alignment_mode"] == "cloud"
    assert config["align_with_whisperx"] is False


async def test_parallel_cloud_settings_validation(client):
    client, _server = client
    response = await client.put("/api/settings", json={"max_parallel_cloud_jobs": 7})
    assert response.status_code == 422
    response = await client.put("/api/settings", json={"max_parallel_cloud_jobs": 2})
    assert response.status_code == 200
    assert response.json()["settings"]["max_parallel_cloud_jobs"] == 2


@requires_ffmpeg
async def test_two_cloud_jobs_transcribe_concurrently(client, sine_wav):
    client, server = client
    await client.put("/api/settings/openrouter_key", json={"key": "sk-test"})
    server.delay = 0.7  # type: ignore[attr-defined]

    jobs = []
    for _ in range(2):
        response = await client.post(
            "/api/jobs",
            json={
                "sources": [{"path": str(sine_wav)}],
                "provider": "openrouter",
                "openrouter_model": "openai/whisper-large-v3",
                "alignment_mode": "none",
                "language": "en",
            },
        )
        assert response.status_code == 201
        jobs.append(response.json()["jobs"][0]["id"])

    for job_id in jobs:
        job = await poll_job(client, job_id, {"completed", "failed"}, timeout=240)
        assert job["status"] == "completed", job.get("error")

    assert server.peak >= 2, f"expected concurrent mock requests, peak={server.peak}"
