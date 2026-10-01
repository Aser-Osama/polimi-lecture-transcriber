"""API behavior on non-macOS platforms (cloud-only path).

Uses ``PT_SIMULATE_PLATFORM`` so the Windows/Linux behavior is exercised on
the macOS development machine.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest


@pytest.fixture
async def windows_client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PT_TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("PT_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("PT_DB_PATH", str(tmp_path / "data" / "db.db"))
    monkeypatch.setenv("PT_FAKE_DELAY", "0")
    monkeypatch.setenv("PT_SIMULATE_PLATFORM", "win32")
    from app.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=30
        ) as client:
            yield client


async def test_health_reports_capabilities(windows_client):
    payload = (await windows_client.get("/api/health")).json()
    caps = payload["capabilities"]
    assert caps["platform"] == "win32"
    assert caps["platform_label"] == "Windows"
    assert caps["local_transcription"] is False
    assert caps["local_alignment"] is False
    assert caps["native_file_picker"] is False
    assert caps["key_storage"] == "file"
    assert payload["mlx_provider"]["supported"] is False
    assert payload["mlx_provider"]["ok"] is True


async def test_models_catalog_empty(windows_client):
    payload = (await windows_client.get("/api/models")).json()
    assert payload == {"models": [], "supported": False}
    response = await windows_client.post("/api/models/quality/download")
    assert response.status_code == 400
    assert "macOS" in response.json()["detail"]


async def test_alignment_endpoints_gated(windows_client):
    status = (await windows_client.get("/api/alignment/status")).json()
    assert status["supported"] is False
    assert status["installed"] is False
    response = await windows_client.post("/api/alignment/install")
    assert response.status_code == 400
    assert "macOS" in response.json()["detail"]


async def test_native_picker_gated(windows_client):
    response = await windows_client.post("/api/media/pick")
    assert response.status_code == 400
    assert "macOS" in response.json()["detail"]


async def test_job_creation_gating(windows_client, tmp_path: Path, sine_wav: Path):
    source = tmp_path / "lecture.mp3"
    source.write_bytes(b"not-really-audio")
    base = {"sources": [{"path": str(source)}], "language": "en"}

    response = await windows_client.post(
        "/api/jobs", json={**base, "provider": "local_mlx"}
    )
    assert response.status_code == 422
    assert "macOS" in response.json()["detail"]

    response = await windows_client.post(
        "/api/jobs",
        json={
            **base,
            "provider": "openrouter",
            "openrouter_model": "openai/whisper-large-v3",
            "alignment_mode": "local_whisperx",
        },
    )
    assert response.status_code == 422
    assert "WhisperX" in response.json()["detail"]

    stored = await windows_client.put(
        "/api/settings/openrouter_key", json={"key": "sk-or-test-key"}
    )
    assert stored.status_code == 200
    base = {"sources": [{"path": str(sine_wav)}], "language": "en"}
    response = await windows_client.post(
        "/api/jobs",
        json={
            **base,
            "provider": "openrouter",
            "openrouter_model": "openai/whisper-large-v3",
            "alignment_mode": "cloud",
        },
    )
    assert response.status_code == 201, response.text
    job_id = response.json()["jobs"][0]["id"]
    assert (await windows_client.post(f"/api/jobs/{job_id}/cancel")).status_code in (200, 204)


async def test_settings_reject_local_defaults(windows_client):
    response = await windows_client.put(
        "/api/settings", json={"default_provider": "local_mlx"}
    )
    assert response.status_code == 422
    response = await windows_client.put(
        "/api/settings", json={"default_alignment_mode": "local_whisperx"}
    )
    assert response.status_code == 422
    response = await windows_client.put(
        "/api/settings", json={"default_provider": "openrouter", "default_alignment_mode": "cloud"}
    )
    assert response.status_code == 200
    settings = response.json()["settings"]
    assert settings["default_provider"] == "openrouter"
    assert settings["default_alignment_mode"] == "cloud"
