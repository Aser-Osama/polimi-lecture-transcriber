from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from tests.conftest import requires_ffmpeg


@pytest.fixture
async def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PT_TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("PT_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("PT_DB_PATH", str(tmp_path / "data" / "db.db"))
    monkeypatch.setenv("PT_FAKE_DELAY", "0")
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


async def test_health(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app_version"]
    assert payload["ffmpeg"]["ok"] is True
    assert payload["mlx_provider"]["ok"] is True


async def test_settings_roundtrip(client, tmp_path):
    response = await client.get("/api/settings")
    assert response.status_code == 200
    payload = response.json()
    assert payload["settings"]["default_model_key"] == "quality"
    assert payload["resolved_output_dir"] == str(tmp_path / "out")

    custom = str(tmp_path / "custom-out")
    response = await client.put(
        "/api/settings", json={"output_dir": custom, "default_language": "it"}
    )
    assert response.status_code == 200
    assert response.json()["settings"]["default_language"] == "it"
    assert response.json()["resolved_output_dir"] == custom

    response = await client.put("/api/settings", json={"default_model_key": "nope"})
    assert response.status_code == 422


async def test_settings_subtitle_preferences(client):
    response = await client.put(
        "/api/settings",
        json={"subtitles": {"max_line_chars": 38, "max_lines": 2, "max_cue_duration": 6.5}},
    )
    assert response.status_code == 200
    assert response.json()["settings"]["subtitles"]["max_line_chars"] == 38


async def test_create_job_invalid_path(client):
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": "/definitely/not/here.mp4"}],
            "model_key": "quality",
            "language": "en",
            "provider": "fake",
        },
    )
    assert response.status_code == 422


async def test_create_job_invalid_model(client, sine_wav):
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "model_key": "gigantic",
            "language": "en",
            "provider": "fake",
        },
    )
    assert response.status_code == 422


async def test_create_job_empty_sources(client):
    response = await client.post("/api/jobs", json={"sources": []})
    assert response.status_code == 422


@requires_ffmpeg
async def test_full_job_lifecycle(client, sine_wav, tmp_path):
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "model_key": "fast",
            "language": "en",
            "provider": "fake",
            "glossary": "NUMA, TLB",
        },
    )
    assert response.status_code == 201
    job = response.json()["jobs"][0]
    job_id = job["id"]
    assert job["status"] == "queued"

    done = await poll_job(client, job_id, {"completed"})
    assert done["outputs"]["srt"].endswith(".srt")
    assert done["realtime_factor"] is not None

    response = await client.get(f"/api/jobs/{job_id}/outputs/vtt")
    assert response.status_code == 200
    assert response.text.startswith("WEBVTT")

    response = await client.get(f"/api/jobs/{job_id}/outputs/txt")
    assert response.status_code == 200

    response = await client.get(f"/api/jobs/{job_id}/media")
    assert response.status_code == 200
    assert int(response.headers["content-length"]) > 0

    response = await client.get(f"/api/jobs/{job_id}/preview")
    assert response.status_code == 200
    preview = response.json()
    assert preview["cues"]
    assert preview["media_available"] is True
    assert preview["media_url"].endswith("/media")

    payload = json.loads(Path(done["outputs"]["json"]).read_text())
    assert payload["glossary"] == "NUMA, TLB"


@requires_ffmpeg
async def test_job_list_and_invalid_id(client, sine_wav):
    response = await client.get("/api/jobs/does-not-exist")
    assert response.status_code == 404
    assert (await client.post("/api/jobs/does-not-exist/cancel")).status_code == 404
    assert (await client.post("/api/jobs/does-not-exist/retry")).status_code == 404
    assert (await client.delete("/api/jobs/does-not-exist")).status_code == 404

    response = await client.get("/api/jobs")
    assert response.status_code == 200
    assert "jobs" in response.json()


@requires_ffmpeg
async def test_cancel_running_job_and_followup(client, sine_wav):
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "provider": "fake",
            "language": "en",
            "options": {"fake_delay": 0.6},
        },
    )
    assert response.status_code == 201
    job_id = response.json()["jobs"][0]["id"]
    await poll_job(client, job_id, {"transcribing"})
    response = await client.post(f"/api/jobs/{job_id}/cancel")
    assert response.status_code == 200
    cancelled = await poll_job(client, job_id, {"cancelled"})
    assert cancelled["status"] == "cancelled"
    # cancelling again is a conflict
    assert (await client.post(f"/api/jobs/{job_id}/cancel")).status_code == 409

    # queue still works
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "provider": "fake",
            "language": "en",
        },
    )
    follow_id = response.json()["jobs"][0]["id"]
    await poll_job(client, follow_id, {"completed"})


@requires_ffmpeg
async def test_video_without_audio_rejected(client, silent_video):
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(silent_video)}],
            "provider": "fake",
            "language": "en",
        },
    )
    assert response.status_code == 422
    assert "no audio" in response.json()["detail"].lower()


async def test_upload_rejects_unsupported_extension(client):
    response = await client.post(
        "/api/media/upload",
        files={"file": ("notes.xyz", b"hello", "application/octet-stream")},
    )
    assert response.status_code == 422


@requires_ffmpeg
async def test_upload_and_transcribe_temp_file(client, sine_wav):
    content = sine_wav.read_bytes()
    response = await client.post(
        "/api/media/upload",
        files={"file": ("lecture.wav", content, "audio/wav")},
    )
    assert response.status_code == 200
    upload = response.json()
    assert upload["media"]["has_audio"] is True

    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"upload_id": upload["upload_id"]}],
            "provider": "fake",
            "language": "en",
        },
    )
    assert response.status_code == 201
    job_id = response.json()["jobs"][0]["id"]
    await poll_job(client, job_id, {"completed"})
    # temporary upload cleaned up after successful processing
    assert not Path(upload["path"]).exists()


@requires_ffmpeg
async def test_regenerate_subtitles(client, sine_wav):
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "provider": "fake",
            "language": "en",
        },
    )
    job_id = response.json()["jobs"][0]["id"]
    done = await poll_job(client, job_id, {"completed"})

    response = await client.post("/api/jobs/regenerate", json={"job_id": job_id})
    assert response.status_code == 200
    payload = response.json()
    assert Path(payload["srt"]).is_file()
    assert Path(payload["vtt"]).is_file()
    assert "regenerated" in Path(payload["srt"]).name
    assert payload["cues"] > 0
    # original outputs untouched
    assert Path(done["outputs"]["srt"]).is_file()


@requires_ffmpeg
async def test_delete_job_history_keeps_outputs_by_default(client, sine_wav):
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "provider": "fake",
            "language": "en",
        },
    )
    job_id = response.json()["jobs"][0]["id"]
    done = await poll_job(client, job_id, {"completed"})

    response = await client.delete(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    assert (await client.get(f"/api/jobs/{job_id}")).status_code == 404
    assert Path(done["outputs"]["txt"]).is_file()
    assert Path(done["outputs"]["json"]).is_file()


@requires_ffmpeg
async def test_delete_job_with_outputs(client, sine_wav):
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "provider": "fake",
            "language": "en",
        },
    )
    job_id = response.json()["jobs"][0]["id"]
    done = await poll_job(client, job_id, {"completed"})

    response = await client.delete(f"/api/jobs/{job_id}?delete_outputs=true")
    assert response.status_code == 200
    for path in done["outputs"].values():
        if path and path.endswith((".txt", ".srt", ".vtt", ".json")):
            assert not Path(path).exists()


@requires_ffmpeg
async def test_clear_finished_archives(client, sine_wav):
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "provider": "fake",
            "language": "en",
        },
    )
    job_id = response.json()["jobs"][0]["id"]
    await poll_job(client, job_id, {"completed"})

    response = await client.post("/api/jobs/clear-finished")
    assert response.status_code == 200
    assert response.json()["archived"] == 1
    response = await client.get("/api/jobs?include_archived=false")
    assert response.json()["jobs"] == []


async def test_models_catalog(client):
    response = await client.get("/api/models")
    assert response.status_code == 200
    models = response.json()["models"]
    keys = {m["key"] for m in models}
    assert keys == {"quality", "fast", "balanced"}
    for model in models:
        assert isinstance(model["cached"], bool)
        assert model["approx_size_bytes"] > 0
    assert (await client.post("/api/models/nope/download")).status_code == 404


async def test_reveal_unknown_target(client):
    response = await client.post("/api/actions/reveal", json={"target": "nope"})
    assert response.status_code == 422


# SSE streaming cannot be tested through httpx's ASGITransport (it buffers the
# full body); it is covered against a real uvicorn server in
# test_server_streaming.py.
