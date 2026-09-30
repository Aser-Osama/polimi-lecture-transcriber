"""Hierarchical context: job creation merging and provider delivery."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx
import pytest

from app.models.result import RawWord
from app.providers.base import CancellationToken, TranscriptionRequest
from app.providers.openrouter import OpenRouterProvider


@pytest.fixture
async def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PT_TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("PT_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("PT_DB_PATH", str(tmp_path / "data" / "db.db"))
    monkeypatch.setenv("PT_FAKE_DELAY", "0")
    monkeypatch.setenv("PT_KEYCHAIN_SERVICE", f"pt-test-{uuid.uuid4().hex[:8]}")
    from app.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=30
        ) as client:
            yield client


async def create_job(client, sine_wav: Path, **extra) -> dict:
    body = {
        "sources": [{"path": str(sine_wav)}],
        "provider": "fake",
        "language": "en",
    }
    body.update(extra)
    response = await client.post("/api/jobs", json=body)
    assert response.status_code == 201, response.text
    return response.json()["jobs"][0]


async def test_no_context_behaves_as_before(client, sine_wav):
    job = await create_job(client, sine_wav)
    config = job["config"]
    assert config["glossary"] == ""
    assert config["global_context"] == ""
    assert config["per_file_context"] == ""
    assert config["project_id"] is None
    assert config["project_name"] is None


async def test_global_context_from_settings(client, sine_wav):
    await client.put("/api/settings", json={"glossary": "Politecnico di Milano"})
    job = await create_job(client, sine_wav)
    config = job["config"]
    assert config["global_context"] == "Politecnico di Milano"
    assert config["glossary"] == "Politecnico di Milano"


async def test_project_context_only(client, sine_wav):
    project_id = (
        await client.post("/api/projects", json={"name": "OS", "context": "NUMA, TLB"})
    ).json()["id"]
    job = await create_job(client, sine_wav, project_id=project_id)
    config = job["config"]
    assert config["project_id"] == project_id
    assert config["project_name"] == "OS"
    assert config["glossary"] == "NUMA\nTLB"


async def test_per_file_context_only(client, sine_wav):
    job = await create_job(client, sine_wav, per_file_context="guest speaker")
    assert job["config"]["per_file_context"] == "guest speaker"
    assert job["config"]["glossary"] == "guest speaker"


async def test_all_three_levels_merge_with_specific_override(client, sine_wav):
    await client.put("/api/settings", json={"glossary": "politecnico di milano\nnuma"})
    project_id = (
        await client.post("/api/projects", json={"name": "OS", "context": "NUMA\nTLB"})
    ).json()["id"]
    job = await create_job(
        client,
        sine_wav,
        project_id=project_id,
        per_file_context="TLB\nDaniele Cattaneo",
    )
    config = job["config"]
    # effective context contains all three levels, most specific spelling wins
    effective = config["glossary"]
    assert "politecnico di milano" in effective
    assert "NUMA" in effective  # project spelling overrides the lowercase global
    assert "Daniele Cattaneo" in effective
    lines = effective.splitlines()
    assert lines.index("politecnico di milano") < lines.index("NUMA")
    assert config["global_context"].startswith("politecnico")
    assert config["per_file_context"] == "TLB\nDaniele Cattaneo"


async def test_legacy_glossary_field_still_works(client, sine_wav):
    job = await create_job(client, sine_wav, glossary="legacy terms, TLB")
    assert job["config"]["per_file_context"] == "legacy terms, TLB"
    assert job["config"]["glossary"] == "legacy terms\nTLB"


async def test_unknown_project_rejected(client, sine_wav):
    response = await client.post(
        "/api/jobs",
        json={
            "sources": [{"path": str(sine_wav)}],
            "provider": "fake",
            "language": "en",
            "project_id": "does-not-exist",
        },
    )
    assert response.status_code == 422
    assert "course/project" in response.json()["detail"].lower()


# ------------------------------------------------------------------ providers


def test_openrouter_mai_sends_phrase_list_context(monkeypatch, tmp_path):
    monkeypatch.setattr("app.providers.openrouter.secrets.load_api_key", lambda: "sk-test")
    from tests.test_openrouter_provider import words_payload

    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=words_payload(0.0))

    # Single short chunk: bypass chunk planning by using a tiny duration.
    audio = tmp_path / "tiny.wav"
    audio.write_bytes(b"RIFF")  # replaced by real extraction below
    import subprocess

    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-y", str(audio),
        ],
        check=True,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)
    request = TranscriptionRequest(
        audio_path=audio,
        model_repo="microsoft/mai-transcribe-2",
        language="en",
        context_terms=["NUMA", "TLB", "Daniele Cattaneo"],
        media_duration=2.0,
    )
    result = provider.transcribe_sync(request, lambda *args, **kwargs: None, CancellationToken())

    assert calls, "no request sent"
    phrases = calls[0]["provider"]["options"]["azure"]["phraseList"]["phrases"]
    assert phrases == ["NUMA", "TLB", "Daniele Cattaneo"]
    assert result.meta["context_applied"] == "phrase_list"
    assert result.meta["context_terms"] == 3


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="ffmpeg required")
def test_openrouter_unsupported_model_warns_and_still_works(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setattr("app.providers.openrouter.secrets.load_api_key", lambda: "sk-test")
    from tests.test_openrouter_provider import words_payload

    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=words_payload(0.0))

    audio = tmp_path / "tiny.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-y", str(audio),
        ],
        check=True,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(base_url="http://mock", http_client=client)
    request = TranscriptionRequest(
        audio_path=audio,
        model_repo="openai/whisper-large-v3",
        language="en",
        context_terms=["NUMA"],
        media_duration=2.0,
    )
    result = provider.transcribe_sync(request, lambda *args, **kwargs: None, CancellationToken())

    assert calls and "provider" not in calls[0]
    assert result.segments, "transcription must still work without context support"
    assert result.meta["context_applied"] == "none"
    assert any("does not support context" in warning for warning in result.meta["warnings"])


def test_raw_words_are_available_in_openrouter_imports():
    # sanity: RawWord import path stays stable for provider mapping
    word = RawWord(text="TLB", start=0.0, end=0.5)
    assert word.text == "TLB"
