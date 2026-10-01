"""Model catalog, cache status and on-demand downloads."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from app import capabilities
from app.api.deps import ModelDownloadTracker, get_bus, get_download_tracker
from app.core.errors import ModelDownloadError
from app.providers.base import CancellationToken
from app.services.model_cache import download_model, is_model_cached
from app.services.openrouter_models import openrouter_model_choices
from app.services.registry import MODEL_CATALOG, get_model_spec, model_choices

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models")


def _cached_state(key: str) -> dict:
    spec = get_model_spec(key)
    return {
        "cached": is_model_cached(spec.repo_id),
        "download": None,
    }


@router.get("/openrouter")
async def list_openrouter_models() -> dict:
    return {"models": openrouter_model_choices()}


@router.get("")
async def list_models(request: Request) -> dict:
    if not capabilities.local_transcription_supported():
        return {"models": [], "supported": False}
    tracker = get_download_tracker(request)
    models = []
    for entry in model_choices():
        key = entry["key"]
        state = tracker.states.get(key, {"state": "idle"})
        models.append({**entry, "cached": is_model_cached(entry["repo_id"]), "download": state})
    return {"models": models, "supported": True}


@router.post("/{key}/download")
async def download(request: Request, key: str) -> dict:
    if not capabilities.local_transcription_supported():
        raise HTTPException(
            status_code=400,
            detail="Local Whisper models are only available on macOS. "
            "Use the OpenRouter backend on this platform.",
        )
    if key not in MODEL_CATALOG:
        raise HTTPException(status_code=404, detail="Unknown model.")
    spec = get_model_spec(key)
    tracker: ModelDownloadTracker = get_download_tracker(request)
    if tracker.states.get(key, {}).get("state") == "downloading":
        return tracker.states[key]
    if await asyncio.to_thread(is_model_cached, spec.repo_id):
        tracker.states[key] = {"state": "completed", "fraction": 1.0, "message": "Already cached"}
        return tracker.states[key]

    bus = get_bus(request)
    tracker.states[key] = {"state": "downloading", "fraction": None, "message": "Downloading model"}

    def progress(fraction: float | None, message: str) -> None:
        state = {
            "state": "downloading",
            "fraction": fraction,
            "message": message,
        }
        tracker.states[key] = state
        bus.publish_threadsafe({"type": "model_download", "key": key, **state})

    def run() -> None:
        try:
            download_model(spec.repo_id, progress=progress, cancel=CancellationToken())
        except ModelDownloadError as exc:
            tracker.states[key] = {
                "state": "failed",
                "fraction": None,
                "message": exc.user_message,
                "error": exc.detail,
            }
            bus.publish_threadsafe({"type": "model_download", "key": key, **tracker.states[key]})
            log.exception("Model download failed for %s", key)
            return
        except Exception as exc:  # pragma: no cover - defensive
            tracker.states[key] = {"state": "failed", "fraction": None, "message": str(exc)}
            bus.publish_threadsafe({"type": "model_download", "key": key, **tracker.states[key]})
            log.exception("Model download failed for %s", key)
            return
        tracker.states[key] = {"state": "completed", "fraction": 1.0, "message": "Downloaded"}
        bus.publish_threadsafe({"type": "model_download", "key": key, **tracker.states[key]})
        log.info("Model %s downloaded", spec.repo_id)

    asyncio.get_running_loop().run_in_executor(None, run)
    return tracker.states[key]
