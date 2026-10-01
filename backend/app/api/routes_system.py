"""System endpoints: health, SSE events, reveal actions."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app import capabilities
from app.api.deps import get_bus, get_manager, get_paths
from app.api.schemas import RevealRequest
from app.core.errors import AppError
from app.providers.base import CancellationToken
from app.services import media as media_service
from app.services import whisperx as whisperx_service
from app.services.model_cache import hf_cache_dir
from app.utils.proc import open_in_finder
from app.version import APP_VERSION, RESULT_SCHEMA_VERSION

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.get("/health")
async def health(request: Request) -> dict:
    ffmpeg_ok, ffmpeg_info = media_service.check_ffmpeg()
    local_supported = capabilities.local_transcription_supported()
    provider_ok = True
    provider_error = None
    if local_supported:
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as exc:
            provider_ok = False
            provider_error = str(exc)
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "ffmpeg": {"ok": ffmpeg_ok, "info": ffmpeg_info},
        "mlx_provider": {
            "ok": provider_ok,
            "error": provider_error,
            "supported": local_supported,
        },
        "capabilities": capabilities.capabilities(),
        "active_job_id": get_manager(request).active_job_id,
    }


@router.get("/events")
async def events(request: Request) -> StreamingResponse:
    bus = get_bus(request)
    manager = get_manager(request)
    queue = bus.subscribe()

    async def generator():
        try:
            yield _sse({"type": "snapshot", "jobs": manager.jobs_snapshot()})
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if event.get("type") == "shutdown":
                    return
                yield _sse(event)
        finally:
            bus.unsubscribe(queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/alignment/status")
async def alignment_status(request: Request) -> dict:
    status = await asyncio.to_thread(whisperx_service.whisperx_status)
    install = getattr(request.app.state, "alignment_install", None) or {"state": "idle"}
    return {**status, "install": install}


@router.post("/alignment/install")
async def alignment_install(request: Request) -> dict:
    if not capabilities.local_alignment_supported():
        raise HTTPException(
            status_code=400,
            detail="WhisperX alignment is only available on macOS. "
            "Use Cloud alignment with the OpenRouter backend.",
        )
    state = getattr(request.app.state, "alignment_install", None)
    if state is None:
        state = {"state": "idle", "message": None}
        request.app.state.alignment_install = state
    if state.get("state") == "installing":
        return dict(state)

    bus = get_bus(request)
    cancel = CancellationToken()
    request.app.state.alignment_install_cancel = cancel

    def progress(message: str, fraction: float | None) -> None:
        state.update(state="installing", message=message)
        bus.publish_threadsafe(
            {"type": "alignment_install", "state": "installing", "message": message, "fraction": fraction}
        )

    def run() -> None:
        try:
            whisperx_service.install_whisperx(progress, cancel)
        except AppError as exc:
            state.update(state="failed", message=exc.user_message)
            bus.publish_threadsafe(
                {"type": "alignment_install", "state": "failed", "message": exc.user_message}
            )
            log.exception("WhisperX installation failed")
            return
        except Exception as exc:  # pragma: no cover - defensive
            state.update(state="failed", message=str(exc))
            bus.publish_threadsafe(
                {"type": "alignment_install", "state": "failed", "message": str(exc)}
            )
            log.exception("WhisperX installation failed")
            return
        state.update(state="completed", message="WhisperX alignment is ready")
        bus.publish_threadsafe(
            {
                "type": "alignment_install",
                "state": "completed",
                "message": "WhisperX alignment is ready",
            }
        )

    state.update(state="installing", message="Starting installation")
    asyncio.get_running_loop().run_in_executor(None, run)
    return dict(state)


@router.post("/actions/reveal")
async def reveal(request: Request, body: RevealRequest) -> dict:
    paths = get_paths(request)
    targets = {
        "output_dir": paths.default_output_dir,
        "logs": paths.logs_dir,
        "model_cache": hf_cache_dir(),
        "temp": paths.temp_dir,
        "data": paths.data_dir,
    }
    target = targets.get(body.target)
    if target is None:
        raise HTTPException(status_code=422, detail=f"Unknown reveal target: {body.target}")
    if body.target == "output_dir":
        from app.api.deps import get_settings_store

        target = get_settings_store(request).resolved_output_dir(paths)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Folder does not exist: {target}")
    ok = await asyncio.to_thread(open_in_finder, target)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not open the folder.")
    return {"opened": str(target)}
