"""System endpoints: health, SSE events, reveal actions."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.deps import get_bus, get_manager, get_paths
from app.api.schemas import RevealRequest
from app.services import media as media_service
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
    provider_ok = True
    provider_error = None
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
        "mlx_provider": {"ok": provider_ok, "error": provider_error},
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
                except asyncio.TimeoutError:
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
        raise HTTPException(status_code=500, detail="Could not open Finder.")
    return {"opened": str(target)}
