"""Settings endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app import capabilities
from app.api.deps import get_paths, get_settings_store
from app.api.schemas import OpenRouterKeyRequest
from app.core.errors import AppError
from app.services import secrets
from app.services.settings_store import resolve_output_dir

router = APIRouter(prefix="/api/settings")


@router.get("")
async def get_settings(request: Request) -> dict:
    paths = get_paths(request)
    settings = get_settings_store(request).get()
    try:
        key_present = await asyncio.to_thread(secrets.has_api_key)
    except AppError:
        key_present = False
    return {
        "settings": settings.model_dump(mode="json"),
        "default_output_dir": str(paths.default_output_dir),
        "resolved_output_dir": str(resolve_output_dir(settings, paths)),
        "data_dir": str(paths.data_dir),
        "logs_dir": str(paths.logs_dir),
        "openrouter_key_present": key_present,
    }


@router.put("/openrouter_key")
async def store_openrouter_key(request: Request, body: OpenRouterKeyRequest) -> dict:
    try:
        await asyncio.to_thread(secrets.store_api_key, body.key)
    except AppError as exc:
        raise HTTPException(status_code=500, detail=exc.user_message) from exc
    return {"stored": True}


@router.delete("/openrouter_key")
async def delete_openrouter_key(request: Request) -> dict:
    try:
        deleted = await asyncio.to_thread(secrets.delete_api_key)
    except AppError as exc:
        raise HTTPException(status_code=500, detail=exc.user_message) from exc
    return {"deleted": deleted}


def _jsonable_errors(exc: ValidationError) -> list[dict]:
    errors = exc.errors()
    for error in errors:
        if "ctx" in error:
            error["ctx"] = {key: str(value) for key, value in error["ctx"].items()}
    return errors


@router.put("")
async def update_settings(request: Request, patch: dict) -> dict:
    if (
        not capabilities.local_transcription_supported()
        and patch.get("default_provider") == "local_mlx"
    ):
        raise HTTPException(
            status_code=422,
            detail="Local transcription (MLX Whisper) is only available on macOS.",
        )
    if not capabilities.local_alignment_supported() and (
        patch.get("default_alignment_mode") == "local_whisperx"
        or patch.get("default_align_with_whisperx") is True
    ):
        raise HTTPException(
            status_code=422,
            detail="Local WhisperX alignment is only available on macOS.",
        )
    try:
        settings = get_settings_store(request).update(patch)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_jsonable_errors(exc)) from exc
    paths = get_paths(request)
    return {
        "settings": settings.model_dump(mode="json"),
        "resolved_output_dir": str(resolve_output_dir(settings, paths)),
    }
