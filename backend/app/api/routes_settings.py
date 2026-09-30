"""Settings endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.api.deps import get_paths, get_settings_store
from app.services.settings_store import resolve_output_dir

router = APIRouter(prefix="/api/settings")


@router.get("")
async def get_settings(request: Request) -> dict:
    paths = get_paths(request)
    settings = get_settings_store(request).get()
    return {
        "settings": settings.model_dump(mode="json"),
        "default_output_dir": str(paths.default_output_dir),
        "resolved_output_dir": str(resolve_output_dir(settings, paths)),
        "data_dir": str(paths.data_dir),
        "logs_dir": str(paths.logs_dir),
    }


def _jsonable_errors(exc: ValidationError) -> list[dict]:
    errors = exc.errors()
    for error in errors:
        if "ctx" in error:
            error["ctx"] = {key: str(value) for key, value in error["ctx"].items()}
    return errors


@router.put("")
async def update_settings(request: Request, patch: dict) -> dict:
    try:
        settings = get_settings_store(request).update(patch)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_jsonable_errors(exc)) from exc
    paths = get_paths(request)
    return {
        "settings": settings.model_dump(mode="json"),
        "resolved_output_dir": str(resolve_output_dir(settings, paths)),
    }
