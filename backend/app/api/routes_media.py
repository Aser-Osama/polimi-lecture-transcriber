"""Media selection, upload streaming and probing."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app import capabilities
from app.api.deps import get_paths
from app.api.schemas import PickResponse, PickResult, ProbeRequest, UploadResponse
from app.core.errors import AppError
from app.services import media as media_service
from app.utils.proc import pick_files_native

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/media")

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".aiff",
    ".aif",
    ".mpeg",
    ".mpg",
    ".ts",
    ".wma",
}

_UNSAFE_NAME = re.compile(r"[/\\:\x00-\x1f\x7f]")
_CHUNK_SIZE = 1024 * 1024


def _safe_stored_name(filename: str | None) -> str:
    name = Path(filename or "upload").name
    name = _UNSAFE_NAME.sub("-", name).strip(" .")
    return name or "upload.bin"


def _upload_dir(paths, upload_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise HTTPException(status_code=422, detail="Invalid upload id.")
    return paths.temp_dir / "uploads" / upload_id


def resolve_upload_path(paths, upload_id: str) -> Path:
    directory = _upload_dir(paths, upload_id)
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Upload not found; please add the file again.")
    files = [p for p in directory.iterdir() if p.is_file()]
    if len(files) != 1:
        raise HTTPException(status_code=404, detail="Upload is incomplete; please add the file again.")
    return files[0]


@router.post("/upload", response_model=UploadResponse)
async def upload_media(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    paths = get_paths(request)
    original_name = _safe_stored_name(file.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{extension or 'unknown'}'. "
            "Supported: MP4, MOV, MKV, WEBM, MP3, M4A, WAV, FLAC and similar.",
        )
    upload_id = uuid.uuid4().hex
    directory = paths.temp_dir / "uploads" / upload_id
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / original_name

    size = 0
    try:
        with open(destination, "wb") as handle:
            while chunk := await file.read(_CHUNK_SIZE):
                handle.write(chunk)
                size += len(chunk)
    except OSError as exc:
        _remove_upload(directory)
        raise HTTPException(
            status_code=500, detail=f"Could not store the uploaded file: {exc}"
        ) from exc
    finally:
        await file.close()

    if size == 0:
        _remove_upload(directory)
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    try:
        media_info = await asyncio.to_thread(media_service.probe_media, destination)
    except AppError as exc:
        _remove_upload(directory)
        raise HTTPException(status_code=422, detail=exc.user_message) from exc

    log.info("Uploaded %s (%d bytes) as %s", original_name, size, upload_id)
    return UploadResponse(
        upload_id=upload_id,
        path=str(destination),
        filename=original_name,
        size_bytes=size,
        media=media_info,
    )


@router.post("/pick", response_model=PickResponse)
async def pick_files(request: Request) -> PickResponse:
    if not capabilities.native_file_picker_supported():
        raise HTTPException(
            status_code=400,
            detail="The native file picker is only available on macOS. "
            "Drag and drop files or use the browser file picker instead.",
        )
    selected = await asyncio.to_thread(
        pick_files_native, "Select lecture video or audio files"
    )
    if not selected:
        return PickResponse(cancelled=True)
    results: list[PickResult] = []
    errors: list[str] = []
    for path in selected:
        try:
            media_info = await asyncio.to_thread(media_service.probe_media, path)
            results.append(
                PickResult(
                    path=str(path),
                    filename=path.name,
                    size_bytes=path.stat().st_size,
                    media=media_info,
                )
            )
        except AppError as exc:
            errors.append(f"{path.name}: {exc.user_message}")
        except OSError as exc:
            errors.append(f"{path.name}: {exc}")
    return PickResponse(files=results, errors=errors)


@router.post("/probe")
async def probe_path(request: Request, body: ProbeRequest) -> dict:
    path = Path(body.path).expanduser()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="The file does not exist.")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        log.warning("Probing file with unusual extension: %s", path.suffix)
    try:
        media_info = await asyncio.to_thread(media_service.probe_media, path)
    except AppError as exc:
        raise HTTPException(status_code=422, detail=exc.user_message) from exc
    return media_info.model_dump(mode="json")


def _remove_upload(directory: Path) -> None:
    for child in directory.glob("*"):
        child.unlink(missing_ok=True)
    directory.rmdir()
