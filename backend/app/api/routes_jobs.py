"""Job lifecycle, outputs, preview and subtitle regeneration."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import ValidationError

from app import capabilities
from app.api.deps import get_db, get_manager, get_paths, get_settings_store
from app.api.routes_media import resolve_upload_path
from app.api.schemas import CreateJobsRequest, RegenerateRequest
from app.core.errors import AppError, InvalidSourceError
from app.exporters import json_exporter, srt, vtt
from app.models.domain import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    JobConfig,
    ProviderName,
)
from app.services import media as media_service
from app.services.context import resolve_context
from app.services.outputs import (
    allocate_stem,
    ensure_output_dir,
    write_text_atomic,
)
from app.services.pipeline import segmenter_options
from app.subtitles.segmenter import segment_from_segments, segment_words
from app.subtitles.validate import quantize_cues, validate_and_repair_cues
from app.version import RESULT_SCHEMA_VERSION

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/jobs")

_OUTPUT_KINDS = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "json": "application/json; charset=utf-8",
}


@router.get("")
async def list_jobs(request: Request, include_archived: bool = True, limit: int = 300) -> dict:
    limit = max(1, min(limit, 2000))
    return {"jobs": get_manager(request).jobs_snapshot(include_archived=include_archived, limit=limit)}


@router.post("", status_code=201)
async def create_jobs(request: Request, body: CreateJobsRequest) -> dict:
    paths = get_paths(request)
    manager = get_manager(request)
    db = get_db(request)
    settings = get_settings_store(request).get()

    project = None
    if body.project_id:
        project = await asyncio.to_thread(db.get_project, body.project_id)
        if project is None:
            raise HTTPException(status_code=422, detail="Unknown course/project.")
    per_file_context = (
        body.per_file_context if body.per_file_context is not None else (body.glossary or "")
    )
    resolved = resolve_context(
        settings.glossary,
        (project.id, project.name, project.context) if project else None,
        per_file_context,
    )
    alignment_mode = body.alignment_mode or (
        "local_whisperx" if body.align_with_whisperx else "none"
    )
    if not capabilities.local_transcription_supported() and body.provider == ProviderName.LOCAL_MLX:
        raise HTTPException(
            status_code=422,
            detail="Local transcription (MLX Whisper) is only available on macOS. "
            "Use the OpenRouter backend on this platform.",
        )
    if not capabilities.local_alignment_supported() and alignment_mode == "local_whisperx":
        raise HTTPException(
            status_code=422,
            detail="Local WhisperX alignment is only available on macOS. "
            "Use Cloud alignment with the OpenRouter backend on this platform.",
        )
    if alignment_mode == "cloud" and body.provider != ProviderName.OPENROUTER:
        raise HTTPException(
            status_code=422,
            detail="Cloud alignment requires the OpenRouter backend. "
            "Select OpenRouter, or use local WhisperX alignment.",
        )
    try:
        config = JobConfig(
            model_key=body.model_key,
            language=body.language,
            provider=body.provider,
            openrouter_model=body.openrouter_model,
            align_with_whisperx=alignment_mode == "local_whisperx",
            alignment_mode=alignment_mode,
            glossary=resolved.effective,
            global_context=resolved.global_context,
            per_file_context=resolved.per_file_context,
            project_id=resolved.project_id,
            project_name=resolved.project_name,
            options=body.options,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if config.provider == ProviderName.OPENROUTER:
        from app.services import secrets

        try:
            has_key = await asyncio.to_thread(secrets.has_api_key)
        except Exception:
            has_key = False
        if not has_key:
            raise HTTPException(
                status_code=422,
                detail="No OpenRouter API key is configured. Add one in Settings, then try again.",
            )

    created: list[dict] = []
    for source in body.sources:
        if source.upload_id:
            path = resolve_upload_path(paths, source.upload_id)
            is_temporary = True
        elif source.path:
            path = Path(source.path).expanduser()
            is_temporary = False
            if not path.exists() or not path.is_file():
                raise HTTPException(
                    status_code=422, detail=f"File does not exist: {path}"
                )
        else:
            raise HTTPException(status_code=422, detail="Each source needs a path or upload_id.")

        try:
            media_info = await asyncio.to_thread(media_service.probe_media, path)
        except AppError as exc:
            raise HTTPException(
                status_code=422, detail=f"{path.name}: {exc.user_message}"
            ) from exc

        if not media_info.has_audio:
            raise HTTPException(
                status_code=422,
                detail=f"{path.name}: this file has no audio track and cannot be transcribed.",
            )

        job = manager.create_job(
            source_path=path,
            source_filename=path.name,
            source_is_temporary=is_temporary,
            size_bytes=media_info.size_bytes,
            media=media_info,
            config=config.model_copy(deep=True),
        )
        await manager.enqueue(job)
        created.append(job.public_view())

    return {"jobs": created}


@router.post("/clear-finished")
async def clear_finished(request: Request) -> dict:
    count = await get_manager(request).archive_finished()
    return {"archived": count}


@router.post("/regenerate")
async def regenerate(request: Request, body: RegenerateRequest) -> dict:
    db = get_db(request)
    settings = get_settings_store(request).get()

    json_path: Path | None = None
    if body.job_id:
        job = db.get_job(body.job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        candidate = Path(job.outputs.get("json", ""))
        if candidate.is_file():
            json_path = candidate
    elif body.json_path:
        candidate = Path(body.json_path).expanduser()
        if candidate.is_file() and candidate.suffix.lower() == ".json":
            json_path = candidate
    if json_path is None:
        raise HTTPException(
            status_code=404,
            detail="No result JSON found for this job. Transcribe it first.",
        )

    try:
        result = json_exporter.parse(json_path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"This JSON file is not a valid Polimi Transcriber result: {exc}",
        ) from exc
    if result.schema_version > RESULT_SCHEMA_VERSION:
        raise HTTPException(
            status_code=422,
            detail=f"Result schema v{result.schema_version} is newer than this app supports.",
        )
    if not result.words:
        raise HTTPException(
            status_code=422,
            detail="This result has no word-level timestamps, so subtitles cannot be regenerated.",
        )

    opts = segmenter_options(settings)
    segmentation = (
        segment_words(result.words, opts, result.media_duration)
        if result.words
        else segment_from_segments(result.segments, opts, result.media_duration)
    )
    validation = validate_and_repair_cues(
        segmentation.cues,
        result.words,
        result.media_duration,
        max_line_chars=opts.max_line_chars,
        max_lines=opts.max_lines,
    )
    quantized, quant_warnings = quantize_cues(validation.cues)
    warnings = segmentation.warnings + validation.warnings + quant_warnings

    output_dir = ensure_output_dir(json_path.parent)
    if body.overwrite:
        srt_path = json_path.with_suffix(".srt")
        vtt_path = json_path.with_suffix(".vtt")
        write_text_atomic(srt_path, srt.render(quantized))
        write_text_atomic(vtt_path, vtt.render(quantized))
    else:
        stem = json_path.stem
        if stem.endswith(" - regenerated"):
            stem = stem[: -len(" - regenerated")]
        stem, output_paths = allocate_stem(
            output_dir, f"{stem} - regenerated", ("srt", "vtt")
        )
        srt_path = output_paths["srt"]
        vtt_path = output_paths["vtt"]
        write_text_atomic(srt_path, srt.render(quantized))
        write_text_atomic(vtt_path, vtt.render(quantized))

    return {
        "srt": str(srt_path),
        "vtt": str(vtt_path),
        "cues": len(quantized),
        "warnings": warnings,
    }


@router.get("/{job_id}")
async def get_job(request: Request, job_id: str) -> dict:
    job = get_db(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.public_view()


@router.post("/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str) -> dict:
    db = get_db(request)
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status in TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail=f"Job is already {job.status.value}.")
    updated = await get_manager(request).cancel_job(job_id)
    return (updated or job).public_view()


@router.post("/{job_id}/archive")
async def archive_job(request: Request, job_id: str) -> dict:
    if get_db(request).get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    archived = await get_manager(request).archive_job(job_id)
    return {"archived": archived}


@router.post("/{job_id}/reveal")
async def reveal_job(request: Request, job_id: str) -> dict:
    job = get_db(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    target: Path | None = next(
        (
            Path(path_str)
            for kind in ("txt", "srt", "vtt", "json")
            if (path_str := job.outputs.get(kind)) and Path(path_str).is_file()
        ),
        None,
    )
    if target is None:
        raise HTTPException(status_code=404, detail="No output files exist for this job.")
    from app.utils.proc import open_in_finder

    ok = await asyncio.to_thread(open_in_finder, target)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not open Finder.")
    return {"opened": str(target)}


@router.post("/{job_id}/retry")
async def retry_job(request: Request, job_id: str) -> dict:
    manager = get_manager(request)
    try:
        job = await manager.retry_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found.") from None
    except InvalidSourceError as exc:
        raise HTTPException(status_code=422, detail=exc.user_message) from exc
    return job.public_view()


@router.delete("/{job_id}")
async def delete_job(request: Request, job_id: str, delete_outputs: bool = False) -> dict:
    manager = get_manager(request)
    db = get_db(request)
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status in ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail="Job is running; cancel it first.")

    deleted_files: list[str] = []
    if delete_outputs:
        for path_str in job.outputs.values():
            path = Path(path_str)
            if path.suffix in {".txt", ".srt", ".vtt", ".json"} and path.is_file():
                try:
                    path.unlink()
                    deleted_files.append(str(path))
                except OSError:
                    log.exception("Failed to delete output %s", path)
    try:
        await manager.remove_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": True, "deleted_outputs": deleted_files}


@router.get("/{job_id}/outputs/{kind}")
async def get_output(request: Request, job_id: str, kind: str) -> FileResponse:
    if kind not in _OUTPUT_KINDS:
        raise HTTPException(status_code=404, detail="Unknown output kind.")
    job = get_db(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    path_str = job.outputs.get(kind)
    if not path_str:
        raise HTTPException(status_code=404, detail="The job has no outputs.")
    path = Path(path_str)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The output file has been moved or deleted.")
    return FileResponse(path, media_type=_OUTPUT_KINDS[kind], filename=path.name)


@router.get("/{job_id}/media")
async def stream_media(request: Request, job_id: str) -> FileResponse:
    job = get_db(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    path = Path(job.source_path)
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="The source media is no longer available at its original path.",
        )
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@router.get("/{job_id}/preview")
async def preview(request: Request, job_id: str) -> dict:
    db = get_db(request)
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    json_path = Path(job.outputs.get("json", ""))
    if not json_path.is_file():
        raise HTTPException(status_code=404, detail="No transcription result is available yet.")
    try:
        result = json_exporter.parse(json_path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Result file unreadable: {exc}") from exc
    media_path = Path(job.source_path)
    media_available = media_path.is_file()
    return {
        "job_id": job_id,
        "cues": [
            {"index": cue.index, "start": cue.start, "end": cue.end, "text": cue.text}
            for cue in result.cues
        ],
        "media_available": media_available,
        "media_url": f"/api/jobs/{job_id}/media" if media_available else None,
        "subtitles_url": f"/api/jobs/{job_id}/outputs/vtt",
        "media_duration": result.media_duration,
        "warnings": result.warnings,
        "realtime_factor": result.realtime_factor,
        "outputs": job.outputs,
    }
