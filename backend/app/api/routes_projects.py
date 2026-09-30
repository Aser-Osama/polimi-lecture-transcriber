"""Course/project context CRUD and document term import."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.api.deps import get_db
from app.api.schemas import ProjectCreate, ProjectUpdate
from app.core.errors import AppError
from app.models.domain import Project
from app.services.doc_extract import (
    MAX_UPLOAD_BYTES,
    extract_document_text,
    extract_terms,
    merge_terms_into_context,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects")


def _view(project: Project) -> dict:
    return project.model_dump(mode="json")


@router.get("")
async def list_projects(request: Request) -> dict:
    projects = await asyncio.to_thread(get_db(request).list_projects)
    return {"projects": [_view(project) for project in projects]}


@router.post("", status_code=201)
async def create_project(request: Request, body: ProjectCreate) -> dict:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="The name cannot be empty.")
    project = Project(name=name, context=body.context.strip())
    await asyncio.to_thread(get_db(request).insert_project, project)
    log.info("Created course/project %s (%s)", name, project.id)
    return _view(project)


@router.get("/{project_id}")
async def get_project(request: Request, project_id: str) -> dict:
    project = await asyncio.to_thread(get_db(request).get_project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Course/project not found.")
    return _view(project)


@router.put("/{project_id}")
async def update_project(request: Request, project_id: str, body: ProjectUpdate) -> dict:
    db = get_db(request)
    project = await asyncio.to_thread(db.get_project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Course/project not found.")
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="The name cannot be empty.")
        project.name = name
    if body.context is not None:
        project.context = body.context.strip()
    await asyncio.to_thread(db.update_project, project)
    return _view(project)


@router.delete("/{project_id}")
async def delete_project(request: Request, project_id: str) -> dict:
    deleted = await asyncio.to_thread(get_db(request).delete_project, project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Course/project not found.")
    return {"deleted": True}


@router.post("/{project_id}/import")
async def import_document(request: Request, project_id: str, file: UploadFile = File(...)) -> dict:
    db = get_db(request)
    project = await asyncio.to_thread(db.get_project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Course/project not found.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail="The document is larger than 20 MB.")
    if not data:
        raise HTTPException(status_code=422, detail="The uploaded document is empty.")

    try:
        text = await asyncio.to_thread(extract_document_text, file.filename or "document.txt", data)
        terms = await asyncio.to_thread(extract_terms, text)
    except AppError as exc:
        raise HTTPException(status_code=422, detail=exc.user_message) from exc

    merged, added = merge_terms_into_context(project.context, terms)
    if added:
        project.context = merged
        await asyncio.to_thread(db.update_project, project)
    log.info("Imported %s into project %s: %d new term(s)", file.filename, project_id, len(added))
    return {
        "project": _view(project),
        "extracted": len(terms),
        "added": added,
        "text_chars": len(text),
    }
