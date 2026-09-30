from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import pytest

from tests.test_doc_extract import make_pdf


@pytest.fixture
async def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PT_TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("PT_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("PT_DB_PATH", str(tmp_path / "data" / "db.db"))
    monkeypatch.setenv("PT_KEYCHAIN_SERVICE", f"pt-test-{uuid.uuid4().hex[:8]}")
    from app.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=30
        ) as client:
            yield client


async def test_project_crud_roundtrip(client):
    response = await client.get("/api/projects")
    assert response.status_code == 200
    assert response.json()["projects"] == []

    response = await client.post(
        "/api/projects", json={"name": "Operating Systems", "context": "NUMA\nTLB"}
    )
    assert response.status_code == 201
    project = response.json()
    assert project["name"] == "Operating Systems"
    project_id = project["id"]

    response = await client.put(
        f"/api/projects/{project_id}", json={"context": "NUMA\nTLB\nMESI"}
    )
    assert response.status_code == 200
    assert response.json()["context"] == "NUMA\nTLB\nMESI"

    response = await client.get("/api/projects")
    assert len(response.json()["projects"]) == 1

    response = await client.delete(f"/api/projects/{project_id}")
    assert response.status_code == 200
    assert (await client.get("/api/projects")).json()["projects"] == []
    assert (await client.delete(f"/api/projects/{project_id}")).status_code == 404


async def test_project_validation(client):
    assert (await client.post("/api/projects", json={"name": "   "})).status_code == 422
    assert (await client.post("/api/projects", json={})).status_code == 422
    assert (await client.put("/api/projects/nope", json={"context": "x"})).status_code == 404
    assert (await client.get("/api/projects/nope")).status_code == 404
    assert (await client.delete("/api/projects/nope")).status_code == 404


async def test_project_document_import_txt(client):
    project_id = (
        await client.post("/api/projects", json={"name": "Architecture", "context": "NUMA"})
    ).json()["id"]
    content = b"The TLB and MESI protocol are covered by Daniele Cattaneo. TLB again."
    response = await client.post(
        f"/api/projects/{project_id}/import",
        files={"file": ("notes.txt", content, "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "TLB" in payload["added"]
    assert "MESI" in payload["added"]
    assert "Daniele Cattaneo" in payload["added"]
    project = payload["project"]
    assert "NUMA" in project["context"]
    assert "MESI" in project["context"]

    # importing the same document again adds nothing
    response = await client.post(
        f"/api/projects/{project_id}/import",
        files={"file": ("notes.txt", content, "text/plain")},
    )
    assert response.json()["added"] == []


async def test_project_document_import_pdf(client):
    project_id = (await client.post("/api/projects", json={"name": "PDF course"})).json()["id"]
    pdf = make_pdf("TLB NUMA and the MESI protocol from Daniele Cattaneo")
    response = await client.post(
        f"/api/projects/{project_id}/import",
        files={"file": ("slides.pdf", pdf, "application/pdf")},
    )
    assert response.status_code == 200
    assert "TLB" in response.json()["added"]


async def test_project_document_import_rejections(client):
    project_id = (await client.post("/api/projects", json={"name": "X"})).json()["id"]
    response = await client.post(
        f"/api/projects/{project_id}/import",
        files={"file": ("notes.docx", b"whatever", "application/octet-stream")},
    )
    assert response.status_code == 422
    response = await client.post(
        f"/api/projects/{project_id}/import",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 422
    response = await client.post(
        "/api/projects/nope/import", files={"file": ("x.txt", b"x", "text/plain")}
    )
    assert response.status_code == 404
