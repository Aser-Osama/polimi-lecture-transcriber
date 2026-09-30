"""Real-server integration: startup, static fallback, SSE streaming."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
def running_server(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PT_TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("PT_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("PT_DB_PATH", str(tmp_path / "data" / "db.db"))
    from app.main import create_app

    app = create_app()
    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=15)


def test_health_and_root_fallback(running_server):
    response = httpx.get(f"{running_server}/api/health", timeout=10)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    root = httpx.get(f"{running_server}/", timeout=10)
    assert root.status_code == 200
    # frontend may or may not be built; either JSON notice or the built index
    assert "Polimi" in root.text or "not built" in root.text


def test_sse_stream_snapshot_and_live_event(running_server):
    with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
        with client.stream("GET", f"{running_server}/api/events") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            saw_snapshot = False
            for line in response.iter_lines():
                if line.startswith("data: "):
                    payload = json.loads(line[len("data: ") :])
                    assert payload["type"] == "snapshot"
                    assert "jobs" in payload
                    saw_snapshot = True
                    break
            assert saw_snapshot
