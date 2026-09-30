"""Test doubles and async helpers."""

from __future__ import annotations

import asyncio
import queue as std_queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import AppPaths
from app.db.database import Database
from app.models.domain import Job, JobStatus
from app.queue.events import EventBus
from app.queue.manager import JobManager, WorkerHandle
from app.services.settings_store import SettingsStore


class InProcessWorkerHandle:
    """WorkerHandle that runs worker_entry in a thread (no process spawn)."""

    def __init__(self, paths: AppPaths, provider_options: dict | None = None):
        self.event_queue: std_queue.Queue = std_queue.Queue(maxsize=8000)
        self._cmd_queue: std_queue.Queue = std_queue.Queue(maxsize=64)
        self.pid: int | None = None
        self._thread: threading.Thread | None = None
        self._killed = False
        self._args = (
            paths.as_dict(),
            self._cmd_queue,
            self.event_queue,
            provider_options or {},
        )

    def start(self) -> None:
        from app.queue.worker import worker_entry

        self._thread = threading.Thread(
            target=worker_entry, args=self._args, daemon=True, name="inproc-worker"
        )
        self._thread.start()

    def send(self, payload: dict) -> None:
        self._cmd_queue.put(payload)

    def is_alive(self) -> bool:
        if self._killed:
            return False
        return self._thread is not None and self._thread.is_alive()

    def terminate_hard(self) -> None:
        self._killed = True

    def shutdown(self) -> None:
        if not self._killed:
            self._cmd_queue.put({"type": "shutdown"})
        if self._thread is not None:
            self._thread.join(timeout=5)


class DeadWorkerHandle:
    """Worker that dies immediately (simulates a crash)."""

    def __init__(self, paths: AppPaths | None = None, provider_options: dict | None = None):
        self.event_queue: std_queue.Queue = std_queue.Queue()
        self.pid = None
        self._alive = False

    def start(self) -> None:
        self._alive = False

    def send(self, payload: dict) -> None:
        pass

    def is_alive(self) -> bool:
        return self._alive

    def terminate_hard(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


@dataclass
class QueueEnv:
    paths: AppPaths
    db: Database
    settings: SettingsStore
    bus: EventBus
    manager: JobManager
    handles: list[WorkerHandle]
    factory_calls: list[int]


def make_queue_env(
    paths: AppPaths,
    handle_factory=None,
    provider_options: dict | None = None,
) -> QueueEnv:
    db = Database(paths.db_path)
    settings = SettingsStore(db)
    bus = EventBus()
    handles: list[WorkerHandle] = []
    factory_calls: list[int] = []

    def default_factory() -> WorkerHandle:
        factory_calls.append(1)
        handle = InProcessWorkerHandle(paths, provider_options)
        handles.append(handle)
        return handle

    manager = JobManager(
        paths,
        db,
        settings,
        bus,
        worker_factory=handle_factory or default_factory,
        provider_options=provider_options,
    )
    return QueueEnv(paths, db, settings, bus, manager, handles, factory_calls)


async def wait_for_status(
    db: Database,
    job_id: str,
    statuses: set[JobStatus],
    timeout: float = 30.0,
) -> Job:
    deadline = time.monotonic() + timeout
    last: Job | None = None
    while time.monotonic() < deadline:
        last = db.get_job(job_id)
        if last is not None and last.status in statuses:
            return last
        await asyncio.sleep(0.03)
    raise AssertionError(
        f"Job {job_id} did not reach {statuses} in {timeout}s (last: {last.status if last else None})"
    )


async def wait_for_path(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"{path} was not created")


async def wait_until(predicate, timeout: float = 10.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"{what} was not met within {timeout}s")
