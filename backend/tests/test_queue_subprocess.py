"""End-to-end tests of the real spawned worker process (macOS spawn context)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.models.domain import Job, JobConfig, JobStatus, ProviderName
from app.queue import manager as manager_module
from app.queue.manager import SubprocessWorkerHandle
from tests.conftest import requires_ffmpeg
from tests.helpers import make_queue_env, wait_for_status


def make_job(paths, source: Path, **config_kwargs) -> Job:
    config_kwargs.setdefault("provider", ProviderName.FAKE)
    return Job(
        source_path=str(source),
        source_filename=source.name,
        size_bytes=source.stat().st_size,
        config=JobConfig(**config_kwargs),
    )


def subprocess_factory(paths, provider_options: dict):
    handles = []

    def factory() -> SubprocessWorkerHandle:
        handle = SubprocessWorkerHandle(paths, provider_options)
        handles.append(handle)
        return handle

    return factory, handles


@requires_ffmpeg
async def test_subprocess_worker_batch_and_reuse(paths, sine_wav):
    factory, handles = subprocess_factory(paths, {"fake_delay": 0.0})
    environment = make_queue_env(paths, handle_factory=factory)
    await environment.manager.start()
    try:
        first = make_job(paths, sine_wav)
        second = make_job(paths, sine_wav)
        await environment.manager.enqueue(first)
        await environment.manager.enqueue(second)
        await wait_for_status(environment.db, first.id, {JobStatus.COMPLETED}, timeout=90)
        await wait_for_status(environment.db, second.id, {JobStatus.COMPLETED}, timeout=90)
        assert len(handles) == 1, "worker process should be reused between batch jobs"
        assert handles[0].is_alive()
    finally:
        await environment.manager.stop()
    assert not handles[0].is_alive(), "worker must be gone after shutdown"


@requires_ffmpeg
async def test_subprocess_graceful_cancellation(paths, sine_wav):
    factory, handles = subprocess_factory(paths, {"fake_delay": 1.0})
    environment = make_queue_env(paths, handle_factory=factory)
    await environment.manager.start()
    try:
        job = make_job(paths, sine_wav)
        await environment.manager.enqueue(job)
        await wait_for_status(environment.db, job.id, {JobStatus.TRANSCRIBING}, timeout=90)
        cancelled = await environment.manager.cancel_job(job.id)
        assert cancelled.status == JobStatus.CANCELLED
        assert handles[0].is_alive(), "graceful cancel keeps the worker for model reuse"
        # A queued follow-up still succeeds with the same worker.
        second = make_job(paths, sine_wav)
        await environment.manager.enqueue(second)
        done = await wait_for_status(environment.db, second.id, {JobStatus.COMPLETED}, timeout=90)
        assert done.outputs
    finally:
        await environment.manager.stop()


@requires_ffmpeg
async def test_subprocess_force_kill(paths, sine_wav, monkeypatch):
    monkeypatch.setattr(manager_module, "CANCEL_GRACE_SECONDS", 0.5)
    factory, handles = subprocess_factory(paths, {"fake_uninterruptible_ms": 20000})
    environment = make_queue_env(paths, handle_factory=factory)
    await environment.manager.start()
    try:
        job = make_job(paths, sine_wav)
        await environment.manager.enqueue(job)
        await wait_for_status(environment.db, job.id, {JobStatus.TRANSCRIBING}, timeout=90)
        cancelled = await environment.manager.cancel_job(job.id)
        assert cancelled.status == JobStatus.CANCELLED
        assert not handles[0].is_alive(), "uninterruptible worker must be killed"
        assert environment.manager._worker is None
        # A completely fresh worker can still process the next job.
        second = make_job(paths, sine_wav)
        await environment.manager.enqueue(second)
        done = await wait_for_status(environment.db, second.id, {JobStatus.COMPLETED}, timeout=90)
        assert done.outputs
        assert len(handles) == 2
    finally:
        await environment.manager.stop()
