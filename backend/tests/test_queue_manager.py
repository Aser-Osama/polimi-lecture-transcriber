from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from app.models.domain import (
    AppSettings,
    Job,
    JobConfig,
    JobStatus,
    LanguageChoice,
    ProviderName,
)
from app.queue import manager as manager_module
from tests.helpers import (
    DeadWorkerHandle,
    InProcessWorkerHandle,
    make_queue_env,
    wait_for_status,
)


def make_job(paths, source: Path, **config_kwargs) -> Job:
    config_kwargs.setdefault("provider", ProviderName.FAKE)
    config_kwargs.setdefault("language", LanguageChoice.ENGLISH)
    return Job(
        source_path=str(source),
        source_filename=source.name,
        size_bytes=source.stat().st_size,
        config=JobConfig(**config_kwargs),
    )


@pytest.fixture
async def env(paths):
    environment = make_queue_env(paths, provider_options={"fake_delay": 0.0})
    await environment.manager.start()
    try:
        yield environment
    finally:
        await environment.manager.stop()


async def test_sequential_completion_order(env, sine_wav):
    jobs = []
    for _ in range(3):
        job = make_job(env.paths, sine_wav)
        await env.manager.enqueue(job)
        jobs.append(job)
    completed = [
        await wait_for_status(env.db, job.id, {JobStatus.COMPLETED}, timeout=60)
        for job in jobs
    ]
    times = [job.completed_at for job in completed]
    assert times == sorted(times)
    # outputs are unique per run
    names = {Path(job.outputs["txt"]).name for job in completed}
    assert len(names) == 3
    for job in completed:
        assert job.realtime_factor is not None


async def test_worker_is_reused_between_jobs(env, sine_wav):
    first = make_job(env.paths, sine_wav)
    second = make_job(env.paths, sine_wav)
    await env.manager.enqueue(first)
    await wait_for_status(env.db, first.id, {JobStatus.COMPLETED})
    await env.manager.enqueue(second)
    await wait_for_status(env.db, second.id, {JobStatus.COMPLETED})
    assert len(env.factory_calls) == 1, "worker should be reused (model reuse)"


async def test_failed_job_does_not_stop_queue(env, sine_wav):
    failing = make_job(env.paths, sine_wav, options={"fake_fail": True})
    following = make_job(env.paths, sine_wav)
    await env.manager.enqueue(failing)
    await env.manager.enqueue(following)
    failed = await wait_for_status(env.db, failing.id, {JobStatus.FAILED})
    assert failed.error
    completed = await wait_for_status(env.db, following.id, {JobStatus.COMPLETED})
    assert completed.outputs


async def test_remove_queued_job(env, sine_wav, monkeypatch):
    monkeypatch.setenv("PT_FAKE_DELAY", "")
    slow_env = make_queue_env(env.paths, provider_options={"fake_delay": 0.5})
    await slow_env.manager.start()
    try:
        first = make_job(env.paths, sine_wav)
        second = make_job(env.paths, sine_wav)
        await slow_env.manager.enqueue(first)
        await wait_for_status(
            slow_env.db, first.id, {JobStatus.TRANSCRIBING, JobStatus.PREPARING}
        )
        await slow_env.manager.enqueue(second)
        assert await slow_env.manager.remove_job(second.id)
        assert slow_env.db.get_job(second.id) is None
        await wait_for_status(slow_env.db, first.id, {JobStatus.COMPLETED})
    finally:
        await slow_env.manager.stop()


async def test_cancel_queued_job(env, sine_wav):
    slow_env = make_queue_env(env.paths, provider_options={"fake_delay": 0.4})
    await slow_env.manager.start()
    try:
        first = make_job(env.paths, sine_wav)
        second = make_job(env.paths, sine_wav)
        await slow_env.manager.enqueue(first)
        await wait_for_status(
            slow_env.db, first.id, {JobStatus.TRANSCRIBING, JobStatus.PREPARING}
        )
        await slow_env.manager.enqueue(second)
        cancelled = await slow_env.manager.cancel_job(second.id)
        assert cancelled.status == JobStatus.CANCELLED
        await wait_for_status(slow_env.db, first.id, {JobStatus.COMPLETED})
    finally:
        await slow_env.manager.stop()


async def test_retry_failed_job_after_source_returns(env, sine_wav, tmp_path):
    source = tmp_path / "temporary.mp4"
    shutil.copy(sine_wav, source)
    job = make_job(env.paths, source)
    broken = make_job(env.paths, sine_wav)
    # First attempt: corrupted source file
    source.write_bytes(b"not media")
    await env.manager.enqueue(job)
    failed = await wait_for_status(env.db, job.id, {JobStatus.FAILED})
    assert failed.status == JobStatus.FAILED
    # Restore the file and retry
    shutil.copy(sine_wav, source)
    retried = await env.manager.retry_job(job.id)
    assert retried.id != job.id
    completed = await wait_for_status(env.db, retried.id, {JobStatus.COMPLETED})
    assert completed.outputs


async def test_retry_missing_source_rejected(env, sine_wav, tmp_path):
    missing = tmp_path / "gone.mp4"
    job = Job(
        source_path=str(missing),
        source_filename="gone.mp4",
        config=JobConfig(provider=ProviderName.FAKE),
    )
    env.db.insert_job(job)
    from app.core.errors import InvalidSourceError

    with pytest.raises(InvalidSourceError):
        await env.manager.retry_job(job.id)


async def test_graceful_cancellation_and_next_job(paths, sine_wav, monkeypatch):
    environment = make_queue_env(paths, provider_options={"fake_delay": 0.45})
    await environment.manager.start()
    try:
        first = make_job(paths, sine_wav)
        second = make_job(paths, sine_wav)
        await environment.manager.enqueue(first)
        await wait_for_status(environment.db, first.id, {JobStatus.TRANSCRIBING})
        await asyncio.sleep(0.1)
        cancelled = await environment.manager.cancel_job(first.id)
        assert cancelled.status == JobStatus.CANCELLED
        await environment.manager.enqueue(second)
        completed = await wait_for_status(
            environment.db, second.id, {JobStatus.COMPLETED}, timeout=30
        )
        assert completed.outputs
        # graceful cancel keeps the worker alive (model stays loaded)
        assert environment.manager._worker is not None
    finally:
        await environment.manager.stop()


async def test_force_kill_when_uninterruptible(paths, sine_wav, monkeypatch):
    monkeypatch.setattr(manager_module, "CANCEL_GRACE_SECONDS", 0.4)
    environment = make_queue_env(
        paths, provider_options={"fake_uninterruptible_ms": 8000}
    )
    await environment.manager.start()
    try:
        first = make_job(paths, sine_wav)
        second = make_job(paths, sine_wav)
        await environment.manager.enqueue(first)
        await wait_for_status(environment.db, first.id, {JobStatus.TRANSCRIBING})
        cancelled = await environment.manager.cancel_job(first.id)
        assert cancelled.status == JobStatus.CANCELLED
        assert environment.handles[0].is_alive() is False
        assert environment.manager._worker is None
        # next queued job gets a fresh worker and succeeds
        await environment.manager.enqueue(second)
        completed = await wait_for_status(
            environment.db, second.id, {JobStatus.COMPLETED}, timeout=60
        )
        assert completed.outputs
        assert len(environment.factory_calls) == 2
    finally:
        await environment.manager.stop()


async def test_worker_crash_marks_job_failed(paths, sine_wav):
    environment = make_queue_env(
        paths, handle_factory=lambda: DeadWorkerHandle()
    )
    await environment.manager.start()
    try:
        job = make_job(paths, sine_wav)
        await environment.manager.enqueue(job)
        failed = await wait_for_status(environment.db, job.id, {JobStatus.FAILED}, timeout=15)
        assert "unexpectedly" in (failed.error or "")
    finally:
        await environment.manager.stop()


async def test_archive_finished_hides_from_queue(env, sine_wav):
    job = make_job(env.paths, sine_wav)
    await env.manager.enqueue(job)
    await wait_for_status(env.db, job.id, {JobStatus.COMPLETED})
    assert env.manager.jobs_snapshot(include_archived=False)
    assert await env.manager.archive_finished() == 1
    assert env.manager.jobs_snapshot(include_archived=False) == []


async def test_startup_marks_stale_jobs_interrupted(paths, sine_wav):
    environment = make_queue_env(paths)
    stale = make_job(paths, sine_wav)
    stale.status = JobStatus.TRANSCRIBING
    environment.db.insert_job(stale)
    queued = make_job(paths, sine_wav)
    environment.db.insert_job(queued)
    recovered = await environment.manager.start()
    try:
        assert set(recovered) == {stale.id, queued.id}
        assert environment.db.get_job(stale.id).status == JobStatus.INTERRUPTED
        assert environment.db.get_job(queued.id).status == JobStatus.INTERRUPTED
        # No automatic restart: nothing becomes active
        await asyncio.sleep(0.3)
        assert environment.manager.active_job_id is None
    finally:
        await environment.manager.stop()


async def test_temp_upload_deleted_after_success(paths, sine_wav):
    upload_dir = paths.temp_dir / "uploads" / "abc"
    upload_dir.mkdir(parents=True)
    uploaded = upload_dir / "lecture.wav"
    shutil.copy(sine_wav, uploaded)
    environment = make_queue_env(paths, provider_options={"fake_delay": 0.0})
    await environment.manager.start()
    try:
        job = Job(
            source_path=str(uploaded),
            source_filename="lecture.wav",
            source_is_temporary=True,
            config=JobConfig(provider=ProviderName.FAKE),
        )
        await environment.manager.enqueue(job)
        await wait_for_status(environment.db, job.id, {JobStatus.COMPLETED})
        assert not uploaded.exists()
        assert not upload_dir.exists()
    finally:
        await environment.manager.stop()


async def test_temp_upload_kept_when_configured(paths, sine_wav):
    upload_dir = paths.temp_dir / "uploads" / "def"
    upload_dir.mkdir(parents=True)
    uploaded = upload_dir / "keep.wav"
    shutil.copy(sine_wav, uploaded)
    environment = make_queue_env(paths, provider_options={"fake_delay": 0.0})
    await environment.manager.start()
    try:
        environment.settings.update({"keep_temp_uploads": True})
        job = Job(
            source_path=str(uploaded),
            source_filename="keep.wav",
            source_is_temporary=True,
            config=JobConfig(provider=ProviderName.FAKE),
        )
        await environment.manager.enqueue(job)
        await wait_for_status(environment.db, job.id, {JobStatus.COMPLETED})
        assert uploaded.exists()
    finally:
        await environment.manager.stop()
