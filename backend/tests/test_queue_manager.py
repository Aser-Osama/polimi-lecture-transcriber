from __future__ import annotations

import asyncio
import shutil
import threading
import time
from pathlib import Path

import pytest

from app.models.domain import (
    Job,
    JobConfig,
    JobStatus,
    LanguageChoice,
    ProviderName,
)
from app.models.result import RawSegment, RawWord
from app.providers.base import RawTranscription
from app.queue import manager as manager_module
from app.services import pipeline as pipeline_module
from tests.helpers import (
    DeadWorkerHandle,
    make_queue_env,
    wait_for_status,
    wait_until,
)


class CountingProvider:
    """Records peak concurrency and honors cancellation while 'working'."""

    name = "openrouter"
    _lock = threading.Lock()
    active = 0
    peak = 0

    def __init__(self, delay: float):
        self._delay = delay

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls.active = 0
            cls.peak = 0

    def transcribe_sync(self, request, progress, cancel):
        progress("loading_model", None)
        with type(self)._lock:
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
        try:
            deadline = time.monotonic() + self._delay
            while time.monotonic() < deadline:
                cancel.raise_if_cancelled()
                time.sleep(0.05)
            progress("transcribing", None, 1.0)
            return RawTranscription(
                text="Hello world.",
                language="en",
                segments=[
                    RawSegment(
                        id=0,
                        start=0.0,
                        end=1.0,
                        text="Hello world.",
                        words=[
                            RawWord(text="Hello", start=0.1, end=0.4),
                            RawWord(text="world.", start=0.5, end=0.9),
                        ],
                    )
                ],
            )
        finally:
            with type(self)._lock:
                type(self).active -= 1


def patch_provider(monkeypatch, delay: float) -> None:
    CountingProvider.reset()
    monkeypatch.setattr(
        pipeline_module, "create_provider", lambda name: CountingProvider(delay)
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


async def test_transcription_fraction_reaches_event_bus(paths, sine_wav):
    environment = make_queue_env(paths, provider_options={"fake_delay": 0.05})
    await environment.manager.start()
    queue = environment.bus.subscribe()
    try:
        job = make_job(paths, sine_wav)
        await environment.manager.enqueue(job)
        await wait_for_status(environment.db, job.id, {JobStatus.COMPLETED})
        fractions: list[float] = []
        while not queue.empty():
            event = queue.get_nowait()
            if event.get("type") == "job" and event.get("fraction") is not None:
                fractions.append(event["fraction"])
        assert fractions, "expected fraction events on the bus"
        assert fractions[-1] == 1.0
    finally:
        await environment.manager.stop()


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


async def test_idle_worker_exits_and_releases_memory(paths, sine_wav, monkeypatch):
    monkeypatch.setattr(manager_module, "WORKER_IDLE_SECONDS", 0.3)
    environment = make_queue_env(paths, provider_options={"fake_delay": 0.0})
    await environment.manager.start()
    try:
        first = make_job(paths, sine_wav)
        await environment.manager.enqueue(first)
        await wait_for_status(environment.db, first.id, {JobStatus.COMPLETED}, timeout=30)
        await wait_until(
            lambda: environment.manager._worker is None,
            timeout=15,
            what="idle worker shutdown",
        )
        await wait_until(
            lambda: environment.handles[0].is_alive() is False,
            timeout=15,
            what="idle worker process exit",
        )
        # A new job is processed by a fresh worker.
        second = make_job(paths, sine_wav)
        await environment.manager.enqueue(second)
        completed = await wait_for_status(
            environment.db, second.id, {JobStatus.COMPLETED}, timeout=30
        )
        assert completed.outputs
        assert len(environment.factory_calls) == 2
    finally:
        await environment.manager.stop()


async def test_batch_keeps_worker_during_idle_gap(paths, sine_wav, monkeypatch):
    monkeypatch.setattr(manager_module, "WORKER_IDLE_SECONDS", 1.0)
    environment = make_queue_env(paths, provider_options={"fake_delay": 0.0})
    await environment.manager.start()
    try:
        first = make_job(paths, sine_wav)
        second = make_job(paths, sine_wav)
        await environment.manager.enqueue(first)
        await wait_for_status(environment.db, first.id, {JobStatus.COMPLETED}, timeout=30)
        # Enqueue the next job well within the idle window; the worker must stay.
        await environment.manager.enqueue(second)
        await wait_for_status(environment.db, second.id, {JobStatus.COMPLETED}, timeout=30)
        assert len(environment.factory_calls) == 1
    finally:
        await environment.manager.stop()


def cloud_job(paths, source: Path) -> Job:
    return make_job(paths, source, provider=ProviderName.OPENROUTER, alignment_mode="none")


async def test_cloud_jobs_run_in_parallel(paths, sine_wav, monkeypatch):
    patch_provider(monkeypatch, delay=0.6)
    environment = make_queue_env(paths)
    await environment.manager.start()
    try:
        jobs = [cloud_job(paths, sine_wav) for _ in range(3)]
        for job in jobs:
            await environment.manager.enqueue(job)
        for job in jobs:
            await wait_for_status(environment.db, job.id, {JobStatus.COMPLETED}, timeout=60)
        assert CountingProvider.peak >= 2, "cloud jobs should overlap"
        assert len(environment.factory_calls) == 3, "each cloud job gets its own worker"
        assert environment.manager._worker is None, "no local worker needed for cloud jobs"
    finally:
        await environment.manager.stop()


async def test_cloud_concurrency_cap_is_respected(paths, sine_wav, monkeypatch):
    patch_provider(monkeypatch, delay=0.3)
    environment = make_queue_env(paths)
    await environment.manager.start()
    try:
        environment.settings.update({"max_parallel_cloud_jobs": 1})
        jobs = [cloud_job(paths, sine_wav) for _ in range(3)]
        for job in jobs:
            await environment.manager.enqueue(job)
        for job in jobs:
            await wait_for_status(environment.db, job.id, {JobStatus.COMPLETED}, timeout=60)
        assert CountingProvider.peak == 1
    finally:
        await environment.manager.stop()


async def test_local_jobs_stay_sequential(paths, sine_wav, monkeypatch):
    patch_provider(monkeypatch, delay=0.25)
    environment = make_queue_env(paths)
    await environment.manager.start()
    try:
        jobs = [make_job(paths, sine_wav) for _ in range(3)]  # fake provider -> local slot
        for job in jobs:
            await environment.manager.enqueue(job)
        for job in jobs:
            await wait_for_status(environment.db, job.id, {JobStatus.COMPLETED}, timeout=60)
        assert CountingProvider.peak == 1
        assert len(environment.factory_calls) == 1, "local jobs reuse one worker"
    finally:
        await environment.manager.stop()


async def test_mixed_local_and_cloud_run_together(paths, sine_wav, monkeypatch):
    patch_provider(monkeypatch, delay=0.5)
    environment = make_queue_env(paths)
    await environment.manager.start()
    try:
        jobs = [make_job(paths, sine_wav), cloud_job(paths, sine_wav), cloud_job(paths, sine_wav)]
        for job in jobs:
            await environment.manager.enqueue(job)
        for job in jobs:
            await wait_for_status(environment.db, job.id, {JobStatus.COMPLETED}, timeout=60)
        assert CountingProvider.peak >= 2
        assert len(environment.factory_calls) == 3  # 1 local + 2 cloud workers
    finally:
        await environment.manager.stop()


async def test_openrouter_with_local_whisperx_uses_local_slot(paths, sine_wav, monkeypatch):
    patch_provider(monkeypatch, delay=0.2)
    environment = make_queue_env(paths)
    await environment.manager.start()
    try:
        jobs = [
            make_job(paths, sine_wav, provider=ProviderName.OPENROUTER, alignment_mode="local_whisperx")
            for _ in range(2)
        ]
        for job in jobs:
            await environment.manager.enqueue(job)
        for job in jobs:
            await wait_for_status(environment.db, job.id, {JobStatus.COMPLETED}, timeout=60)
        assert CountingProvider.peak == 1
        assert len(environment.factory_calls) == 1
    finally:
        await environment.manager.stop()


async def test_cancel_one_parallel_cloud_job_keeps_the_other(paths, sine_wav, monkeypatch):
    patch_provider(monkeypatch, delay=2.0)
    environment = make_queue_env(paths)
    await environment.manager.start()
    try:
        first = cloud_job(paths, sine_wav)
        second = cloud_job(paths, sine_wav)
        await environment.manager.enqueue(first)
        await environment.manager.enqueue(second)
        await wait_until(
            lambda: len(environment.manager._active_jobs) == 2,
            timeout=15,
            what="both cloud jobs running",
        )
        cancelled = await environment.manager.cancel_job(first.id)
        assert cancelled.status == JobStatus.CANCELLED
        completed = await wait_for_status(
            environment.db, second.id, {JobStatus.COMPLETED}, timeout=60
        )
        assert completed.outputs
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
