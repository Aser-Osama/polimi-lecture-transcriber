"""Sequential job queue with a cancellable worker process.

Design:
- one persistent worker process (spawn context; MLX is not fork-safe) that
  reuses the loaded model across jobs via mlx-whisper's ModelHolder cache
- jobs run strictly sequentially
- cancellation: cooperative token first; if the worker does not stop within a
  grace period (e.g. a blocking MLX call), its process group is terminated and
  the next queued job gets a fresh worker
- stale queued/active jobs are marked interrupted on startup, never restarted
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import queue as std_queue
import signal
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from app.config import AppPaths, debug_mode
from app.core.errors import InvalidSourceError
from app.db.database import Database
from app.models.domain import (
    ACTIVE_STATUSES,
    STATUS_LABELS,
    TERMINAL_STATUSES,
    Job,
    JobStatus,
    utcnow,
)
from app.queue.events import EventBus
from app.queue.worker import worker_entry
from app.services.settings_store import SettingsStore
from app.utils.proc import kill_process_group, open_in_finder

log = logging.getLogger(__name__)

CANCEL_GRACE_SECONDS = 5.0
WORKER_SWEEP_INTERVAL = 0.3

_STAGE_TO_STATUS = {
    "preparing": JobStatus.PREPARING,
    "extracting_audio": JobStatus.EXTRACTING_AUDIO,
    "loading_model": JobStatus.LOADING_MODEL,
    "transcribing": JobStatus.TRANSCRIBING,
    "aligning": JobStatus.ALIGNING,
    "formatting": JobStatus.FORMATTING,
    "saving": JobStatus.SAVING,
}


class WorkerHandle(Protocol):
    event_queue: object
    pid: int | None

    def start(self) -> None: ...
    def send(self, payload: dict) -> None: ...
    def is_alive(self) -> bool: ...
    def terminate_hard(self) -> None: ...
    def shutdown(self) -> None: ...


class SubprocessWorkerHandle:
    """Real worker: a spawned process with its own session/process group."""

    def __init__(self, paths: AppPaths, provider_options: dict | None = None):
        context = mp.get_context("spawn")
        self._cmd_queue = context.Queue(maxsize=64)
        self.event_queue = context.Queue(maxsize=8000)
        self._process = context.Process(
            target=worker_entry,
            args=(paths.as_dict(), self._cmd_queue, self.event_queue, provider_options or {}),
            daemon=True,
            name="polimi-worker",
        )
        self.pid: int | None = None

    def start(self) -> None:
        self._process.start()
        self.pid = self._process.pid

    def send(self, payload: dict) -> None:
        try:
            self._cmd_queue.put(payload, timeout=10)
        except (std_queue.Full, OSError):
            log.exception("Failed to send command to worker: %s", payload.get("type"))

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def terminate_hard(self) -> None:
        if self.pid is None or not self._process.is_alive():
            return
        log.warning("Force-terminating worker process group pid=%s", self.pid)
        kill_process_group(self.pid, signal.SIGTERM)
        self._process.join(timeout=3)
        if self._process.is_alive():
            kill_process_group(self.pid, signal.SIGKILL)
            self._process.join(timeout=3)

    def shutdown(self) -> None:
        if not self._process.is_alive():
            return
        self.send({"type": "shutdown"})
        self._process.join(timeout=5)
        if self._process.is_alive():
            self.terminate_hard()
        try:
            self.event_queue.close()
            self.event_queue.join_thread()
            self._cmd_queue.close()
            self._cmd_queue.join_thread()
        except (OSError, ValueError):
            pass


class JobManager:
    def __init__(
        self,
        paths: AppPaths,
        database: Database,
        settings_store: SettingsStore,
        event_bus: EventBus,
        worker_factory: Callable[[], WorkerHandle] | None = None,
        provider_options: dict | None = None,
    ):
        self._paths = paths
        self._db = database
        self._settings = settings_store
        self._bus = event_bus
        self._provider_options = provider_options or {}
        self._worker_factory = worker_factory or (
            lambda: SubprocessWorkerHandle(paths, self._provider_options)
        )
        self._worker: WorkerHandle | None = None
        self._task: asyncio.Task | None = None
        self._wake: asyncio.Event | None = None
        self._active_job: Job | None = None
        self._cancel_requested: str | None = None
        self._job_done: asyncio.Event | None = None
        self._shutting_down = False

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> list[str]:
        recovered = self._db.recover_stale_jobs()
        self._wake = asyncio.Event()
        self._shutting_down = False
        self._task = asyncio.create_task(self._dispatcher(), name="job-dispatcher")
        log.info("Job manager started (%d stale jobs recovered)", len(recovered))
        return recovered

    async def stop(self) -> None:
        self._shutting_down = True
        active = self._active_job
        if active is not None and self._worker is not None:
            self._worker.send({"type": "cancel", "job_id": active.id})
            done_event = self._job_done
            if done_event is not None:
                try:
                    await asyncio.wait_for(done_event.wait(), timeout=2.5)
                except asyncio.TimeoutError:
                    log.warning("Active job did not stop quickly during shutdown")
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        worker, self._worker = self._worker, None
        if worker is not None:
            await asyncio.to_thread(worker.shutdown)

    # ------------------------------------------------------------- queries

    @property
    def active_job_id(self) -> str | None:
        return self._active_job.id if self._active_job else None

    def jobs_snapshot(self, include_archived: bool = True, limit: int = 300) -> list[dict]:
        return [
            job.public_view()
            for job in self._db.list_jobs(limit=limit, include_archived=include_archived)
        ]

    # ------------------------------------------------------------ mutations

    def create_job(
        self,
        *,
        source_path: Path,
        source_filename: str,
        source_is_temporary: bool,
        size_bytes: int | None,
        media,
        config,
    ) -> Job:
        return Job(
            source_path=str(source_path),
            source_filename=source_filename,
            source_is_temporary=source_is_temporary,
            size_bytes=size_bytes,
            media=media,
            config=config,
        )

    async def enqueue(self, job: Job) -> Job:
        self._db.insert_job(job)
        self._publish(job)
        self._wake_dispatcher()
        log.info("Job %s queued (%s)", job.id, job.source_filename)
        return job

    async def enqueue_many(self, jobs: list[Job]) -> list[Job]:
        for job in jobs:
            await self.enqueue(job)
        return jobs

    async def cancel_job(self, job_id: str) -> Job | None:
        job = self._db.get_job(job_id)
        if job is None:
            return None
        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.status_message = "Cancelled before start"
            job.completed_at = utcnow()
            self._db.update_job(job)
            self._publish(job)
            log.info("Queued job %s cancelled", job_id)
            return job
        if job.status in ACTIVE_STATUSES:
            if self._active_job and self._active_job.id == job_id and self._worker is not None:
                self._cancel_requested = job_id
                self._worker.send({"type": "cancel", "job_id": job_id})
                done_event = self._job_done
                if done_event is not None:
                    try:
                        await asyncio.wait_for(done_event.wait(), timeout=CANCEL_GRACE_SECONDS)
                    except asyncio.TimeoutError:
                        log.warning(
                            "Worker did not stop within %.0fs; terminating process group",
                            CANCEL_GRACE_SECONDS,
                        )
                        self._worker.terminate_hard()
                        try:
                            await asyncio.wait_for(done_event.wait(), timeout=10)
                        except asyncio.TimeoutError:
                            log.error("Job %s did not finalize after forced termination", job_id)
            return self._db.get_job(job_id)
        return job

    async def retry_job(self, job_id: str) -> Job:
        old = self._db.get_job(job_id)
        if old is None:
            raise KeyError(job_id)
        source = Path(old.source_path)
        if not source.is_file():
            raise InvalidSourceError(
                f"Source file for retry no longer exists: {source}",
                user_message="The original media file is no longer available at its saved path.",
            )
        new_job = Job(
            source_path=old.source_path,
            source_filename=old.source_filename,
            source_is_temporary=old.source_is_temporary,
            size_bytes=old.size_bytes,
            media=old.media,
            config=old.config,
            status_message=f"Retry of job {old.id}",
        )
        await self.enqueue(new_job)
        return new_job

    async def remove_job(self, job_id: str) -> bool:
        job = self._db.get_job(job_id)
        if job is None:
            return False
        if job.status in ACTIVE_STATUSES:
            raise ValueError("Cannot remove a running job; cancel it first.")
        self._db.delete_job(job_id)
        self._bus.publish({"type": "job_removed", "job_id": job_id})
        return True

    async def archive_finished(self) -> int:
        count = self._db.archive_finished()
        self._bus.publish({"type": "refresh"})
        return count

    async def archive_job(self, job_id: str) -> bool:
        archived = self._db.set_archived(job_id, True)
        if archived:
            self._bus.publish({"type": "refresh"})
        return archived

    # ------------------------------------------------------------ internals

    def _wake_dispatcher(self) -> None:
        if self._wake is not None:
            self._wake.set()

    def _publish(self, job: Job) -> None:
        self._bus.publish({"type": "job", "job": job.public_view()})

    async def _dispatcher(self) -> None:
        while True:
            try:
                job = self._db.find_next_queued()
                if job is None:
                    assert self._wake is not None
                    await self._wake.wait()
                    self._wake.clear()
                    continue
                await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Dispatcher error")
                await asyncio.sleep(1)

    def _ensure_worker(self) -> WorkerHandle:
        if self._worker is None or not self._worker.is_alive():
            self._worker = self._worker_factory()
            self._worker.start()
            log.info("Started worker process pid=%s", self._worker.pid)
        return self._worker

    async def _run_job(self, job: Job) -> None:
        assert self._wake is not None
        self._active_job = job
        self._cancel_requested = None
        self._job_done = asyncio.Event()
        job.status = JobStatus.PREPARING
        job.status_message = STATUS_LABELS[JobStatus.PREPARING]
        job.started_at = utcnow()
        self._db.update_job(job)
        self._publish(job)

        events: asyncio.Queue = asyncio.Queue()
        reader_stop = threading.Event()
        loop = asyncio.get_running_loop()
        terminal: str | None = None

        try:
            handle = self._ensure_worker()
            handle.send(
                {
                    "type": "run_job",
                    "job": job.model_dump(mode="json"),
                    "settings": self._settings.get().model_dump(mode="json"),
                }
            )
            reader = threading.Thread(
                target=self._reader_loop,
                args=(handle, events, reader_stop, loop),
                name=f"worker-events-{job.id}",
                daemon=True,
            )
            reader.start()
            watcher = asyncio.create_task(self._watch_worker(handle, reader_stop))
            try:
                while terminal is None:
                    event_task = asyncio.create_task(events.get())
                    done, _ = await asyncio.wait(
                        {event_task, watcher}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if event_task in done:
                        terminal = self._apply_event(job, event_task.result())
                    else:
                        event_task.cancel()
                        try:
                            await event_task
                        except asyncio.CancelledError:
                            pass
                        # give in-flight events a moment to arrive, then drain
                        await asyncio.sleep(0.25)
                        while True:
                            try:
                                terminal = terminal or self._apply_event(job, events.get_nowait())
                            except asyncio.QueueEmpty:
                                break
                        if terminal is None:
                            if self._cancel_requested == job.id:
                                log.info("Job %s cancelled via worker termination", job.id)
                                terminal = "cancelled"
                            else:
                                log.error("Worker for job %s died unexpectedly", job.id)
                                terminal = "crashed"
            finally:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
        except Exception as exc:
            log.exception("Failed to run job %s", job.id)
            job.error = "The transcription worker could not be started."
            terminal = "crashed"
        finally:
            reader_stop.set()
            fallback = "interrupted" if self._shutting_down else "crashed"
            self._finalize(job, terminal or fallback)
            self._cleanup_temp_source(job)
            self._active_job = None
            self._cancel_requested = None
            if self._job_done is not None:
                self._job_done.set()
            worker = self._worker
            if worker is not None and not worker.is_alive():
                self._worker = None

    @staticmethod
    def _reader_loop(handle, events: asyncio.Queue, stop: threading.Event, loop) -> None:
        while not stop.is_set():
            try:
                event = handle.event_queue.get(timeout=0.2)
            except (std_queue.Empty, EOFError, OSError, ValueError):
                continue
            try:
                loop.call_soon_threadsafe(events.put_nowait, event)
            except RuntimeError:
                return

    async def _watch_worker(self, handle, stop: threading.Event) -> None:
        while not stop.is_set():
            if not handle.is_alive():
                return
            await asyncio.sleep(WORKER_SWEEP_INTERVAL)

    def _apply_event(self, job: Job, event: dict) -> str | None:
        kind = event.get("type")
        if kind in ("ready", "bye", "cancel_ack"):
            return None
        event_job_id = event.get("job_id")
        if event_job_id and event_job_id != job.id:
            return None
        if kind == "started":
            job.status = JobStatus.PREPARING
            job.status_message = STATUS_LABELS[JobStatus.PREPARING]
            if job.started_at is None:
                job.started_at = utcnow()
            self._db.update_job(job)
            self._publish(job)
            return None
        if kind == "stage":
            status = _STAGE_TO_STATUS.get(event.get("stage"))
            if status is not None:
                if job.status != status:
                    job.status = status
                    job.status_message = event.get("message") or STATUS_LABELS[status]
                    self._db.update_job(job)
                elif event.get("message"):
                    job.status_message = event["message"]
                payload = {"type": "job", "job": job.public_view()}
                if event.get("fraction") is not None:
                    payload["fraction"] = event["fraction"]
                self._bus.publish(payload)
            return None
        if kind == "completed":
            job.outputs = event.get("outputs", {})
            job.timings = {k: float(v) for k, v in (event.get("timings") or {}).items()}
            job.detected_language = event.get("detected_language")
            if event.get("media_duration") and job.media:
                job.media.duration_seconds = event["media_duration"]
            job.processing_duration = event.get("processing_duration")
            job.realtime_factor = event.get("realtime_factor")
            return "completed"
        if kind == "failed":
            job.error = event.get("error") or "Processing failed."
            log.error("Job %s failed: %s | %s", job.id, event.get("error"), event.get("detail"))
            return "failed"
        if kind == "cancelled":
            return "cancelled"
        log.debug("Ignoring unknown worker event: %s", kind)
        return None

    def _finalize(self, job: Job, terminal: str) -> None:
        if terminal == "completed":
            job.status = JobStatus.COMPLETED
            job.status_message = STATUS_LABELS[JobStatus.COMPLETED]
        elif terminal == "cancelled":
            job.status = JobStatus.CANCELLED
            job.status_message = "Cancelled"
        elif terminal in ("failed", "crashed"):
            job.status = JobStatus.FAILED
            if not job.error:
                job.error = "The transcription worker stopped unexpectedly."
        else:
            job.status = JobStatus.INTERRUPTED
            job.status_message = "Interrupted. Retry to process it."
        if job.status in TERMINAL_STATUSES and job.completed_at is None:
            job.completed_at = utcnow()
        self._db.update_job(job)
        self._publish(job)
        log.info("Job %s -> %s", job.id, job.status.value)

        if job.status == JobStatus.COMPLETED and self._settings.get().reveal_outputs_on_finish:
            target = next(
                (
                    Path(path)
                    for path in job.outputs.values()
                    if path and Path(path).is_file()
                ),
                None,
            )
            if target is not None:
                try:
                    asyncio.get_running_loop().run_in_executor(None, open_in_finder, target)
                except RuntimeError:  # no running loop (tests)
                    pass

    def _cleanup_temp_source(self, job: Job) -> None:
        if not job.source_is_temporary:
            return
        if debug_mode() or self._settings.get().keep_temp_uploads:
            return
        path = Path(job.source_path)
        try:
            if not path.is_relative_to(self._paths.temp_dir):
                log.warning("Refusing to delete temp source outside temp dir: %s", path)
                return
            path.unlink(missing_ok=True)
            parent = path.parent
            if parent.is_relative_to(self._paths.temp_dir):
                try:
                    parent.rmdir()
                except OSError:
                    pass
            log.info("Deleted temporary upload %s", path)
        except OSError:
            log.exception("Failed to delete temporary upload %s", path)
