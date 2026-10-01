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
import contextlib
import logging
import multiprocessing as mp
import os
import queue as std_queue
import signal
import threading
import time
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
    ProviderName,
    utcnow,
)
from app.queue.events import EventBus
from app.queue.worker import worker_entry
from app.services.settings_store import SettingsStore
from app.utils.proc import kill_process_group, open_in_finder

log = logging.getLogger(__name__)

CANCEL_GRACE_SECONDS = 5.0
WORKER_SWEEP_INTERVAL = 0.3
DISPATCH_POLL_SECONDS = 0.2
# Keep the loaded model while a batch is running, but exit the worker (and
# release its unified memory) when no work has arrived for this long.
WORKER_IDLE_SECONDS = float(os.environ.get("PT_WORKER_IDLE_SECONDS", "60"))

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
            # join_thread() can block forever if the feeder is parked on an
            # empty queue while the child is already gone (observed as a rare
            # teardown deadlock). Buffered events are irrelevant at shutdown.
            self._cmd_queue.cancel_join_thread()
            self._cmd_queue.close()
            self.event_queue.cancel_join_thread()
            self.event_queue.close()
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
        # Local jobs share one persistent worker (MLX model reuse); cloud-only
        # jobs get an ephemeral worker each and may run in parallel.
        self._worker: WorkerHandle | None = None
        self._cloud_workers: dict[str, WorkerHandle] = {}
        self._task: asyncio.Task | None = None
        self._tasks: dict[str, asyncio.Task] = {}
        self._active_jobs: dict[str, Job] = {}
        self._cancel_requested: set[str] = set()
        self._job_done: dict[str, asyncio.Event] = {}
        self._shutting_down = False

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> list[str]:
        recovered = self._db.recover_stale_jobs()
        self._shutting_down = False
        self._task = asyncio.create_task(self._dispatcher(), name="job-dispatcher")
        log.info("Job manager started (%d stale jobs recovered)", len(recovered))
        return recovered

    async def stop(self) -> None:
        self._shutting_down = True
        log.debug("stop: %d active job(s)", len(self._active_jobs))
        for job_id in list(self._active_jobs):
            handle = self._handle_for(job_id)
            if handle is not None:
                handle.send({"type": "cancel", "job_id": job_id})
        for job_id in list(self._active_jobs):
            done_event = self._job_done.get(job_id)
            if done_event is not None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(done_event.wait(), timeout=2.5)
                if not done_event.is_set():
                    log.warning("Job %s did not stop quickly during shutdown", job_id)

        if self._task is not None:
            log.debug("stop: cancelling dispatcher")
            dispatcher = self._task
            self._task = None
            dispatcher.cancel()
            try:
                await asyncio.wait_for(dispatcher, timeout=5)
            except (asyncio.CancelledError, TimeoutError):
                if not dispatcher.done():
                    dispatcher.cancel()
                    with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                        await asyncio.wait_for(dispatcher, timeout=5)
            log.debug("stop: dispatcher done")

        tasks = list(self._tasks.values())
        log.debug("stop: cancelling %d job task(s)", len(tasks))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

        cloud_workers = list(self._cloud_workers.values())
        self._cloud_workers.clear()
        if cloud_workers:
            await asyncio.gather(
                *(asyncio.to_thread(worker.shutdown) for worker in cloud_workers)
            )

        worker, self._worker = self._worker, None
        if worker is not None:
            log.debug("stop: shutting down local worker")
            await asyncio.to_thread(worker.shutdown)
        log.debug("stop: complete")

    # ------------------------------------------------------------- queries

    @property
    def active_job_id(self) -> str | None:
        return next(iter(self._active_jobs), None)

    @property
    def active_job_ids(self) -> list[str]:
        return list(self._active_jobs)

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
        if job.status == JobStatus.QUEUED and job_id not in self._active_jobs:
            job.status = JobStatus.CANCELLED
            job.status_message = "Cancelled before start"
            job.completed_at = utcnow()
            self._db.update_job(job)
            self._publish(job)
            log.info("Queued job %s cancelled", job_id)
            return job
        if job.status in ACTIVE_STATUSES or job_id in self._active_jobs:
            handle = self._handle_for(job_id)
            if handle is not None:
                self._cancel_requested.add(job_id)
                handle.send({"type": "cancel", "job_id": job_id})
                done_event = self._job_done.get(job_id)
                if done_event is not None:
                    try:
                        await asyncio.wait_for(done_event.wait(), timeout=CANCEL_GRACE_SECONDS)
                    except TimeoutError:
                        log.warning(
                            "Worker did not stop within %.0fs; terminating process group",
                            CANCEL_GRACE_SECONDS,
                        )
                        handle.terminate_hard()
                        try:
                            await asyncio.wait_for(done_event.wait(), timeout=10)
                        except TimeoutError:
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

    def _publish(self, job: Job) -> None:
        self._bus.publish({"type": "job", "job": job.public_view()})

    async def _dispatcher(self) -> None:
        """Poll-based scheduler.

        Deliberately avoids asyncio.Event wakeups: Python's Event.wait() can
        swallow a cancellation that lands exactly when the event is set, which
        made shutdown hang. A short poll is negligible for a local app.
        """
        while True:
            try:
                self._start_ready_jobs()
                if self._tasks:
                    await asyncio.sleep(DISPATCH_POLL_SECONDS)
                elif self._worker is not None:
                    await self._wait_idle_or_shutdown()
                else:
                    await asyncio.sleep(DISPATCH_POLL_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Dispatcher error")
                await asyncio.sleep(1)

    @staticmethod
    def _uses_local_compute(job: Job) -> bool:
        """Local slot: anything that loads MLX or runs WhisperX on this Mac."""
        if job.config.provider == ProviderName.OPENROUTER:
            return job.config.effective_alignment_mode == "local_whisperx"
        return True

    def _handle_for(self, job_id: str) -> WorkerHandle | None:
        if job_id in self._cloud_workers:
            return self._cloud_workers[job_id]
        return self._worker

    def _start_ready_jobs(self) -> None:
        """Start as many queued jobs as the slot rules allow."""
        while True:
            local_busy = any(
                self._uses_local_compute(job) for job in self._active_jobs.values()
            )
            cloud_active = sum(
                1 for job in self._active_jobs.values() if not self._uses_local_compute(job)
            )
            max_cloud = max(1, int(self._settings.get().max_parallel_cloud_jobs))

            picked: Job | None = None
            for job in self._db.list_queued_jobs():
                if job.id in self._active_jobs:
                    continue
                if self._uses_local_compute(job):
                    if not local_busy:
                        picked = job
                        break
                elif cloud_active < max_cloud:
                    picked = job
                    break
            if picked is None:
                return

            self._active_jobs[picked.id] = picked
            task = asyncio.create_task(self._run_job_task(picked), name=f"job-{picked.id}")
            self._tasks[picked.id] = task
            task.add_done_callback(
                lambda _task, jid=picked.id: self._tasks.pop(jid, None)
            )
            log.info(
                "Starting job %s (%s slot, %d cloud active)",
                picked.id,
                "local" if self._uses_local_compute(picked) else "cloud",
                cloud_active,
            )

    async def _run_job_task(self, job: Job) -> None:
        """Own the worker handle for one job, then release it."""
        is_cloud = not self._uses_local_compute(job)
        handle: WorkerHandle | None = None
        try:
            if is_cloud:
                handle = self._worker_factory()
                handle.start()
                self._cloud_workers[job.id] = handle
                log.info("Started cloud worker pid=%s for job %s", handle.pid, job.id)
            else:
                handle = self._ensure_worker()
            await self._run_job(job, handle)
        finally:
            self._active_jobs.pop(job.id, None)
            if is_cloud:
                cloud = self._cloud_workers.pop(job.id, None)
                if cloud is not None:
                    await asyncio.to_thread(cloud.shutdown)
            elif self._worker is not None and not self._worker.is_alive():
                self._worker = None
            self._cancel_requested.discard(job.id)

    async def _wait_idle_or_shutdown(self) -> None:
        """Wait for new work; if none arrives, exit the worker and free RAM."""
        deadline = time.monotonic() + WORKER_IDLE_SECONDS
        while time.monotonic() < deadline:
            if self._db.find_next_queued() is not None:
                return
            await asyncio.sleep(min(0.3, max(0.05, deadline - time.monotonic())))
        worker, self._worker = self._worker, None
        if worker is not None:
            log.info(
                "Worker idle for %.0fs; shutting it down to release model memory",
                WORKER_IDLE_SECONDS,
            )
            await asyncio.to_thread(worker.shutdown)

    def _ensure_worker(self) -> WorkerHandle:
        if self._worker is None or not self._worker.is_alive():
            self._worker = self._worker_factory()
            self._worker.start()
            log.info("Started worker process pid=%s", self._worker.pid)
        return self._worker

    async def _run_job(self, job: Job, handle: WorkerHandle) -> None:
        self._job_done[job.id] = asyncio.Event()
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
                        with contextlib.suppress(asyncio.CancelledError):
                            await event_task
                        # give in-flight events a moment to arrive, then drain
                        await asyncio.sleep(0.25)
                        while True:
                            try:
                                terminal = terminal or self._apply_event(job, events.get_nowait())
                            except asyncio.QueueEmpty:
                                break
                        if terminal is None:
                            if job.id in self._cancel_requested:
                                log.info("Job %s cancelled via worker termination", job.id)
                                terminal = "cancelled"
                            else:
                                log.error("Worker for job %s died unexpectedly", job.id)
                                terminal = "crashed"
            finally:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher
        except Exception:
            log.exception("Failed to run job %s", job.id)
            job.error = "The transcription worker could not be started."
            terminal = "crashed"
        finally:
            reader_stop.set()
            fallback = "interrupted" if self._shutting_down else "crashed"
            self._finalize(job, terminal or fallback)
            self._cleanup_temp_source(job)
            done_event = self._job_done.get(job.id)
            if done_event is not None:
                done_event.set()

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
            meta = dict(event.get("provider_meta") or {})
            if event.get("alignment_provider"):
                meta["alignment_provider"] = event["alignment_provider"]
            job.provider_meta = meta
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
                # No running loop when called from tests.
                with contextlib.suppress(RuntimeError):
                    asyncio.get_running_loop().run_in_executor(None, open_in_finder, target)

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
                with contextlib.suppress(OSError):
                    parent.rmdir()
            log.info("Deleted temporary upload %s", path)
        except OSError:
            log.exception("Failed to delete temporary upload %s", path)
