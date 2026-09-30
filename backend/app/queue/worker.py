"""Worker process entry point.

A dedicated (spawned) process runs all heavy work: FFmpeg audio extraction and
MLX Whisper inference. The queue manager talks to it through two multiprocessing
queues. Because inference cannot be interrupted mid-call, a forced cancellation
kills the worker's process group; the next job simply gets a fresh worker.

The worker is persistent: it survives between jobs so mlx-whisper's built-in
``ModelHolder`` cache keeps the model in memory across batch jobs. It exits on a
``shutdown`` command or when killed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue as std_queue
import sys
import threading
import traceback
from pathlib import Path

from app.core.errors import AppError, CancelledError
from app.models.domain import AppSettings, Job
from app.providers.base import CancellationToken

log = logging.getLogger("app.worker")


def worker_entry(paths_dict: dict, cmd_queue, event_queue, provider_options: dict | None = None) -> None:
    """Runs in the child process (must stay a module-level function for spawn)."""
    # New session/process group so killpg() reaches FFmpeg children; skipping
    # when already a session leader (e.g. running in a test thread).
    with contextlib.suppress(OSError):
        os.setsid()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s worker: %(message)s",
        stream=sys.stderr,
        force=True,
    )

    from app.config import AppPaths

    paths = AppPaths(
        data_dir=Path(paths_dict["data_dir"]),
        logs_dir=Path(paths_dict["logs_dir"]),
        temp_dir=Path(paths_dict["temp_dir"]),
        default_output_dir=Path(paths_dict["default_output_dir"]),
        db_path=Path(paths_dict["db_path"]),
    )
    options = provider_options or {}

    cancel_holder: dict[str, CancellationToken | None] = {"token": None}
    stop = threading.Event()
    run_queue: std_queue.Queue = std_queue.Queue()

    def command_reader() -> None:
        while not stop.is_set():
            try:
                command = cmd_queue.get(timeout=0.1)
            except (std_queue.Empty, EOFError, OSError):
                continue
            if not isinstance(command, dict):
                continue
            kind = command.get("type")
            if kind == "shutdown":
                stop.set()
                token = cancel_holder.get("token")
                if token is not None:
                    token.cancel()
                run_queue.put(None)
                return
            if kind == "cancel":
                token = cancel_holder.get("token")
                if token is not None:
                    token.cancel()
                event_queue.put({"type": "cancel_ack", "job_id": command.get("job_id")})
            elif kind == "run_job":
                run_queue.put(command)

    reader = threading.Thread(target=command_reader, name="worker-commands", daemon=True)
    reader.start()
    _start_parent_watchdog(stop)
    event_queue.put({"type": "ready", "pid": os.getpid()})

    while True:
        command = run_queue.get()
        if command is None:
            break
        job_payload = command["job"]
        job = Job.model_validate(job_payload)
        settings = AppSettings.model_validate(command["settings"])
        token = CancellationToken()
        cancel_holder["token"] = token
        try:
            _execute_job(paths, job, settings, token, event_queue, options)
        except CancelledError:
            log.info("Job %s cancelled", job.id)
            event_queue.put({"type": "cancelled", "job_id": job.id})
        except AppError as exc:
            log.warning("Job %s failed: %s", job.id, exc.detail)
            event_queue.put(
                {
                    "type": "failed",
                    "job_id": job.id,
                    "error": exc.user_message,
                    "detail": exc.detail,
                }
            )
        except Exception as exc:  # pragma: no cover - last-resort safety net
            traceback.print_exc()
            event_queue.put(
                {
                    "type": "failed",
                    "job_id": job.id,
                    "error": "Unexpected error during processing. See the application logs.",
                    "detail": str(exc),
                }
            )
        finally:
            cancel_holder["token"] = None
            _release_mlx_cache()

    reader.join(timeout=1)
    event_queue.put({"type": "bye", "pid": os.getpid()})


def _release_mlx_cache() -> None:
    """Return transient MLX/Metal buffers to the OS between jobs."""
    import sys

    if "mlx.core" not in sys.modules:
        return  # provider never used MLX in this worker
    import mlx.core as mx

    with contextlib.suppress(Exception):
        mx.clear_cache()


def _start_parent_watchdog(stop: threading.Event) -> None:
    """Exit if the parent process disappears (e.g. SIGKILL), so a model-loaded
    worker can never survive as an orphan."""

    original_parent = os.getppid()

    def watch() -> None:
        while not stop.is_set():
            if os.getppid() != original_parent:
                os._exit(1)
            stop.wait(1.0)

    threading.Thread(target=watch, name="worker-parent-watchdog", daemon=True).start()


def _execute_job(paths, job: Job, settings: AppSettings, token: CancellationToken, event_queue, options: dict) -> None:
    from app.services.pipeline import PipelineContext, run_pipeline
    from app.services.settings_store import resolve_output_dir

    work_dir = paths.temp_dir / "jobs" / job.id

    def progress(stage: str, message: str | None = None, fraction: float | None = None) -> None:
        payload = {"type": "stage", "job_id": job.id, "stage": stage, "message": message}
        if fraction is not None:
            payload["fraction"] = round(max(0.0, min(1.0, fraction)), 4)
        event_queue.put(payload)

    event_queue.put({"type": "started", "job_id": job.id})
    context = PipelineContext(
        job=job,
        settings=settings,
        paths=paths,
        output_dir=resolve_output_dir(settings, paths),
        work_dir=work_dir,
        progress=progress,
        cancel=token,
        provider_options=options,
    )
    try:
        outcome = run_pipeline(context)
    except BaseException:
        _cleanup_work_dir(work_dir)
        raise

    event_queue.put(
        {
            "type": "completed",
            "job_id": job.id,
            "outputs": outcome.outputs,
            "timings": outcome.timings.model_dump(),
            "warnings": outcome.warnings,
            "detected_language": outcome.detected_language,
            "media_duration": outcome.media_duration,
            "processing_duration": outcome.result.processing_duration_seconds,
            "realtime_factor": outcome.result.realtime_factor,
        }
    )
    _cleanup_work_dir(work_dir)


def _cleanup_work_dir(work_dir: Path) -> None:
    from app.config import debug_mode
    from app.services.pipeline import cleanup_work_dir

    cleanup_work_dir(work_dir, keep=debug_mode())
