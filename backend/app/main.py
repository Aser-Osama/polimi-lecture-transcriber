"""FastAPI application entry point.

Serves the API and (when built) the React frontend from the same origin.
Binds to 127.0.0.1 only; override with PT_HOST/PT_PORT if needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import (
    routes_jobs,
    routes_media,
    routes_models,
    routes_settings,
    routes_system,
)
from app.api.deps import ModelDownloadTracker
from app.config import AppPaths, debug_mode, default_host, default_port
from app.core.logging import configure_logging, log_environment
from app.db.database import Database
from app.queue.events import EventBus
from app.queue.manager import JobManager
from app.services.settings_store import SettingsStore
from app.version import APP_NAME, APP_VERSION

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def provider_options_from_env() -> dict:
    options: dict = {}
    raw_delay = os.environ.get("PT_FAKE_DELAY", "").strip()
    if raw_delay:
        try:
            options["fake_delay"] = float(raw_delay)
        except ValueError:
            log.warning("Ignoring invalid PT_FAKE_DELAY=%r", raw_delay)
    raw_uninterruptible = os.environ.get("PT_FAKE_UNINTERRUPTIBLE_MS", "").strip()
    if raw_uninterruptible:
        try:
            options["fake_uninterruptible_ms"] = float(raw_uninterruptible)
        except ValueError:
            log.warning("Ignoring invalid PT_FAKE_UNINTERRUPTIBLE_MS=%r", raw_uninterruptible)
    return options


def cleanup_stale_temp(temp_dir: Path) -> None:
    """Remove leftover per-job work dirs from unclean shutdowns."""
    jobs_dir = temp_dir / "jobs"
    if jobs_dir.is_dir():
        for child in jobs_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        log.info("Cleaned stale temporary job directories in %s", jobs_dir)


def create_app() -> FastAPI:
    paths = AppPaths.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        paths.ensure_directories()
        configure_logging(paths.logs_dir, debug_mode())
        log_environment()
        cleanup_stale_temp(paths.temp_dir)

        database = Database(paths.db_path)
        settings_store = SettingsStore(database)
        event_bus = EventBus()
        event_bus.bind_loop(asyncio.get_running_loop())
        manager = JobManager(
            paths,
            database,
            settings_store,
            event_bus,
            provider_options=provider_options_from_env(),
        )

        app.state.paths = paths
        app.state.db = database
        app.state.settings_store = settings_store
        app.state.event_bus = event_bus
        app.state.manager = manager
        app.state.model_downloads = ModelDownloadTracker()

        await manager.start()
        log.info("Application ready on http://%s:%s", default_host(), default_port())
        try:
            yield
        finally:
            event_bus.close_all()
            await manager.stop()
            log.info("Application stopped")

    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )

    app.include_router(routes_system.router)
    app.include_router(routes_media.router)
    app.include_router(routes_jobs.router)
    app.include_router(routes_settings.router)
    app.include_router(routes_models.router)

    dist_dir = PROJECT_ROOT / "frontend" / "dist"
    if (dist_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")
    else:

        @app.get("/")
        async def frontend_missing() -> dict:
            return {
                "message": (
                    "Frontend is not built yet. Run ./setup.sh (or ./dev.sh for "
                    "development mode), then open this URL again."
                ),
                "api_docs": "/api/docs",
            }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = default_host()
    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "PT_HOST=%s is not a loopback address; the app will be reachable "
            "from your network. This is not the default.",
            host,
        )
    # Pass the app object itself: with "app.main:app" uvicorn would re-import
    # the module, creating a second instance whose state (event bus) is never
    # populated when started via `python -m app.main`.
    config = uvicorn.Config(
        app,
        host=host,
        port=default_port(),
        log_level="info",
        access_log=False,
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)

    # uvicorn waits for open connections before running lifespan shutdown, so an
    # open SSE stream (the browser UI) would stall Ctrl+C. Close SSE streams as
    # soon as a shutdown signal arrives.
    original_handle_exit = server.handle_exit

    def handle_exit(sig, frame) -> None:
        log.info("Shutdown signal received (%s); closing live streams", sig)
        bus = getattr(app.state, "event_bus", None)
        loop = getattr(bus, "_loop", None) if bus is not None else None
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(bus.close_all)
            except RuntimeError:
                pass
        original_handle_exit(sig, frame)

    server.handle_exit = handle_exit  # type: ignore[method-assign]
    server.run()


if __name__ == "__main__":
    sys.exit(main())
