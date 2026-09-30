"""Accessors for application-scoped state."""

from __future__ import annotations

from fastapi import Request

from app.config import AppPaths
from app.db.database import Database
from app.queue.events import EventBus
from app.queue.manager import JobManager
from app.services.settings_store import SettingsStore


class ModelDownloadTracker:
    def __init__(self) -> None:
        self.states: dict[str, dict] = {}


def get_paths(request: Request) -> AppPaths:
    return request.app.state.paths


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_settings_store(request: Request) -> SettingsStore:
    return request.app.state.settings_store


def get_manager(request: Request) -> JobManager:
    return request.app.state.manager


def get_bus(request: Request) -> EventBus:
    return request.app.state.event_bus


def get_download_tracker(request: Request) -> ModelDownloadTracker:
    return request.app.state.model_downloads
