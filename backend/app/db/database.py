"""SQLite persistence (metadata only; no media blobs).

One small schema, initialized/migrated via PRAGMA user_version. A simple
threading lock serializes access; all writes happen through JobManager or the
API, so contention is negligible for a local single-user app.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.models.domain import Job, JobStatus, Project
from app.version import DB_SCHEMA_VERSION

log = logging.getLogger(__name__)

_ACTIVE_OR_QUEUED = (
    JobStatus.QUEUED.value,
    JobStatus.PREPARING.value,
    JobStatus.EXTRACTING_AUDIO.value,
    JobStatus.LOADING_MODEL.value,
    JobStatus.TRANSCRIBING.value,
    JobStatus.ALIGNING.value,
    JobStatus.FORMATTING.value,
    JobStatus.SAVING.value,
)


class Database:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            self._migrate(conn)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 1:
            conn.executescript(
                """
                BEGIN;
                ALTER TABLE jobs ADD COLUMN provider_meta_json TEXT NOT NULL DEFAULT '{}';
                PRAGMA user_version = 2;
                COMMIT;
                """
            )
            version = 2
        if version == 2:
            conn.executescript(
                """
                BEGIN;
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    context TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                PRAGMA user_version = 3;
                COMMIT;
                """
            )
            version = 3
        if version == 0:
            conn.executescript(
                f"""
                BEGIN;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    source_is_temporary INTEGER NOT NULL DEFAULT 0,
                    size_bytes INTEGER,
                    media_json TEXT,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_message TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    processing_duration REAL,
                    realtime_factor REAL,
                    detected_language TEXT,
                    error TEXT,
                    outputs_json TEXT NOT NULL DEFAULT '{{}}',
                    timings_json TEXT NOT NULL DEFAULT '{{}}',
                    provider_meta_json TEXT NOT NULL DEFAULT '{{}}',
                    archived INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                    ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    context TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                PRAGMA user_version = {DB_SCHEMA_VERSION};
                COMMIT;
                """
            )
        elif version > DB_SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {version} is newer than this app supports "
                f"({DB_SCHEMA_VERSION})."
            )

    # ------------------------------------------------------------------ jobs

    def insert_job(self, job: Job) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, source_path, source_filename, source_is_temporary, size_bytes,
                    media_json, config_json, status, status_message, created_at,
                    started_at, completed_at, processing_duration, realtime_factor,
                    detected_language, error, outputs_json, timings_json,
                    provider_meta_json, archived
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                self._row_values(job),
            )

    def update_job(self, job: Job) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    source_path=?, source_filename=?, source_is_temporary=?, size_bytes=?,
                    media_json=?, config_json=?, status=?, status_message=?, created_at=?,
                    started_at=?, completed_at=?, processing_duration=?, realtime_factor=?,
                    detected_language=?, error=?, outputs_json=?, timings_json=?,
                    provider_meta_json=?, archived=?
                WHERE id=?
                """,
                (*self._row_values(job)[1:], job.id),
            )

    def _row_values(self, job: Job) -> tuple:
        return (
            job.id,
            job.source_path,
            job.source_filename,
            1 if job.source_is_temporary else 0,
            job.size_bytes,
            job.media.model_dump_json() if job.media else None,
            job.config.model_dump_json(),
            job.status.value,
            job.status_message,
            job.created_at.isoformat(),
            job.started_at.isoformat() if job.started_at else None,
            job.completed_at.isoformat() if job.completed_at else None,
            job.processing_duration,
            job.realtime_factor,
            job.detected_language,
            job.error,
            json.dumps(job.outputs),
            json.dumps(job.timings),
            json.dumps(job.provider_meta),
            1 if job.archived else 0,
        )

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        columns = set(row.keys())
        payload = {
            "id": row["id"],
            "source_path": row["source_path"],
            "source_filename": row["source_filename"],
            "source_is_temporary": bool(row["source_is_temporary"]),
            "size_bytes": row["size_bytes"],
            "media": json.loads(row["media_json"]) if row["media_json"] else None,
            "config": json.loads(row["config_json"]),
            "status": row["status"],
            "status_message": row["status_message"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "processing_duration": row["processing_duration"],
            "realtime_factor": row["realtime_factor"],
            "detected_language": row["detected_language"],
            "error": row["error"],
            "outputs": json.loads(row["outputs_json"] or "{}"),
            "timings": json.loads(row["timings_json"] or "{}"),
            # sqlite3.Row supports keys() but not "in" for column names.
            "provider_meta": json.loads(row["provider_meta_json"] or "{}")
            if "provider_meta_json" in columns
            else {},
            "archived": bool(row["archived"]),
        }
        return Job.model_validate(payload)

    def get_job(self, job_id: str) -> Job | None:
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, limit: int = 300, include_archived: bool = True) -> list[Job]:
        query = "SELECT * FROM jobs"
        if not include_archived:
            query += " WHERE archived=0"
        query += " ORDER BY created_at DESC LIMIT ?"
        with self._lock, self._connection() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
        return [self._row_to_job(row) for row in rows]

    def find_next_queued(self) -> Job | None:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at ASC LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def delete_job(self, job_id: str) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return cursor.rowcount > 0

    def recover_stale_jobs(self) -> list[str]:
        """Mark queued/active jobs as interrupted after an unclean shutdown."""
        now = datetime.now(UTC).isoformat()
        placeholders = ",".join("?" for _ in _ACTIVE_OR_QUEUED)
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                f"SELECT id FROM jobs WHERE status IN ({placeholders})",
                _ACTIVE_OR_QUEUED,
            ).fetchall()
            ids = [row["id"] for row in rows]
            if ids:
                conn.execute(
                    f"""
                    UPDATE jobs SET status=?, status_message=?,
                        completed_at=COALESCE(completed_at, ?),
                        error=COALESCE(error, ?)
                    WHERE status IN ({placeholders})
                    """,
                    (
                        JobStatus.INTERRUPTED.value,
                        "The application closed before this job finished. Retry to process it.",
                        now,
                        "Interrupted by application shutdown.",
                        *_ACTIVE_OR_QUEUED,
                    ),
                )
        if ids:
            log.info("Recovered %d interrupted job(s) at startup", len(ids))
        return ids

    def archive_finished(self) -> int:
        terminal = (
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.INTERRUPTED.value,
        )
        placeholders = ",".join("?" for _ in terminal)
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                f"UPDATE jobs SET archived=1 WHERE archived=0 AND status IN ({placeholders})",
                terminal,
            )
        return cursor.rowcount

    def set_archived(self, job_id: str, archived: bool = True) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET archived=? WHERE id=?", (1 if archived else 0, job_id)
            )
        return cursor.rowcount > 0

    # -------------------------------------------------------------- projects

    def insert_project(self, project: Project) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, context, created_at, updated_at) VALUES (?,?,?,?,?)",
                (
                    project.id,
                    project.name,
                    project.context,
                    project.created_at.isoformat(),
                    project.updated_at.isoformat(),
                ),
            )

    def update_project(self, project: Project) -> None:
        project.updated_at = datetime.now(UTC)
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE projects SET name=?, context=?, updated_at=? WHERE id=?",
                (
                    project.name,
                    project.context,
                    project.updated_at.isoformat(),
                    project.id,
                ),
            )

    def get_project(self, project_id: str) -> Project | None:
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        return self._row_to_project(row) if row else None

    def list_projects(self) -> list[Project]:
        with self._lock, self._connection() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY name COLLATE NOCASE").fetchall()
        return [self._row_to_project(row) for row in rows]

    def delete_project(self, project_id: str) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> Project:
        return Project.model_validate(
            {
                "id": row["id"],
                "name": row["name"],
                "context": row["context"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    # -------------------------------------------------------------- settings

    def get_setting(self, key: str) -> str | None:
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
