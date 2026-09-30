from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.models.domain import (
    Job,
    JobConfig,
    JobStatus,
)
from app.version import DB_SCHEMA_VERSION


def make_job(job_id: str = "job1", status: JobStatus = JobStatus.QUEUED, created_offset: int = 0) -> Job:
    job = Job(
        id=job_id,
        source_path=f"/tmp/{job_id}.mp4",
        source_filename=f"{job_id}.mp4",
        size_bytes=123,
        config=JobConfig(),
        status=status,
        created_at=datetime.now(UTC) + timedelta(seconds=created_offset),
    )
    return job


def test_schema_version(database):
    with database._connect() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == DB_SCHEMA_VERSION


def test_insert_and_get_roundtrip(database):
    job = make_job()
    job.status_message = "hello"
    job.outputs = {"txt": "/tmp/x.txt"}
    database.insert_job(job)
    loaded = database.get_job("job1")
    assert loaded is not None
    assert loaded.source_filename == "job1.mp4"
    assert loaded.config.model_key == "quality"
    assert loaded.outputs["txt"] == "/tmp/x.txt"
    assert loaded.status == JobStatus.QUEUED


def test_update_transitions(database):
    job = make_job()
    database.insert_job(job)
    job.status = JobStatus.TRANSCRIBING
    job.started_at = datetime.now(UTC)
    database.update_job(job)
    loaded = database.get_job("job1")
    assert loaded.status == JobStatus.TRANSCRIBING
    assert loaded.started_at is not None


def test_list_order_newest_first(database):
    database.insert_job(make_job("older", created_offset=-10))
    database.insert_job(make_job("newer", created_offset=10))
    jobs = database.list_jobs()
    assert [j.id for j in jobs] == ["newer", "older"]


def test_find_next_queued_fifo(database):
    database.insert_job(make_job("second", created_offset=5))
    database.insert_job(make_job("first", created_offset=-5))
    database.insert_job(make_job("running", status=JobStatus.TRANSCRIBING))
    next_job = database.find_next_queued()
    assert next_job is not None and next_job.id == "first"


def test_recover_stale_jobs(database):
    database.insert_job(make_job("queued"))
    database.insert_job(make_job("active", status=JobStatus.TRANSCRIBING))
    database.insert_job(make_job("done", status=JobStatus.COMPLETED))
    recovered = database.recover_stale_jobs()
    assert set(recovered) == {"queued", "active"}
    assert database.get_job("queued").status == JobStatus.INTERRUPTED
    assert database.get_job("active").status == JobStatus.INTERRUPTED
    assert database.get_job("done").status == JobStatus.COMPLETED
    # recoverable message preserved
    assert "Retry" in database.get_job("active").status_message


def test_archive_finished(database):
    database.insert_job(make_job("done", status=JobStatus.COMPLETED))
    database.insert_job(make_job("running", status=JobStatus.TRANSCRIBING))
    assert database.archive_finished() == 1
    visible = database.list_jobs(include_archived=False)
    assert [j.id for j in visible] == ["running"]
    assert len(database.list_jobs(include_archived=True)) == 2


def test_delete_job(database):
    database.insert_job(make_job())
    assert database.delete_job("job1")
    assert database.get_job("job1") is None
    assert not database.delete_job("job1")


def test_settings_kv(database):
    assert database.get_setting("x") is None
    database.set_setting("x", "1")
    assert database.get_setting("x") == "1"
    database.set_setting("x", "2")
    assert database.get_setting("x") == "2"
