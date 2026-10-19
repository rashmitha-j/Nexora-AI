"""Worker job manager and HTTP API, with a fake runner instead of MetaGPT."""

import asyncio
import importlib
import io
import json
import zipfile

import pytest

from worker.jobs import JobManager, QueueFullError, collect_files


async def ok_runner(job, job_dir):
    out = job_dir / "workspace" / "app"
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.py").write_text("print('hi')\n")
    (job_dir / "workspace" / "storage").mkdir()
    (job_dir / "workspace" / "storage" / "team.json").write_text("{}")
    return {"cost_usd": 0.01}


async def failing_runner(job, job_dir):
    raise RuntimeError("model said no")


async def slow_runner(job, job_dir):
    await asyncio.sleep(10)
    return {}


def run_job(tmp_path, runner, timeout=5):
    manager = JobManager(tmp_path, runner, timeout_s=timeout)
    job = manager.submit("alice", "a todo app")
    asyncio.run(manager.run_one(job))
    return manager, job


def test_successful_job_collects_files_and_skips_storage(tmp_path):
    manager, job = run_job(tmp_path, ok_runner)
    assert job.status == "succeeded" and job.cost_usd == 0.01
    assert job.files == ["app/main.py"]
    with zipfile.ZipFile(io.BytesIO(manager.archive(job))) as zf:
        assert zf.namelist() == ["app/main.py"]


def test_failed_and_timed_out_jobs(tmp_path):
    _, job = run_job(tmp_path / "f", failing_runner)
    assert job.status == "failed" and "model said no" in job.error
    _, job = run_job(tmp_path / "t", slow_runner, timeout=0.05)
    assert job.status == "timed_out"


def test_owner_scoping_and_validation(tmp_path):
    manager = JobManager(tmp_path, ok_runner, timeout_s=5)
    job = manager.submit("alice", "idea")
    assert manager.get(job.id, "alice") is job
    assert manager.get(job.id, "bob") is None
    assert "owner" not in job.public()
    with pytest.raises(ValueError):
        manager.submit("alice", "x" * 5000)
    with pytest.raises(ValueError):
        manager.submit("", "idea")


def test_queue_limit(tmp_path):
    manager = JobManager(tmp_path, ok_runner, timeout_s=5, max_queued=2)
    manager.submit("a", "1")
    manager.submit("a", "2")
    with pytest.raises(QueueFullError):
        manager.submit("a", "3")


def test_restart_marks_unfinished_jobs_failed(tmp_path):
    manager = JobManager(tmp_path, ok_runner, timeout_s=5)
    job = manager.submit("alice", "idea")
    reloaded = JobManager(tmp_path, ok_runner, timeout_s=5)
    again = reloaded.get(job.id, "alice")
    assert again.status == "failed" and "restarted" in again.error
    saved = json.loads((tmp_path / job.id / "job.json").read_text())
    assert saved["status"] == "failed"


def test_collect_files_missing_dir(tmp_path):
    assert collect_files(tmp_path / "nope") == []


# -- HTTP API ----------------------------------------------------------------

TOKEN = "t" * 40


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("METAGPT_WORKER_TOKEN", TOKEN)
    monkeypatch.setenv("WORKER_JOBS_DIR", str(tmp_path))
    import worker.app as app_module

    app_module = importlib.reload(app_module)
    app_module.manager.runner = ok_runner
    with TestClient(app_module.app) as c:
        yield c


AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_api_requires_token(client):
    assert client.get("/healthz").status_code == 200
    assert client.post("/jobs", json={"idea": "x", "owner": "a"}).status_code == 401
    bad = {"Authorization": "Bearer wrong"}
    assert client.post("/jobs", json={"idea": "x", "owner": "a"}, headers=bad).status_code == 401


def test_api_job_lifecycle(client):
    created = client.post("/jobs", json={"idea": "todo app", "owner": "alice"}, headers=AUTH)
    assert created.status_code == 202
    job_id = created.json()["id"]
    for _ in range(100):
        body = client.get(f"/jobs/{job_id}", params={"owner": "alice"}, headers=AUTH).json()
        if body["status"] == "succeeded":
            break
        asyncio.run(asyncio.sleep(0.02))
    assert body["status"] == "succeeded" and body["files"] == ["app/main.py"]
    assert client.get(f"/jobs/{job_id}", params={"owner": "bob"}, headers=AUTH).status_code == 404
    archive = client.get(f"/jobs/{job_id}/archive", params={"owner": "alice"}, headers=AUTH)
    assert archive.status_code == 200 and archive.headers["content-type"] == "application/zip"


def test_api_validates_input(client):
    assert client.post("/jobs", json={"idea": "", "owner": "a"}, headers=AUTH).status_code == 422
    assert client.post("/jobs", json={"idea": "x" * 1300, "owner": "a"}, headers=AUTH).status_code == 422
