"""HTTP API for the MetaGPT worker. Only the Nexora web app calls it, using a shared
bearer token; every job lookup is also scoped to the owner that created it.

Run: uvicorn worker.app:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

from worker.jobs import MAX_IDEA_CHARS, JobManager, QueueFullError
from worker.runner import subprocess_runner

logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv("METAGPT_WORKER_TOKEN", "")
if len(TOKEN) < 32:
    raise RuntimeError("METAGPT_WORKER_TOKEN must be set to a random string of at least 32 characters.")

manager = JobManager(
    root=Path(os.getenv("WORKER_JOBS_DIR", "worker-jobs")).resolve(),
    runner=subprocess_runner,
    timeout_s=float(os.getenv("METAGPT_JOB_TIMEOUT_SECONDS", "900")),
    max_queued=int(os.getenv("METAGPT_MAX_QUEUED", "5")),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.start()
    yield
    await manager.stop()


app = FastAPI(title="Nexora MetaGPT worker", lifespan=lifespan, docs_url=None, redoc_url=None)


def require_token(authorization: str = Header(default="")) -> None:
    scheme, _, supplied = authorization.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.encode(), TOKEN.encode()):
        raise HTTPException(status_code=401, detail="Invalid worker token")


class JobRequest(BaseModel):
    idea: str = Field(min_length=1, max_length=MAX_IDEA_CHARS)
    owner: str = Field(min_length=1, max_length=200)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "queued": manager.pending_count()}


@app.post("/jobs", status_code=202, dependencies=[Depends(require_token)])
def create_job(body: JobRequest) -> dict:
    try:
        job = manager.submit(body.owner, body.idea)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return job.public()


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
def get_job(job_id: str, owner: str = Query(min_length=1)) -> dict:
    job = manager.get(job_id, owner)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.public()


@app.get("/jobs/{job_id}/archive", dependencies=[Depends(require_token)])
def get_archive(job_id: str, owner: str = Query(min_length=1)) -> Response:
    job = manager.get(job_id, owner)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.files:
        raise HTTPException(status_code=409, detail="Job has no output files")
    try:
        data = manager.archive(job)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="nexora-job-{job.id[:8]}.zip"'},
    )
